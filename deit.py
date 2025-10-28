import jax
import jax.numpy as jnp
from flax import linen as nn
from models.mae import ViTLayers, load_fc, load_norm, load_layer, DDD
from models.resnet import test_model_forward
from typing import Any
import torch
from functools import partial
from utils.logging_utils import log_for_0

class DeiT(nn.Module):
    patch_size: int
    latent_dim: int
    decoder_embed_dim: int
    encoder_num_heads: int
    decoder_num_heads: int
    encoder_depth: int
    decoder_depth: int
    mlp_ratio: float = 4.
    mask_ratio: float = 0.75
    image_size: int = 224
    dtype: Any = jnp.float32

    def setup(self):
        assert self.dtype == jnp.float32
        self.encoder = ViTLayers(self.latent_dim, self.encoder_num_heads, self.mlp_ratio, self.encoder_depth, name="encoder")
        length = (self.image_size // self.patch_size)**2

        # cls_token and mask_token
        self.cls_token = self.param('cls_token', nn.initializers.normal(stddev=0.02), (self.latent_dim,))

        # encoder: positional embed, patch embed
        # self.encoder_patch_embed = torch_linear_layer(self.latent_dim)
        self.encoder_patch_embed = nn.Conv(features=self.latent_dim, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))
        # self.encoder_pos_embed = lambda: get_pos_embed(length, self.latent_dim, has_cls_token=True)
        self.encoder_pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02), (length + 1, self.latent_dim))
        
        # classifier head
        self.head = nn.Dense(features=1000)  # assuming 1000 classes for ImageNet
        
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = jnp.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs

    def __call__(self, image, train=False): # MAE has no dropout
        # train is useless -- only for interface consistency
        
        # image.shape: (B, H, W, 3)
        # image = image.transpose((0, 3, 1, 2)) # (B, 3, H, W)
        assert image.shape[1] == image.shape[2] and image.shape[1] == self.image_size, image.shape

        # x = self.patchify(image)
        x = self.encoder_patch_embed(image)  # (B, H/P, W/P, latent_dim)
        x = x.reshape(x.shape[0], -1, x.shape[3])  # (B, L, latent_dim)
        B, L, D = x.shape
        assert L + 1 == self.encoder_pos_embed.shape[0], "pos embed shape mismatch: {} vs {}".format(L + 1, self.encoder_pos_embed.shape[0])

        ### Encoder ###

        # encoder pre-process
        h = x + self.encoder_pos_embed[None, 1:, :] # [B, L, H_enc]

        # mask
        encoder_input = h
        
        # add cls token
        cls_token = self.cls_token.reshape(1,1,-1).repeat(B, axis=0) + self.encoder_pos_embed[None, :1, :]  # [B, 1, H_enc]
        h = jnp.concatenate([cls_token, encoder_input], axis=1) # [B, remained+1, H_enc]
        latents = self.encoder(h) # [B, remained, H_enc]
        latents = latents[:, 0]  # cls token
        latents = self.head(latents)

        return latents

DeiT_B = partial(DeiT,
    patch_size=16,
    latent_dim=768,
    decoder_embed_dim=512,
    encoder_num_heads=12,
    decoder_num_heads=16,
    encoder_depth=12,
    decoder_depth=8,
)

from flax import core
import torch
import os

def from_pretrained():
    param = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        map_location="cpu", check_hash=True
    )['model']
    
    tree_param = DDD()

    for k, v in param.items():
        o = tree_param
        parts = k.split('.')
        for p in parts[:-1]:
            o = o[p]
        o[parts[-1]] = v
    tree_param = core.unfreeze(core.freeze(tree_param))
    
    jax_param = DDD()

    # encoder patch embed
    # load_fc(jax_param['encoder_patch_embed'], tree_param['patch_embed']['proj'])
    jax_param['encoder_patch_embed']['kernel'] = tree_param['patch_embed']['proj']['weight'].permute(2, 3, 1, 0)
    jax_param['encoder_patch_embed']['bias'] = tree_param['patch_embed']['proj']['bias']
    jax_param['cls_token'] = tree_param['cls_token'][0, 0, :]
    jax_param['pos_embed'] = tree_param['pos_embed'][0, :, :]
    
    # encoder blocks
    for i in range(12):
        load_layer(jax_param['encoder']['blocks'][f'layers_{i}'], tree_param['blocks'][f'{i}'])
    # encoder norm
    load_norm(jax_param['encoder']['norm'], tree_param['norm'])
    # head
    load_fc(jax_param['head'], tree_param['head'])
    
    # to dict
    jax_param = core.unfreeze(core.freeze(jax_param))
    jax_param = jax.tree_util.tree_map(lambda x: x.numpy(), jax_param)
    
    n_jax_params = sum(x.size for x in jax.tree_util.tree_leaves(jax_param))
    n_torch_params = sum(x.numel() for x in param.values())
    assert n_jax_params == n_torch_params, "Parameter count mismatch between JAX and Torch models"

    jax_model = DeiT_B()
    param_shape_expected = jax.eval_shape(jax_model.init, {'params': jax.random.PRNGKey(0)}, jnp.ones((1, 224, 224, 3)))['params']
    assert jax.tree_util.tree_reduce(lambda b1, b2: b1 and b2, jax.tree_map(lambda x, y: x.shape == y.shape, jax_param, param_shape_expected), True), f"Parameter shape mismatch between loaded params and expected model params: {jax.tree_map(lambda x, y: x.shape if x.shape != y.shape else None, jax_param, param_shape_expected)}"
    
    log_for_0("Pretrained DeiT-B model loaded successfully.")
    
    return jax_model, {'params': jax_param}

if __name__ == "__main__":
    # test model forward
    model, params = from_pretrained()
    apply_fn = lambda x: model.apply(params, x, train=False)
    test_model_forward(apply_fn)