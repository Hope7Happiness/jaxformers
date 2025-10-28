# reference: https://github.com/Hope7Happiness/mae/blob/main/models.py

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from utils.logging_utils import log_for_0

ModuleDef = Any

def index_array(arr, idx_arr):
    return jax.vmap(lambda obj, idx: obj[idx])(arr, idx_arr)

def get_pos_embed(length, dim, has_cls_token=False):
    side = int(length**.5)
    assert side * side == length, f"length {length} is not a perfect square"
    return get_2d_sincos_pos_embed(dim, side, cls_token=has_cls_token)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = jnp.concatenate([jnp.zeros([1, embed_dim]), pos_embed], axis=0)
    # print("pos embed:", pos_embed)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

torch_linear_layer = partial(nn.Dense, kernel_init=nn.initializers.xavier_uniform())

class Mlp(nn.Module):
    in_features: int
    hidden_features: int
    out_features: int = None

    @nn.compact
    def __call__(self, x):
        x = torch_linear_layer(self.hidden_features or self.in_features)(x)
        # x = self.act_layer(x)
        x = nn.gelu(x, approximate=False)
        x = torch_linear_layer(self.out_features or self.in_features)(x)
        return x


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False

    def setup(self):
        dim = self.dim; num_heads = self.num_heads; qkv_bias = self.qkv_bias
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    @nn.compact
    def __call__(self, x):
        B, N, C = x.shape
        qkv = torch_linear_layer(self.dim * 3, use_bias=self.qkv_bias)(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose((2, 0, 3, 1, 4))
        q, k, v = jnp.split(qkv, 3, axis=0)
      
        q = q * self.scale
        attn = q @ k.transpose((0, 1, 2, 4, 3))
        attn = jax.nn.softmax(attn, axis=-1)
        x = attn @ v

        x = x[0].transpose(0, 2, 1, 3).reshape(B, N, C)
        x = torch_linear_layer(self.dim)(x)
        return x

def drop_path(x, drop_prob: float = 0., train: bool = False, scale_by_keep: bool = True, rng: Any = None):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not train:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor = jr.bernoulli(rng, keep_prob, shape)
    if keep_prob > 0.0 and scale_by_keep:
        # random_tensor.div_(keep_prob)
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    drop_prob: float = 0.0
    scale_by_keep: bool = True
    
    def __call__(self, x, train=True, rng=None):
        return drop_path(x, self.drop_prob, train=train, scale_by_keep=self.scale_by_keep, rng=rng)

def goodsplit(rng):
    # avoid TypeError: unexpected PRNG key type <class 'NoneType'>
    if rng is None:
        return None, None
    else:
        return jr.split(rng)

IDENTITY = lambda *args, **kwargs: args[0]

class ViTBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    drop_path: float = 0.0
    
    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.attn = Attention(
            self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
        )
        self.dph1 = DropPath(self.drop_path) if self.drop_path > 0. else IDENTITY

        self.norm2 = nn.LayerNorm()
        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=int(self.dim * self.mlp_ratio),
        )
        self.dph2 = DropPath(self.drop_path) if self.drop_path > 0. else IDENTITY

    def __call__(self, x, train=True, rng=None):
        rng_no_passin = rng
        rng_no_passin, rng_used = goodsplit(rng_no_passin)
        x = x + self.dph1(self.attn(self.norm1(x)), train=train, rng=rng_used)
        del rng_used
        rng_no_passin, rng_used = goodsplit(rng_no_passin)
        x = x + self.dph2(self.mlp(self.norm2(x)), train=train, rng=rng_used)
        del rng_used
        return x

class ViTLayers(nn.Module):
    # NOTE: both ViTEncoder and ViTDecoder are pure transformer blocks, and we handle image things (e.g. patch embed & position embed) outside.

    embed_dim: int
    num_heads: int
    mlp_ratio: float
    depth: int
    
    def setup(self):
        self.blocks = nn.Sequential([
            ViTBlock(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True) for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm()

    def __call__(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x

class MAEEncoderOnly(nn.Module):
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
        

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0, imgs.shape

        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = jnp.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x

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

        return latents

MAE_B = partial(MAEEncoderOnly,
    patch_size=16,
    latent_dim=768,
    decoder_embed_dim=512,
    encoder_num_heads=12,
    decoder_num_heads=16,
    encoder_depth=12,
    decoder_depth=8,
    mlp_ratio=4.,
    image_size=224,
)


# loading utils

def load_fc(o, p):
    o['kernel'] = p['weight'].T
    o['bias'] = p['bias']

def load_norm(o, p):
    o['scale'] = p['weight']
    o['bias'] = p['bias']

def load_layer(o, p):
    load_norm(o['norm1'], p['norm1'])
    load_norm(o['norm2'], p['norm2'])
    load_fc(o['attn']['Dense_0'], p['attn']['qkv'])
    load_fc(o['attn']['Dense_1'], p['attn']['proj'])
    load_fc(o['mlp']['Dense_0'], p['mlp']['fc1'])
    load_fc(o['mlp']['Dense_1'], p['mlp']['fc2'])

class DDD(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = DDD()
        return dict.__getitem__(self, key)

from flax import core
import torch
import requests
from hashlib import md5
import os

def from_pretrained():
    if not os.path.exists('/tmp/mae_pretrain_vit_base.pth'):
        param = requests.get('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth').content
        # check md5
        md5_expected = '8cad7c'
        md5_actual = md5(param).hexdigest()
        assert md5_actual.startswith(md5_expected), f"MD5 mismatch: expected {md5_expected}, got {md5_actual}"
        with open('/tmp/mae_pretrain_vit_base.pth', 'wb') as f:
            f.write(param)
        log_for_0("Downloaded MAE pretrained weights to /tmp/mae_pretrain_vit_base.pth")
    
    param = torch.load('/tmp/mae_pretrain_vit_base.pth', map_location='cpu')['model']

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
    
    # to dict
    jax_param = core.unfreeze(core.freeze(jax_param))
    jax_param = jax.tree_util.tree_map(lambda x: x.numpy(), jax_param)
    
    n_jax_params = sum(x.size for x in jax.tree_util.tree_leaves(jax_param))
    n_torch_params = sum(x.numel() for x in param.values())
    assert n_jax_params == n_torch_params, "Parameter count mismatch between JAX and Torch models"
    
    jax_model = MAE_B()
    param_shape_expected = jax.eval_shape(jax_model.init, {'params': jr.PRNGKey(0)}, jnp.ones((1, 224, 224, 3)))['params']
    assert jax.tree_util.tree_reduce(lambda b1, b2: b1 and b2, jax.tree_map(lambda x, y: x.shape == y.shape, jax_param, param_shape_expected), True), f"Parameter shape mismatch between loaded params and expected model params: {jax.tree_map(lambda x, y: x.shape if x.shape != y.shape else None, jax_param, param_shape_expected)}"
    
    log_for_0("Pretrained MAE-B model loaded successfully.")
    
    return jax_model, {'params': jax_param}

def test_linprob():
    # NOTE{zhh}: this is just a fast demo, the hyperparams are not good.
    
    bs = 2048
    
    jax_model, jax_params = from_pretrained()
    apply_fn = lambda x: jax_model.apply(jax_params, x)[:, 0, :]  # use cls token output
    fake_batch = jax.ShapeDtypeStruct(shape=(bs, 224, 224, 3), dtype=jnp.float32)
    print("Compiling MAE apply function...")
    apply_fn = jax.jit(apply_fn).lower(fake_batch).compile()
    print("MAE apply function compiled.")
    
    data_path = "/kmh-nfs-ssd-us-mount/data/imagenet/val"
    from PIL import Image
    import optax
    import torch
    import torch.nn as tnn
    from torchvision import transforms
    import torchvision.datasets as datasets
    from torch.optim import Adam
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dl = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_path, preprocess),
        batch_size=bs, shuffle=True, num_workers=4, drop_last=True)

    # linear_layer = tnn.Linear(768, 1000)
    linear_layer = tnn.Sequential(tnn.BatchNorm1d(768, affine=False), tnn.Linear(768, 1000))
    linear_layer.train()
    linear_opt = Adam(linear_layer.parameters(), lr=0.01)

    # train
    from tqdm import tqdm
    class Avger(list):
        def __str__(self):
            return f'{sum(self)/len(self):.4f}' if self else 'N/A'
    
    print("Starting linear probing training...")
    for epoch in range(5):
        loss_avger = Avger()
        acc_avger = Avger()
        with tqdm(enumerate(dl), total=len(dl)) as bar:
            for i, (images, labels) in bar:
                images = images.permute(0, 2, 3, 1).numpy()  # to (B, H, W, C)
                labels = labels.numpy()
                feats = apply_fn(images)  # (B, 768)
                
                # to torch
                feats = torch.from_numpy(np.array(feats))
                labels = torch.from_numpy(labels)

                logits = linear_layer(feats)
                loss = tnn.functional.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean().item()

                linear_opt.zero_grad()
                loss.backward()
                linear_opt.step()
                
                loss_avger.append(loss.item())
                acc_avger.append(acc)
                bar.set_description(f"Epoch {epoch} Loss {loss_avger} Acc {acc_avger}")
                if float(str(acc_avger)) > 0.5:
                    print("Achieved over 50% accuracy, success!")
                    bar.close()
                    exit(0)

if __name__ == '__main__':
    # model, params = from_pretrained()
    test_linprob()