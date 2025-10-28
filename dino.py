from time import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import torch
import re
import functools
from typing import Type
import tempfile
import os
import pickle
from utils.logging_utils import log_for_0
from flax import jax_utils
import ml_collections
from models.torch_models import TorchLinear

IMAGE_SIZE = 518 # DINOv2 default image size
IMGNET_IMAGE_SIZE = 224 # should be multiple of 14. Center crop from 256.
HIDDEN_DIM = 768 # DINOv2 vit-b hidden dim

class REPAMLP(nn.Module):
    in_features: int = 256
    out_features: int = HIDDEN_DIM
    hidden_features: int = 2048
    act_layer: nn.Module = nn.silu
    bias: bool = True
    
    def setup(self):
        self.ln = nn.LayerNorm(reduction_axes=-1)
        self.fc1 = TorchLinear(in_features=self.in_features, out_features=self.hidden_features, bias=self.bias)
        self.fc2 = TorchLinear(in_features=self.hidden_features, out_features=self.hidden_features, bias=self.bias)
        self.fc3 = TorchLinear(in_features=self.hidden_features, out_features=self.out_features, bias=self.bias)
        self.act = self.act_layer
    
    def __call__(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class DinoMLP(nn.Module):
    hidden_features: int = 1536
    out_features: int = HIDDEN_DIM
    act_layer: nn.Module = nn.gelu
    dropout_rate: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(features=self.hidden_features, use_bias=self.bias, name="fc1")(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=self.dropout_rate, name="drop1")(
            x, deterministic=not training
        )
        x = nn.Dense(features=self.out_features, use_bias=self.bias, name="fc2")(x)
        x = nn.Dropout(rate=self.dropout_rate, name="drop2")(
            x, deterministic=not training
        )
        return x
    
class DinoAtt(nn.Module):
    num_heads: int = 8
    attn_bias: bool = True
    attn_drop_rate: float = 0.0
    proj_bias: bool = True
    proj_drop_rate: float = 0.0
    embed_dim: int = HIDDEN_DIM

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, N, C = x.shape
        assert (
            C == self.embed_dim
        ), f"Input embedding dimension ({C}) should match layer embedding dimension ({self.embed_dim})."
        qkv = nn.Dense(features=3 * C, use_bias=self.attn_bias, name="qkv")(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)

        # Attention matrix: (B, H, N, N)
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(C // self.num_heads)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop_rate, name="attn_drop")(
            attn, deterministic=not training
        )

        # Output: (B, N, H, C // H)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)

        x = nn.Dense(features=C, use_bias=self.proj_bias, name="proj")(x)
        x = nn.Dropout(rate=self.proj_drop_rate, name="proj_drop")(
            x, deterministic=not training
        )

        return x

class DinoLayerScale(nn.Module):
    initial_value: float = 1.0

    @nn.compact
    def __call__(self, x):
        gamma = self.param(
            "gamma",
            lambda _, shape: self.initial_value * jnp.ones(shape),
            (x.shape[-1],),
        )
        return x * gamma


class DinoDropPath(nn.Module):
    rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if self.rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.rate
            shape = (x.shape[0], 1, 1, 1)
            random_tensor = jax.random.bernoulli(
                self.make_rng("dropout"), keep_prob, shape=shape
            )
            return x / keep_prob * random_tensor
        else:
            return x


class DinoBlock(nn.Module):
    num_heads: int = 6
    embed_dim: int = HIDDEN_DIM
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0

    AttentionClass: Type[nn.Module] = DinoAtt
    FfnClass: Type[nn.Module] = DinoMLP

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        def attn_residual_func(x: jnp.ndarray) -> jnp.ndarray:
            x = nn.LayerNorm(name="norm1")(x)
            x = self.AttentionClass(
                num_heads=self.num_heads, embed_dim=self.embed_dim, name="attn"
            )(x, training=training)
            x = DinoLayerScale(name="ls1")(x)
            return x

        def ffn_residual_func(x: jnp.ndarray) -> jnp.ndarray:
            x = nn.LayerNorm(name="norm2")(x)
            x = self.FfnClass(
                hidden_features=int(self.mlp_ratio * self.embed_dim),
                out_features=self.embed_dim,
                name="mlp",
            )(x, training=training)
            x = DinoLayerScale(name="ls2")(x)
            return x
        
        if training:
            x = x + DinoDropPath(
                rate=self.drop_path_rate, name="drop_path1", deterministic=not training
            )(attn_residual_func(x))
            x = x + DinoDropPath(
                rate=self.drop_path_rate, name="drop_path2", deterministic=not training
            )(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)

        return x

class DinoPatchEmbed(nn.Module):
    img_size: int = IMAGE_SIZE
    patch_size: int = 14
    in_channels: int = 3
    embed_dim: int = HIDDEN_DIM
    norm_layer: Type[nn.Module] = None
    flatten_embedding: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, H, W, C = x.shape
        patch_H, patch_W = self.patch_size, self.patch_size
        assert (
            H % patch_H == 0 and W % patch_W == 0
        ), f"Image size ({H}*{W}) cannot be evenly divided by patch size ({patch_H}*{patch_W})."

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(patch_H, patch_W),
            strides=(patch_H, patch_W),
            name="proj",
            padding="VALID",
        )(x)

        _, H, W, _ = x.shape
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.norm_layer is not None:
            x = self.norm_layer(name="norm")(x)

        if not self.flatten_embedding:
            x = jnp.reshape(x, (-1, H, W, self.embed_dim))

        return x
    
class DinoViT(nn.Module):
    img_size: int = IMAGE_SIZE
    in_channels: int = 3

    patch_size: int = 14
    embed_dim: int = HIDDEN_DIM

    depth: int = 12

    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    
    num_register_tokens: int = 0

    BlockClass: Type[nn.Module] = DinoBlock
    AttentionClass: Type[nn.Module] = DinoAtt
    FfnClass: Type[nn.Module] = DinoMLP
    EmbedLayer: Type[nn.Module] = DinoPatchEmbed

    def _interpolate_pos_encoding(
        self, x: jnp.ndarray, w: int, h: int, pos_embed: jnp.ndarray
    ):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed

        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = jax.image.resize(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim),
            (1, w0, h0, dim),
            method="bicubic",
        )
        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, -1, dim))

        return jnp.concatenate((class_pos_embed[None], patch_pos_embed), axis=1).astype(
            previous_dtype
        )

    @nn.compact
    def __call__(self, x, train: bool = False, for_linear_probe: bool = False):
        # assert not training, "DINO model only for inference, but got training={}".format(training)
        B, H, W, C = x.shape
        assert H == W == self.img_size, "x size must be (B, {}, {}, {})".format(
            self.img_size, self.img_size, C
        )

        x = self.EmbedLayer(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            name="patch_embed",
        )(x)
        cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.embed_dim)
        )
        cls_token = jnp.broadcast_to(cls_token, (x.shape[0], *cls_token.shape[1:]))
        x = jnp.concatenate((cls_token, x), axis=1)

        num_patches = (self.img_size // self.patch_size) ** 2
        num_tokens = 1

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, num_patches + num_tokens, self.embed_dim),
        )
        x = x + self._interpolate_pos_encoding(
            x, self.img_size, self.img_size, pos_embed
        )
        
        if self.num_register_tokens > 0:
            register_tokens = self.param(
                "register_tokens",
                nn.initializers.zeros,
                (1, self.num_register_tokens, self.embed_dim),
            )
            register_tokens = jnp.repeat(register_tokens, repeats=B, axis=0)
            x = jnp.concatenate(
                # cls token + register tokens + patch tokens
                (x[:, :1, :], register_tokens, x[:, 1:, :]), axis=1
            )
        
        for i in range(self.depth):
            x = self.BlockClass(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                drop_path_rate=self.drop_path_rate,
                AttentionClass=self.AttentionClass,
                FfnClass=self.FfnClass,
                name=f"blocks.{i}",
            )(x, training=train)

        x_norm = nn.LayerNorm(name="norm")(x)
        
        if for_linear_probe:
            return x_norm[:, 0]  # return cls token for linear probe
        # only return x_norm_patchtokens here for REPA
        return x_norm[:, (self.num_register_tokens + 1):]
    
class LinearProbeHead(nn.Module):
    num_classes: int = 1000
    in_features: int = HIDDEN_DIM
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        x = TorchLinear(
            in_features=self.in_features, out_features=self.num_classes, bias=self.bias, name="head"
        )(x)
        return x

ViT_B_imgnet = functools.partial(
    DinoViT,
    patch_size=14,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    img_size=IMGNET_IMAGE_SIZE,
)

ViT_B = functools.partial(
    DinoViT,
    patch_size=14,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    img_size=IMAGE_SIZE,
)

ViT_B_reg_imgnet = functools.partial(
    DinoViT,
    patch_size=14,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    img_size=IMGNET_IMAGE_SIZE,
    num_register_tokens=4,   
)

ViT_B_reg = functools.partial(
    DinoViT,
    patch_size=14,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    img_size=IMAGE_SIZE,
    num_register_tokens=4,   
)

ViT_S_reg_imgnet = functools.partial(
    DinoViT,
    patch_size=14,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    img_size=IMGNET_IMAGE_SIZE,
    num_register_tokens=4,   
)

ViT_S_reg = functools.partial(
    DinoViT,
    patch_size=14,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    img_size=IMAGE_SIZE,
    num_register_tokens=4,   
)
    
def convert_weights_to_jax(jax_params: dict, module_pt: torch.nn.Module):
    log_for_0("Converting DINOv2 weights to jax...", logging_fn=print)
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(jax_params)
    pt_params = {path: param for path, param in module_pt.named_parameters()}

    direct_copy = {"cls_token", "pos_embed", "register_tokens"}
    ignore_keys = {"mask_token"}
    pt_params_flat = []
    for path, param in jax_params_flat:
        shape = param.shape
        path = ".".join([p.key for p in path])
        path = re.sub(r"\.scale|.kernel", ".weight", path)
        if path in pt_params:
            pt_param = pt_params[path]
            if path in direct_copy:
                if shape != pt_param.shape:
                    log_for_0(f"[WARNING] shape mismatch for direct copy: expect param '{path}' have shape {shape}, but got {pt_param.shape}", logging_fn=print)
                pt_params_flat.append(jnp.asarray(pt_param.detach().numpy()))
            else:
                if len(shape) == 4:
                    pt_param = torch.permute(pt_param, (2, 3, 1, 0))
                else:
                    pt_param = torch.permute(
                        pt_param, tuple(reversed(range(len(shape))))
                    )
                if shape != pt_param.shape:
                    log_for_0(f"[WARNING] shape mismatch after transpose: expect param '{path}' have shape {shape}, but got {pt_param.shape}", logging_fn=print)
                pt_params_flat.append(jnp.asarray(pt_param.detach().numpy()))
            log_for_0(f"  Loaded param '{path}' with shape {shape}", logging_fn=print)
            pt_params.pop(path)
        else:
            log_for_0(f"[WARNING] missing param '{path}' with shape {shape} from PyTorch model", logging_fn=print)
            pt_params_flat.append(None)
            
    for path, param in pt_params.items():
        if path in ignore_keys:
            continue
        log_for_0(f"[WARNING] params not loaded '{path}' with shape {param.shape} from PyTorch model", logging_fn=print)

    log_for_0("DINOv2 conversion done.", logging_fn=print)

    return jax.tree_util.tree_unflatten(jax_param_pytree, pt_params_flat)

def load_dino(config: ml_collections.ConfigDict):
    if config.vit_type == "vit-b":
        vit_cls = ViT_B
        vit_cls_imgnet = ViT_B_imgnet
        cache_ckpt_name = "dinov2_vit-b-s14.pkl"
        hub_name = "dinov2_vitb14"
    elif config.vit_type == "vit-s-reg":
        vit_cls = ViT_S_reg
        vit_cls_imgnet = ViT_S_reg_imgnet
        cache_ckpt_name = "dinov2_vit-s-reg-14.pkl"
        hub_name = "dinov2_vits14_reg"
    elif config.vit_type == "vit-b-reg":
        vit_cls = ViT_B_reg
        vit_cls_imgnet = ViT_B_reg_imgnet
        cache_ckpt_name = "dinov2_vit-b-reg-14.pkl"
        hub_name = "dinov2_vitb14_reg"
    else:
        raise NotImplementedError(f"Unknown vit_type {config.vit_type}")
    
    vit_def = vit_cls()
    hidden_dim = vit_def.embed_dim
    vit_params = vit_def.init(jax.random.PRNGKey(0), jnp.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3)))["params"]
    
    ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, "jax_dinov2")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(
        ckpt_dir, cache_ckpt_name
    )
    
    log_for_0(f"Loading DINO-ViT from {ckpt_file}", logging_fn=print)
    if not os.path.exists(ckpt_file):
        log_for_0("Downloading DINO-ViT weights...", logging_fn=print)
        dinov2_model = torch.hub.load("facebookresearch/dinov2", hub_name)
        params = convert_weights_to_jax(vit_params, dinov2_model)
        with open(ckpt_file, 'wb') as f:
            pickle.dump(params, f)
    else:
        log_for_0("Cached DINO-ViT weights found, loading...", logging_fn=print)
        with open(ckpt_file, 'rb') as f:
            params = pickle.load(f)
            
    log_for_0("Number of parameters: {}".format(sum(x.size for x in jax.tree_util.tree_leaves(params))), logging_fn=print)
    
    vit_def_imgnet = vit_cls_imgnet()
    _ = vit_def_imgnet.init(jax.random.PRNGKey(0), jnp.ones((1, IMGNET_IMAGE_SIZE, IMGNET_IMAGE_SIZE, 3)))["params"]
    
    # load params into the 224 model
    # also use params, but change pos_embed shape: interpolate pos_embed to match 224 size (1, 1370, hidden) -> (1, 257, hidden)
    pos_embed = params["pos_embed"]
    pos_embed_wo_class = pos_embed[:, 1:, :].reshape(1, 37, 37, -1)  # (1, 37, 37, hidden)
    pos_embed_224 = jax.image.scale_and_translate(
        pos_embed_wo_class,
        shape=(1, 16, 16, hidden_dim),  # target shape for 224px image
        spatial_dims=(1, 2),  # resize spatial dimensions
        scale=jnp.array([16. / 37., 16. / 37.]),  # scale factors for height and width
        translation=jnp.array([0.0, 0.0]),
        method=jax.image.ResizeMethod.CUBIC,
    )
    params["pos_embed"] = jnp.concatenate(
        (pos_embed[:, :1, :], pos_embed_224.reshape(1, -1, hidden_dim)), axis=1
    )  # (1, 257, hidden)
    log_for_0("Params for image size 224*224 loaded, with pos_embed shape: {}".format(params["pos_embed"].shape), logging_fn=print)
    
    log_for_0("Compiling DINO-ViT for image size 224*224...", logging_fn=print)
    apply_fn = jax.jit(vit_def_imgnet.apply, static_argnums=(2, 3))
    _ = apply_fn({"params": params}, jnp.ones((1, IMGNET_IMAGE_SIZE, IMGNET_IMAGE_SIZE, 3)))
    log_for_0("DINO-ViT compilation done.", logging_fn=print)
    
    # testing speed
    # import time
    # x = jnp.ones((1, IMGNET_IMAGE_SIZE, IMGNET_IMAGE_SIZE, 3))
    # for _ in range(10):
    #     _ = apply_fn({"params": params}, x, False).block_until_ready()
    # t0 = time.time()
    # for times in range(100):
    #     _ = apply_fn({"params": params}, x, False).block_until_ready()
    #     if times % 10 == 0:
    #         print(f"DINO-ViT forward pass iteration {times} done.")
    # t1 = time.time()
    # print("DINO-ViT forward pass time for image size 224*224: {:.2f} ms".format((t1 - t0) * 10))
    # w/ JIT: 3.55 ms
    # w/o JIT: 519.15 ms
    
    return (apply_fn, params, vit_def_imgnet) # apply function, parameters, model definition

def test_dino_224_imgnet():
    # load two dissimilar images
    # apply two augmentations: random crop + horizontal flip
    # check that the embeddings are different
    path = "/kmh-nfs-ssd-eu-mount/code/xianbang/samples_imgnet"
    from PIL import Image
    img1 = Image.open(os.path.join(path, "1.JPEG")).convert("RGB")
    img2 = Image.open(os.path.join(path, "2.JPEG")).convert("RGB") # different image
    # img2 = img1 # same image
    import torchvision.transforms as T
    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    timg1 = transform(img1).unsqueeze(0).numpy()
    timg2 = transform(img2).unsqueeze(0).numpy()
    image = np.concatenate([timg1, timg2], axis=0) # (2, 3, 224, 224)
    assert image.shape == (2, 3, 224, 224), f"Image shape incorrect: {image.shape}"
    image = np.transpose(image, (0, 2, 3, 1)) # (2, 224, 224, 3)
    
    config: ml_collections.ConfigDict = ml_collections.ConfigDict()
    config.vit_type = "vit-b"
    jax_apply_fn, jax_params = load_dino(config)

    embed_jax = jax_apply_fn({"params": jax_params}, image, False)
    print("x_norm_patchtokens shape:", embed_jax.shape) # (2, 256, H)
    
    emb_1 = embed_jax[0].reshape(-1) # (256*H)
    emb_2 = embed_jax[1].reshape(-1) # (256*H)
    cos_sim = jnp.dot(emb_1, emb_2) / (jnp.linalg.norm(emb_1) * jnp.linalg.norm(emb_2))
    print("Cosine similarity between two different images:", cos_sim)
    
def train_linear_probe(vit_type="vit-s-reg"):
    # load dataloader
    import input_pipeline
    from input_pipeline import prepare_batch_data
    
    global_batch_size = 512
    assert global_batch_size % jax.process_count() == 0, f"Global batch size must be divisible by number of processes, but got {global_batch_size} and {jax.process_count()}"
    local_batch_size = global_batch_size // jax.process_count()
    assert local_batch_size % jax.local_device_count() == 0, f"Local batch size must be divisible by number of devices, but got {local_batch_size} and {jax.local_device_count()}"
    print(f"Global batch size: {global_batch_size}, local batch size: {local_batch_size}, number of local devices: {jax.local_device_count()}")
    
    dataset_config = ml_collections.ConfigDict()
    dataset_config.name = "imgnet_latent"
    dataset_config.root = '/kmh-nfs-ssd-us-mount/data/vae_cached_muvar_imagenet_zhh/'
    dataset_config.num_workers = 8
    dataset_config.prefetch_factor = 4
    dataset_config.pin_memory = False
    dataset_config.cache = False
    dataset_config.image_size = 256
    dataset_config.num_classes = 1000
    dataset_config.vae = "mse"
    dataset_config.also_img = True
    
    train_loader, steps_per_epoch = input_pipeline.create_split(
        dataset_config,
        local_batch_size,
        split="train",
    )
    
    print("steps_per_epoch: {}".format(steps_per_epoch))
    
    # load dino
    config: ml_collections.ConfigDict = ml_collections.ConfigDict()
    config.vit_type = vit_type
    jax_apply_fn, jax_params, jax_model = load_dino(config)
    hidden_dim = jax_model.embed_dim
    
    # load linear probe head
    head_def = LinearProbeHead(num_classes=1000, in_features=hidden_dim)
    head_params = head_def.init(jax.random.PRNGKey(0), jnp.ones((1, hidden_dim)))["params"]
    print("linear probe head initialized.")
    
    # optimizer
    import optax
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 10
    
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=True,)
    opt_state = optimizer.init(head_params)
    print("optimizer initialized.")
    
    apply_fn = jax.jit(head_def.apply)
    _ = apply_fn({"params": head_params}, jnp.ones((1, hidden_dim)))
    print("head compiled.")
    
    state = {
        "head_params": head_params,
        "opt_state": opt_state,
    }

    state = jax_utils.replicate(state)
    
    def train_step(state, batch):
        images, labels = batch["aux"][0], batch["label"]
        images = jax.lax.stop_gradient(jax_apply_fn({"params": jax_params}, images, False, True))
        one_hot_labels = jax.nn.one_hot(labels, num_classes=1000)
        def loss_fn(head_params):
            logits = apply_fn({'params': head_params}, images)
            # test acc
            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.mean(preds == labels)
            loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
            return loss, acc

        value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, acc), grads = value_and_grad_fn(state["head_params"])
        loss = jax.lax.pmean(loss, axis_name='batch')
        acc = jax.lax.pmean(acc, axis_name='batch')
        grads = jax.lax.pmean(grads, axis_name='batch')
        updates, opt_state = optimizer.update(grads, state["opt_state"], state["head_params"])
        head_params = optax.apply_updates(state["head_params"], updates)
        state = {"head_params": head_params, "opt_state": opt_state}
        return state, loss, acc
    
    p_train_step = jax.pmap(train_step, axis_name='batch')

    def eval_step(state, batch):
        head_params = state["head_params"]
        images, labels = batch["aux"][0], batch["label"]
        logits = apply_fn({'params': head_params}, images)
        preds = jnp.argmax(logits, axis=-1)
        return preds
    
    p_eval_step = jax.pmap(eval_step, axis_name='batch')
    
    # training loop
    print("Starting linear probe training...")
    timer = time()
    for epoch in range(num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        for n_batch, batch in enumerate(train_loader): 
            if n_batch == steps_per_epoch - 1: # drop last incomplete batch
                break
            batch = prepare_batch_data(batch)
            state, loss, acc = p_train_step(state, batch)
            if n_batch % 10 == 0:
                # Unreplicate to get scalar values
                loss_scalar = jax.device_get(loss)[0]
                acc_scalar = jax.device_get(acc)[0]
                
                # for vit-b: expected > 80% after 1000ish iters
                print(f"Epoch {epoch}, Iter {n_batch} done. Loss: {loss_scalar:.4f}, Acc: {acc_scalar:.4f}, Time elapsed: {time() - timer:.2f} s")
        # evaluation
        total = 0
        correct = 0
        for n_batch, batch in enumerate(train_loader):
            if n_batch >= 100: # only eval 100 batches
                break
            batch = prepare_batch_data(batch)
            preds = p_eval_step(state, batch)
            preds = np.array(jax.device_get(preds)).reshape(-1)
            batch_labels = np.array(jax.device_get(batch["label"])).reshape(-1)
            total += len(batch_labels)
            correct += (preds == batch_labels).sum()
        acc = correct / total
        print(f"Epoch {epoch} done. Validation accuracy: {acc*100:.2f}% over {total} samples.")
    
if __name__ == "__main__":
    # test_dino_224_imgnet()
    train_linear_probe(vit_type="vit-b-reg")