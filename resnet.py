# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1.5."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random
from flax import core

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1.5."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(self.dtype)

        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            axis_name="batch",
        )

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock) # 25M
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock) # 44.5M
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)
ResNet500 = partial(
    ResNet, stage_sizes=[3, 124, 36, 3], block_cls=BottleneckResNetBlock
)
ResNet800 = partial(
    ResNet, stage_sizes=[3, 124, 136, 3], block_cls=BottleneckResNetBlock
)


ResNet18Local = partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, conv=nn.ConvLocal
)


# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
_ResNet1Local = partial(
    ResNet, stage_sizes=[1], block_cls=ResNetBlock, conv=nn.ConvLocal
)

def test_speed():
    model = ResNet101(num_classes=1000)
    rng = random.PRNGKey(0)
    input_shape = (1, 224, 224, 3)
    variables = model.init(rng, jnp.ones(input_shape), train=False)
    print("Model initialized.")
    print("Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(variables['params'])))

    x = jnp.ones(input_shape)
    import time
    start = time.time()
    num_iters = 10
    for _ in range(num_iters):
        y = model.apply(variables, x, train=False)
    end = time.time()
    print(f"Average inference time over {num_iters} iterations: {(end - start) / num_iters:.6f} seconds")
    
def load_resnet_model(model_name: str = "resnet101"):
    assert model_name == "resnet101", "Only resnet101 is supported in this loader."
    ckpt_path = "gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files/resnet_jax/launch_20251025_021823_git6f0ce22_4698297d/logs/log1_20251025_021954_VMkmh-tpuvm-v6e-32-kangyang-1_Zus-central1-b_41822faa"
    model = ResNet101(num_classes=1000)

    # load ckpt
    from flax.training import checkpoints
    state = checkpoints.restore_checkpoint(ckpt_path, target=None)
    variables = {'params': state['params'], 'batch_stats': state['batch_stats']}
    
    return (model, variables)

def test_model_forward(apply_fn):
    img_path = "/kmh-nfs-ssd-us-mount/code/xianbang/samples_imgnet/1.JPEG"
    from PIL import Image
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
    
    input = jnp.asarray(img_tensor.numpy())
    input = input.transpose((0, 2, 3, 1))  # (1, 3, 224, 224) -> (1, 224, 224, 3)
    print("Input shape:", input.shape)
    output: jnp.ndarray = apply_fn(input) # (1, 1000)
    output = jax.nn.softmax(output, axis=-1)
    print("Output shape:", output.shape)
    print("output probability:", output.max(), "on position", output.argmax())
    
def test_load_model_and_forward():
    model, variables = load_resnet_model("resnet101")
    print("Model and variables loaded.")
    print("Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(variables['params'])))
    print("Key names in variables:", variables['params'].keys())
    apply_fn = lambda x: model.apply(variables, x, train=False)
    test_model_forward(apply_fn)
    
def from_official_pretrained(cls_name: str = "resnet101"):
    SUF = 'V1'
    if cls_name.endswith('v2'):
        cls_name = cls_name[:-2]
        SUF = 'V2'
    
    import torchvision
    if cls_name == "resnet101":
        model = torchvision.models.resnet101(weights=f'IMAGENET1K_{SUF}')
        stage_sizes = [3, 4, 23, 3]
    elif cls_name == "resnet152":
        model = torchvision.models.resnet152(weights=f'IMAGENET1K_{SUF}')
        stage_sizes = [3, 8, 36, 3]
    else:
        raise NotImplementedError(f"cls_name {cls_name} not implemented.")
    param = model.state_dict()

    # step1: convert a.b.c.d to a -> b -> c -> d
    
    # helper
    class DDD(dict):
        def __getitem__(self, k):
            if k not in self:
                self[k] = DDD()
            return dict.__getitem__(self, k)

    tree_param = DDD()

    for k, v in param.items():
        o = tree_param
        parts = k.split('.')
        for p in parts[:-1]:
            o = o[p]
        o[parts[-1]] = v
    tree_param = core.unfreeze(core.freeze(tree_param))
        
    # step 2: load 
    def load_conv(op, ob, p):
        op['kernel'] = p['weight'].numpy().transpose((2, 3, 1, 0))
        if 'bias' in p:
            op['bias'] = p['bias'].numpy()

    def load_bn(op, ob, p):
        op['scale'] = p['weight'].numpy()
        op['bias'] = p['bias'].numpy()
        ob['mean'] = p['running_mean'].numpy()
        ob['var'] = p['running_var'].numpy()
        
    def load_block(op, ob, p):
        load_conv(op['Conv_0'], None, p['conv1'])
        load_bn(op['BatchNorm_0'], ob['BatchNorm_0'], p['bn1'])
        load_conv(op['Conv_1'], None, p['conv2'])
        load_bn(op['BatchNorm_1'], ob['BatchNorm_1'], p['bn2'])
        load_conv(op['Conv_2'], None, p['conv3'])
        load_bn(op['BatchNorm_2'], ob['BatchNorm_2'], p['bn3'])
        if 'downsample' in p:
            load_conv(op['conv_proj'], None, p['downsample']['0'])
            load_bn(op['norm_proj'], ob['norm_proj'], p['downsample']['1'])

    def load_resnet(op, ob, p, stage_sizes=()):
        load_conv(op['conv_init'], None, p['conv1'])
        load_bn(op['bn_init'], ob['bn_init'], p['bn1'])
        # stages
        cnt = 0
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                block_p = p[f'layer{i+1}'][f'{j}']
                block_op = op[f'BottleneckResNetBlock_{cnt}']
                block_ob = ob[f'BottleneckResNetBlock_{cnt}']
                load_block(block_op, block_ob, block_p)
                cnt += 1
        # fc
        op['Dense_0']['kernel'] = p['fc']['weight'].numpy().transpose((1, 0))
        op['Dense_0']['bias'] = p['fc']['bias'].numpy()
        
    param_jax = DDD()
    batch_stat_jax = DDD()
    load_resnet(param_jax, batch_stat_jax, tree_param, stage_sizes=stage_sizes)
    param_jax = core.unfreeze(core.freeze(param_jax))
    batch_stat_jax = core.unfreeze(core.freeze(batch_stat_jax))
    return ResNet(stage_sizes=stage_sizes, block_cls=BottleneckResNetBlock, num_classes=1000), {'params': param_jax, 'batch_stats': batch_stat_jax}

def test_from_official(cls_name: str = "resnet101"):
    if cls_name == "resnet101":
        stage_sizes = [3, 4, 23, 3]
    elif cls_name == "resnet152":
        stage_sizes = [3, 8, 36, 3]
    
    jax_model, variables = from_official_pretrained(cls_name)
    param_jax = variables['params']
    batch_stat_jax = variables['batch_stats']
    
    n_params_loaded = sum(x.size for x in jax.tree_util.tree_leaves(param_jax)) + sum(x.size for x in jax.tree_util.tree_leaves(batch_stat_jax))
    
    variables = jax.eval_shape(jax_model.init, jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)))
    param2 = variables['params']
    batch_stat2 = variables['batch_stats']
    n_params_official = sum(x.size for x in jax.tree_util.tree_leaves(param2)) + sum(x.size for x in jax.tree_util.tree_leaves(batch_stat2))
    print(f'Number of parameters loaded: {n_params_loaded}, official: {n_params_official}')
    assert jax.tree_util.tree_structure(param_jax) == jax.tree_util.tree_structure(param2), "Parameter tree structure mismatch!" and print(jax.tree_util.tree_map(lambda x: x.shape, param_jax), '\n', jax.tree_util.tree_map(lambda x: x.shape, param2))
    assert jax.tree_util.tree_structure(batch_stat_jax) == jax.tree_util.tree_structure(batch_stat2), "Batch stat tree structure mismatch!" and print(jax.tree_util.tree_map(lambda x: x.shape, batch_stat_jax), '\n', jax.tree_util.tree_map(lambda x: x.shape, batch_stat2))
    n_params_official = sum(x.size for x in jax.tree_util.tree_leaves(param2)) + sum(x.size for x in jax.tree_util.tree_leaves(batch_stat2))
    
    model = ResNet(stage_sizes=stage_sizes, block_cls=BottleneckResNetBlock, num_classes=1000)
    apply_fn = lambda x: model.apply({'params': param_jax, 'batch_stats': batch_stat_jax}, x, train=False)
    apply_fn = jax.jit(apply_fn)
    test_model_forward(apply_fn)
    
if __name__ == "__main__":
    # test_speed()
    # test_load_model_and_forward()
    # from_official_pretrained()
    # _, v = load_resnet_model("resnet101")
    # print(jax.tree_util.tree_map(lambda x: x.shape, v['params']))
    # from_official_pretrained('resnet152')
    # from_official_pretrained('resnet101')
    test_from_official('resnet101')