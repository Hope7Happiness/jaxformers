# 📦 安装与发布指南

## 本地开发安装

### 方法 1: 开发模式安装（推荐）

这种方式安装后，代码修改会立即生效，无需重新安装。

```bash
# 进入项目目录
cd jaxformers

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装包
pip install -e .

# 或者安装所有可选依赖（包括 PyTorch）
pip install -e ".[all]"

# 或者只安装开发工具
pip install -e ".[dev]"
```

### 方法 2: 普通安装

```bash
cd jaxformers
pip install .
```

### 方法 3: 从 requirements.txt 安装依赖

```bash
pip install -r requirements.txt
```

## 验证安装

安装完成后，可以通过以下方式验证：

```bash
# 方式 1: 运行测试脚本
python test_api.py

# 方式 2: 运行 demo
python demo.py

# 方式 3: Python 交互式测试
python -c "import __init__ as jaxformers; print(jaxformers.list_models())"
```

或者在 Python 中：

```python
# 如果是开发模式安装，可以直接 import
import jaxformers

# 列出所有模型
print(jaxformers.list_models())

# 创建一个模型
model = jaxformers.create_model('resnet50')
print("Success!")
```

## 构建分发包

### 准备工作

安装构建工具：

```bash
pip install build twine
```

### 构建源码包和 wheel 包

```bash
# 清理之前的构建
rm -rf build/ dist/ *.egg-info/

# 构建包
python -m build

# 或者使用传统方式
python setup.py sdist bdist_wheel
```

这将在 `dist/` 目录下生成：
- `jaxformers-0.1.0.tar.gz` (源码包)
- `jaxformers-0.1.0-py3-none-any.whl` (wheel 包)

### 检查构建的包

```bash
# 检查包的内容
twine check dist/*

# 查看包里包含的文件
tar -tzf dist/jaxformers-0.1.0.tar.gz
```

## 发布到 PyPI

### 测试发布（推荐先测试）

发布到 TestPyPI：

```bash
# 上传到 TestPyPI
twine upload --repository testpypi dist/*

# 从 TestPyPI 安装测试
pip install --index-url https://test.pypi.org/simple/ jaxformers
```

### 正式发布

```bash
# 上传到 PyPI
twine upload dist/*
```

发布后，用户就可以通过以下命令安装：

```bash
pip install jaxformers
```

## 版本管理

更新版本号时，需要修改以下文件：

1. `setup.py` 中的 `version="0.1.0"`
2. `__init__.py` 中的 `__version__ = "0.1.0"`

推荐使用语义化版本号：
- `0.1.0` - 初始版本
- `0.1.1` - Bug 修复
- `0.2.0` - 新功能
- `1.0.0` - 稳定版本

## 从 GitHub 安装

用户也可以直接从 GitHub 仓库安装：

```bash
# 安装最新版本
pip install git+https://github.com/yourusername/jaxformers.git

# 安装特定分支
pip install git+https://github.com/yourusername/jaxformers.git@main

# 安装特定标签
pip install git+https://github.com/yourusername/jaxformers.git@v0.1.0
```

## 卸载

```bash
pip uninstall jaxformers
```

## 常见问题

### Q1: 安装后无法 import？

**A:** 确保安装了所有依赖：
```bash
pip install -r requirements.txt
```

### Q2: 开发模式安装失败？

**A:** 尝试先升级 pip 和 setuptools：
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

### Q3: 如何在虚拟环境中安装？

**A:** 
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装
pip install -e .
```

### Q4: JAX 安装失败？

**A:** JAX 需要根据你的系统选择合适的版本：

```bash
# CPU 版本
pip install jax jaxlib

# GPU 版本（CUDA 11）
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# GPU 版本（CUDA 12）
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 依赖项说明

### 核心依赖
- `jax>=0.4.0` - JAX 框架
- `jaxlib>=0.4.0` - JAX 库
- `flax>=0.7.0` - Flax 神经网络库
- `numpy>=1.20.0` - 数值计算
- `ml_collections>=0.1.0` - 配置管理

### 可选依赖
- `torch>=1.10.0` - 用于 PyTorch 权重转换
- 开发工具：pytest, black, isort, flake8

## 文件清单

确保你的包包含以下文件：

```
jaxformers/
├── __init__.py          # 主要 API
├── setup.py             # 安装配置
├── requirements.txt     # 依赖列表
├── MANIFEST.in          # 包含的非 Python 文件
├── README.md            # 文档
├── LICENSE              # 许可证
├── QUICKSTART.md        # 快速开始
├── .gitignore          # Git 忽略文件
├── convnext.py         # 模型文件
├── deit.py
├── dino.py
├── mae.py
├── resnet.py
├── demo.py             # 演示脚本
├── examples.py         # 示例代码
└── test_api.py         # 测试脚本
```

## 发布检查清单

发布前检查：

- [ ] 更新版本号
- [ ] 更新 CHANGELOG (如果有)
- [ ] 运行所有测试: `python test_api.py`
- [ ] 检查代码格式: `black .` (如果安装了)
- [ ] 更新文档
- [ ] 清理构建目录: `rm -rf build/ dist/ *.egg-info/`
- [ ] 构建包: `python -m build`
- [ ] 检查包: `twine check dist/*`
- [ ] 在 TestPyPI 测试
- [ ] 创建 Git 标签: `git tag v0.1.0`
- [ ] 推送到 GitHub: `git push && git push --tags`
- [ ] 发布到 PyPI: `twine upload dist/*`

---

**祝你发布顺利！🚀**
