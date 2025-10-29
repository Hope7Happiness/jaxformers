# 📦 PyPI 上传完整指南

## ⚠️ 你遇到的错误

```
TrustedPublishingFailure: The token request failed
invalid-publisher: valid token, but no corresponding publisher
```

这是因为你还没有在 PyPI 上设置账号和权限。

---

## 🔐 解决方案：配置 PyPI 认证

### 步骤 1️⃣: 注册 PyPI 账号

1. **访问 PyPI**: https://pypi.org/account/register/
2. **填写信息**:
   - Username（用户名）
   - Email（邮箱）
   - Password（密码）
3. **验证邮箱**: 检查邮件并点击验证链接

### 步骤 2️⃣: 创建 API Token

1. **登录 PyPI**: https://pypi.org/account/login/
2. **进入账号设置**: https://pypi.org/manage/account/
3. **滚动到 "API tokens" 部分**
4. **点击 "Add API token"**
5. **填写信息**:
   - Token name: 比如 "jaxformers-upload"
   - Scope: 选择 "Entire account (all projects)" 或创建项目后选择特定项目
6. **点击 "Add token"**
7. **复制 token**: 形如 `pypi-AgEIcHlwaS5vcmc...` （只会显示一次！）

### 步骤 3️⃣: 配置本地认证

#### 选项 A: 使用 .pypirc 文件（推荐）

创建或编辑 `~/.pypirc` 文件：

```bash
nano ~/.pypirc
```

添加以下内容：

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...你的实际token...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...你的testpypi token...
```

**重要**: 设置文件权限
```bash
chmod 600 ~/.pypirc
```

#### 选项 B: 在上传时输入（不推荐）

```bash
twine upload dist/*
# 会提示输入:
# Username: __token__
# Password: pypi-AgEIcHlwaS5vcmc...（你的token）
```

---

## 🧪 推荐：先测试上传到 TestPyPI

在正式上传到 PyPI 之前，建议先在 TestPyPI 测试：

### 1. 注册 TestPyPI 账号

- 访问: https://test.pypi.org/account/register/
- 注意: TestPyPI 和 PyPI 是**独立的账号系统**

### 2. 获取 TestPyPI API Token

- 登录: https://test.pypi.org/
- 创建 token: https://test.pypi.org/manage/account/token/

### 3. 上传到 TestPyPI

```bash
# 构建包
python -m build

# 上传到 TestPyPI
twine upload --repository testpypi dist/*
```

### 4. 从 TestPyPI 安装测试

```bash
pip install --index-url https://test.pypi.org/simple/ jaxformers
```

---

## 📦 完整的上传流程

### 准备工作

```bash
# 1. 安装工具（如果还没有）
pip install build twine

# 2. 确保在项目目录
cd /kmh-nfs-ssd-us-mount/code/siri/jaxformers

# 3. 清理之前的构建
rm -rf build/ dist/ *.egg-info/
```

### 构建包

```bash
# 方式 1: 使用 build 模块（推荐）
python -m build

# 方式 2: 使用 setup.py（传统方式）
python setup.py sdist bdist_wheel
```

构建成功后，`dist/` 目录会包含：
- `jaxformers-0.1.0.tar.gz` (源码包)
- `jaxformers-0.1.0-py3-none-any.whl` (wheel包)

### 检查包

```bash
# 检查包是否符合 PyPI 规范
twine check dist/*
```

应该看到：
```
Checking dist/jaxformers-0.1.0-py3-none-any.whl: PASSED
Checking dist/jaxformers-0.1.0.tar.gz: PASSED
```

### 上传

#### 上传到 TestPyPI（测试）

```bash
twine upload --repository testpypi dist/*
```

#### 上传到 PyPI（正式）

```bash
twine upload dist/*
```

---

## 🔧 常见问题解决

### Q1: 包名已被占用

**错误**: `HTTPError: 403 Forbidden from https://upload.pypi.org/legacy/`

**解决**: 
1. 在 `setup.py` 中更改包名
2. 使用更独特的名字，如 `jaxformers-yourname`

### Q2: 版本已存在

**错误**: `File already exists`

**解决**: 
1. 在 `setup.py` 中增加版本号 (如 `0.1.0` -> `0.1.1`)
2. 同时更新 `__init__.py` 中的 `__version__`
3. 清理并重新构建

### Q3: 认证失败

**错误**: `403 Forbidden` 或 `Invalid or non-existent authentication information`

**解决**:
1. 确认 token 正确（包括 `pypi-` 前缀）
2. 检查 `~/.pypirc` 文件权限: `chmod 600 ~/.pypirc`
3. 确认 token 没有过期

### Q4: README 格式问题

**警告**: `The description has invalid markup`

**解决**:
```bash
# 检查 README
python -m readme_renderer README.md
```

---

## 📋 发布前检查清单

在上传到 PyPI 之前，确保：

- [ ] **版本号正确**: 在 `setup.py` 和 `__init__.py` 中更新
- [ ] **README 完整**: 包含安装和使用说明
- [ ] **许可证**: LICENSE 文件存在
- [ ] **依赖正确**: requirements.txt 和 setup.py 一致
- [ ] **测试通过**: 运行 `python test_api.py`
- [ ] **包检查通过**: `twine check dist/*`
- [ ] **在 TestPyPI 测试**: 先测试上传
- [ ] **Git 提交**: `git commit -am "Release v0.1.0"`
- [ ] **创建标签**: `git tag v0.1.0`

---

## 🚀 快速命令参考

### 首次上传流程

```bash
# 1. 清理
rm -rf build/ dist/ *.egg-info/

# 2. 构建
python -m build

# 3. 检查
twine check dist/*

# 4. 测试上传到 TestPyPI
twine upload --repository testpypi dist/*

# 5. 测试安装
pip install --index-url https://test.pypi.org/simple/ jaxformers

# 6. 正式上传到 PyPI
twine upload dist/*
```

### 更新版本流程

```bash
# 1. 更新版本号（在 setup.py 和 __init__.py）
# version="0.1.1"

# 2. 清理并重新构建
rm -rf build/ dist/ *.egg-info/
python -m build

# 3. 检查并上传
twine check dist/*
twine upload dist/*
```

---

## 💡 当前你需要做的

### 立即操作步骤：

1. **注册 TestPyPI 账号**（测试用）
   - 访问: https://test.pypi.org/account/register/

2. **获取 API Token**
   - 登录后访问: https://test.pypi.org/manage/account/token/
   - 创建 token 并复制

3. **配置认证**
   ```bash
   nano ~/.pypirc
   ```
   
   添加：
   ```ini
   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-你的token
   ```

4. **测试上传**
   ```bash
   cd /kmh-nfs-ssd-us-mount/code/siri/jaxformers
   rm -rf build/ dist/ *.egg-info/
   python -m build
   twine upload --repository testpypi dist/*
   ```

---

## 🎯 或者：本地使用就够了

如果你**不需要公开发布**到 PyPI，可以：

### 选项 1: Git 安装
```bash
pip install git+https://your-git-url/jaxformers.git
```

### 选项 2: 本地安装
```bash
cd /kmh-nfs-ssd-us-mount/code/siri/jaxformers
pip install -e .
```

### 选项 3: 直接使用
```bash
cd /kmh-nfs-ssd-us-mount/code/siri/jaxformers
python demo.py
```

---

**需要帮助？** 告诉我：
1. 你是想公开发布到 PyPI？
2. 还是只在本地/团队内使用？

我可以根据你的需求提供具体的步骤！
