#!/bin/bash
# JAXFormers PyPI 上传脚本

echo "========================================"
echo "  JAXFormers PyPI 上传助手"
echo "========================================"
echo ""

# 检查必要的工具
echo "🔍 检查必要工具..."
if ! command -v twine &> /dev/null; then
    echo "❌ twine 未安装"
    echo "   运行: pip install twine"
    exit 1
fi
echo "✅ twine 已安装"

# 检查 dist 目录
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "⚠️  dist 目录不存在或为空"
    echo "   运行: python -m build"
    exit 1
fi
echo "✅ dist 目录存在"

# 显示可上传的文件
echo ""
echo "📦 待上传的文件:"
ls -lh dist/

echo ""
echo "========================================"
echo "  上传选项"
echo "========================================"
echo ""
echo "请选择上传目标:"
echo "  1) TestPyPI (测试环境 - 推荐首次使用)"
echo "  2) PyPI (正式环境)"
echo "  3) 取消"
echo ""
read -p "请输入选项 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "🧪 准备上传到 TestPyPI..."
        echo ""
        echo "⚠️  上传前请确保:"
        echo "   1. 已注册 TestPyPI 账号: https://test.pypi.org/account/register/"
        echo "   2. 已创建 API token: https://test.pypi.org/manage/account/token/"
        echo "   3. 已配置 ~/.pypirc 文件"
        echo ""
        read -p "确认继续？ (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            echo ""
            echo "🚀 上传中..."
            twine upload --repository testpypi dist/*
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ 上传成功！"
                echo ""
                echo "📥 安装测试:"
                echo "   pip install --index-url https://test.pypi.org/simple/ jaxformers"
                echo ""
                echo "🌐 查看包:"
                echo "   https://test.pypi.org/project/jaxformers/"
            else
                echo ""
                echo "❌ 上传失败"
                echo "   请查看上面的错误信息"
                echo "   参考: PYPI_UPLOAD_GUIDE.md"
            fi
        fi
        ;;
    2)
        echo ""
        echo "🚀 准备上传到 PyPI..."
        echo ""
        echo "⚠️  上传前请确保:"
        echo "   1. 已注册 PyPI 账号: https://pypi.org/account/register/"
        echo "   2. 已创建 API token: https://pypi.org/manage/account/token/"
        echo "   3. 已配置 ~/.pypirc 文件"
        echo "   4. 已在 TestPyPI 测试成功"
        echo ""
        read -p "确认继续？ (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            echo ""
            echo "🚀 上传中..."
            twine upload dist/*
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ 上传成功！"
                echo ""
                echo "📥 用户可以安装:"
                echo "   pip install jaxformers"
                echo ""
                echo "🌐 查看包:"
                echo "   https://pypi.org/project/jaxformers/"
            else
                echo ""
                echo "❌ 上传失败"
                echo "   请查看上面的错误信息"
                echo "   参考: PYPI_UPLOAD_GUIDE.md"
            fi
        fi
        ;;
    3)
        echo "已取消"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
