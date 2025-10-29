#!/bin/bash
# 修复 PyPI 配置文件

echo "🔧 PyPI 配置文件修复工具"
echo "================================"
echo ""

# 检查 .pypirc 是否存在
if [ ! -f ~/.pypirc ]; then
    echo "❌ ~/.pypirc 文件不存在"
    echo "请先创建该文件"
    exit 1
fi

echo "⚠️  检测到的问题："
echo "   Token 可能被分成了多行，导致认证失败"
echo ""
echo "💡 解决方案："
echo "   1. Token 必须在同一行，不能有换行"
echo "   2. 格式: password = pypi-完整的token字符串"
echo ""
echo "📝 当前配置文件内容:"
echo "---"
cat ~/.pypirc | grep -A 2 "testpypi"
echo "---"
echo ""
echo "🔑 请按以下步骤操作："
echo ""
echo "1. 获取新的 TestPyPI Token:"
echo "   https://test.pypi.org/manage/account/token/"
echo ""
echo "2. 编辑配置文件:"
echo "   nano ~/.pypirc"
echo ""
echo "3. 确保 token 在一行内，格式如下:"
echo "   [testpypi]"
echo "   repository = https://test.pypi.org/legacy/"
echo "   username = __token__"
echo "   password = pypi-完整token不换行"
echo ""
echo "4. 保存并退出 (Ctrl+X, Y, Enter)"
echo ""
echo "5. 重新上传"
echo ""

read -p "是否现在打开编辑器? (y/n): " choice
if [ "$choice" = "y" ]; then
    nano ~/.pypirc
    echo ""
    echo "✅ 配置文件已编辑"
    echo "   现在可以运行: ./upload_to_pypi.sh"
fi
