#!/bin/bash
# 代码质量检查脚本
# 运行所有代码质量工具和安全扫描

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Ask LLM 代码质量检查"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在项目根目录
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi

# 检查工具是否安装
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}错误: $1 未安装。请运行: pip install -e \".[dev]\"${NC}"
        exit 1
    fi
}

echo "检查工具安装..."
check_tool ruff
check_tool mypy
check_tool pydocstyle
check_tool bandit
check_tool pytest
echo -e "${GREEN}✓ 所有工具已安装${NC}"
echo ""

# 运行检查
ERRORS=0

echo "=========================================="
echo "1. Ruff - 代码检查"
echo "=========================================="
if ruff check src/ask_llm/; then
    echo -e "${GREEN}✓ Ruff 检查通过${NC}"
else
    echo -e "${RED}✗ Ruff 检查失败${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "=========================================="
echo "2. Ruff - 代码格式化检查"
echo "=========================================="
if ruff format --check src/ask_llm/; then
    echo -e "${GREEN}✓ 代码格式正确${NC}"
else
    echo -e "${YELLOW}⚠ 代码格式需要调整，运行: ruff format src/ask_llm/${NC}"
fi
echo ""

echo "=========================================="
echo "3. MyPy - 类型检查"
echo "=========================================="
if mypy src/ask_llm/; then
    echo -e "${GREEN}✓ MyPy 检查通过${NC}"
else
    echo -e "${YELLOW}⚠ MyPy 发现类型问题（非致命）${NC}"
fi
echo ""

echo "=========================================="
echo "4. Pydocstyle - 文档字符串检查"
echo "=========================================="
if pydocstyle src/ask_llm/; then
    echo -e "${GREEN}✓ 文档字符串检查通过${NC}"
else
    echo -e "${YELLOW}⚠ 文档字符串需要改进${NC}"
fi
echo ""

echo "=========================================="
echo "5. Bandit - 安全扫描"
echo "=========================================="
if bandit -r src/ask_llm/ -ll; then
    echo -e "${GREEN}✓ Bandit 安全检查通过${NC}"
else
    echo -e "${YELLOW}⚠ Bandit 发现潜在安全问题${NC}"
fi
echo ""

echo "=========================================="
echo "6. Pytest - 运行测试"
echo "=========================================="
if pytest --cov=src/ask_llm --cov-report=term-missing; then
    echo -e "${GREEN}✓ 所有测试通过${NC}"
else
    echo -e "${RED}✗ 测试失败${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 总结
echo "=========================================="
echo "检查完成"
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有关键检查通过！${NC}"
    exit 0
else
    echo -e "${RED}✗ 发现 $ERRORS 个关键错误${NC}"
    exit 1
fi
