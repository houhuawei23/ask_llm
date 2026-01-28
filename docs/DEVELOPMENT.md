# 开发指南

本文档介绍如何使用项目中的代码质量工具和安全扫描工具。

## 目录

- [安装开发依赖](#安装开发依赖)
- [代码质量工具](#代码质量工具)
  - [Ruff - 代码检查和格式化](#ruff---代码检查和格式化)
  - [MyPy - 类型检查](#mypy---类型检查)
  - [Pydocstyle - 文档字符串检查](#pydocstyle---文档字符串检查)
- [安全扫描工具](#安全扫描工具)
  - [Bandit - 安全漏洞扫描](#bandit---安全漏洞扫描)
  - [Safety - 依赖漏洞扫描](#safety---依赖漏洞扫描)
- [Pre-commit Hooks](#pre-commit-hooks)
- [测试工具](#测试工具)
- [一键运行脚本](#一键运行脚本)

## 安装开发依赖

### 方式一：使用 pip 安装（推荐）

```bash
# 安装所有开发依赖
pip install -e ".[dev]"

# 或分别安装
pip install -e ".[lint]"      # 代码质量工具
pip install -e ".[security]"  # 安全扫描工具
```

### 方式二：使用虚拟环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -e ".[dev]"
```

## 代码质量工具

### Ruff - 代码检查和格式化

Ruff 是一个快速的 Python linter 和代码格式化工具，替代了 flake8、isort 和 black。

#### 检查代码问题

```bash
# 检查所有 Python 文件
ruff check src/ask_llm/

# 检查特定文件
ruff check src/ask_llm/cli.py

# 自动修复可修复的问题
ruff check --fix src/ask_llm/

# 检查并显示所有规则
ruff check --show-source src/ask_llm/
```

#### 格式化代码

```bash
# 格式化所有文件
ruff format src/ask_llm/

# 检查格式（不修改文件）
ruff format --check src/ask_llm/

# 格式化特定文件
ruff format src/ask_llm/cli.py
```

#### 常用命令组合

```bash
# 检查并修复 + 格式化（推荐在提交前运行）
ruff check --fix src/ask_llm/ && ruff format src/ask_llm/

# 只检查不修复
ruff check src/ask_llm/
```

#### 配置说明

Ruff 的配置在 `pyproject.toml` 的 `[tool.ruff]` 部分：

```toml
[tool.ruff]
line-length = 100
target-version = "py38"
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM", "ARG", "PIE", "T20", "PT", "Q", "RUF"]
ignore = ["E501", "B008", "ARG001", "ARG002"]
```

### MyPy - 类型检查

MyPy 用于静态类型检查，确保类型注解的正确性。

#### 基本使用

```bash
# 检查所有文件
mypy src/ask_llm/

# 检查特定文件
mypy src/ask_llm/cli.py

# 显示详细的错误信息
mypy --show-error-codes src/ask_llm/

# 忽略缺失导入（用于第三方库）
mypy --ignore-missing-imports src/ask_llm/
```

#### 常用选项

```bash
# 严格模式（更严格的类型检查）
mypy --strict src/ask_llm/

# 显示未使用的忽略
mypy --warn-unused-ignores src/ask_llm/

# 生成 HTML 报告
mypy --html-report html-report src/ask_llm/
```

#### 配置说明

MyPy 的配置在 `pyproject.toml` 的 `[tool.mypy]` 部分。当前配置允许渐进式类型检查，不会因为缺少类型注解而报错。

### Pydocstyle - 文档字符串检查

Pydocstyle 检查文档字符串是否符合规范（Google 风格）。

#### 基本使用

```bash
# 检查所有文件
pydocstyle src/ask_llm/

# 检查特定文件
pydocstyle src/ask_llm/cli.py

# 显示源代码
pydocstyle --source src/ask_llm/

# 只检查公共接口（推荐）
pydocstyle --convention=google src/ask_llm/
```

#### 常用选项

```bash
# 检查所有文件（包括私有方法）
pydocstyle --match='.*' src/ask_llm/

# 忽略特定错误代码
pydocstyle --ignore=D100,D104 src/ask_llm/

# 显示统计信息
pydocstyle --count src/ask_llm/
```

## 安全扫描工具

### Bandit - 安全漏洞扫描

Bandit 扫描 Python 代码中的安全漏洞。

#### 基本使用

```bash
# 扫描所有代码
bandit -r src/ask_llm/

# 扫描特定文件
bandit src/ask_llm/cli.py

# 生成 JSON 报告
bandit -r src/ask_llm/ -f json -o bandit-report.json

# 生成 HTML 报告
bandit -r src/ask_llm/ -f html -o bandit-report.html

# 只显示高/中危漏洞
bandit -r src/ask_llm/ -ll  # 低危及以上
bandit -r src/ask_llm/ -li  # 中危及以上
bandit -r src/ask_llm/ -lll # 高危及以上
```

#### 常用选项

```bash
# 排除测试目录
bandit -r src/ask_llm/ --exclude tests/

# 跳过特定测试
bandit -r src/ask_llm/ --skip B101,B601

# 显示详细信息
bandit -r src/ask_llm/ -v

# 配置文件扫描（使用 pyproject.toml 中的配置）
bandit -r src/ask_llm/ -c pyproject.toml
```

#### 配置说明

Bandit 的配置在 `pyproject.toml` 的 `[tool.bandit]` 部分：

```toml
[tool.bandit]
exclude_dirs = ["tests", "venv", ".venv", "build", "dist"]
skips = ["B101"]  # 允许在测试中使用 assert
```

### Safety - 依赖漏洞扫描

Safety 检查已安装的 Python 包是否存在已知的安全漏洞。

#### 基本使用

```bash
# 检查当前环境的依赖
safety check

# 检查 requirements 文件
safety check --file requirements.txt

# 检查并显示详细信息
safety check --full-report

# 只显示高危漏洞
safety check --json
```

#### 常用选项

```bash
# 检查并自动更新数据库
safety check --update

# 忽略特定漏洞（使用 CVE ID）
safety check --ignore 12345

# 生成报告文件
safety check --output safety-report.json
```

#### 注意事项

Safety 需要访问在线数据库，首次运行可能需要下载漏洞数据库。

## Pre-commit Hooks

Pre-commit hooks 在每次 git commit 前自动运行代码质量检查。

### 安装 Pre-commit

```bash
# 安装 pre-commit
pip install pre-commit

# 或使用项目依赖
pip install -e ".[dev]"
```

### 设置 Hooks

```bash
# 安装 git hooks
pre-commit install

# 安装 commit-msg hook（可选）
pre-commit install --hook-type commit-msg
```

### 手动运行

```bash
# 检查所有文件
pre-commit run --all-files

# 运行特定 hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files
pre-commit run bandit --all-files

# 跳过 hooks（不推荐）
git commit --no-verify
```

## 测试工具

### Pytest

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_cli.py

# 运行并显示覆盖率
pytest --cov=src/ask_llm --cov-report=term-missing

# 运行并生成 HTML 覆盖率报告
pytest --cov=src/ask_llm --cov-report=html

# 并行运行测试
pytest -n auto
```

## 一键运行脚本

项目提供了一个一键运行所有检查的脚本：

```bash
# 运行所有代码质量检查和安全扫描
./scripts/check_code_quality.sh
```

脚本会自动运行：
1. Ruff 代码检查
2. Ruff 代码格式化检查
3. MyPy 类型检查
4. Pydocstyle 文档字符串检查
5. Bandit 安全扫描
6. Pytest 测试

如果所有检查通过，脚本会返回退出码 0；否则返回非零退出码。

## CI/CD 集成

### GitHub Actions 示例

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: ./scripts/check_code_quality.sh
```

## 常见问题

### Q: 如何忽略特定的 ruff 规则？

A: 在 `pyproject.toml` 的 `[tool.ruff.lint]` 部分的 `ignore` 列表中添加规则代码。

### Q: MyPy 报告很多第三方库的类型错误怎么办？

A: 在 `pyproject.toml` 的 `[[tool.mypy.overrides]]` 部分添加模块覆盖配置。

### Q: 如何跳过某些文件的检查？

A: 使用 `per-file-ignores` 配置，例如在 `[tool.ruff.lint.per-file-ignores]` 中配置。

### Q: Bandit 误报怎么办？

A: 在代码中添加 `# nosec` 注释，或在 `pyproject.toml` 的 `[tool.bandit]` 的 `skips` 列表中添加测试 ID。

## 参考资源

- [Ruff 文档](https://docs.astral.sh/ruff/)
- [MyPy 文档](https://mypy.readthedocs.io/)
- [Pydocstyle 文档](https://www.pydocstyle.org/)
- [Bandit 文档](https://bandit.readthedocs.io/)
- [Safety 文档](https://pyup.io/safety/)
- [Pytest 文档](https://docs.pytest.org/)
