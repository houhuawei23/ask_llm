# Ask LLM 重构计划

## 1. 现有代码分析

### 1.1 项目现状
- **主文件**: `ask_llm.py` (524行) - 包含CLI参数解析、主流程控制
- **聊天模式**: `chat_mode.py` (501行) - 交互式聊天实现
- **提供者模块**: `providers/` - OpenAI兼容API封装
- **工具模块**: `utils/` - 配置加载、文件处理、日志等

### 1.2 存在问题
1. **CLI使用argparse**: 参数解析繁琐，帮助信息不够美观
2. **自定义日志系统**: 未使用成熟的loguru库
3. **缺少进度条**: 大文件处理无进度显示
4. **类型注解不完整**: 部分函数缺少返回类型
5. **模块职责不清**: ask_llm.py包含过多逻辑
6. **缺少测试**: 无单元测试和集成测试
7. **错误处理分散**: 未统一异常处理机制

## 2. 重构目标

### 2.1 专业性
- 使用Typer替代argparse实现现代化CLI
- 使用loguru替代自定义logger
- 使用tqdm添加进度条
- 完善类型注解

### 2.2 可维护性
- 模块化设计，分文件管理
- 统一异常处理
- 完善的日志记录

### 2.3 可读性
- 清晰的注释和文档字符串
- 完整的类型提示
- 遵循PEP8规范

### 2.4 交互性
- 美观的命令行界面 (rich + typer)
- 清晰的输出格式
- 进度条显示

### 2.5 测试
- 单元测试覆盖核心逻辑
- 集成测试验证整体功能

## 3. 新架构设计

```
ask_llm/
├── src/
│   └── ask_llm/
│       ├── __init__.py          # 版本信息
│       ├── __main__.py          # 入口点
│       ├── cli.py               # Typer CLI实现
│       ├── core/
│       │   ├── __init__.py
│       │   ├── processor.py     # 请求处理核心
│       │   ├── chat.py          # 聊天模式
│       │   └── models.py        # 数据模型 (Pydantic)
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py          # 抽象基类
│       │   └── openai_compatible.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py        # 配置加载
│       │   ├── validator.py     # 配置验证
│       │   └── manager.py       # 配置管理
│       └── utils/
│           ├── __init__.py
│           ├── file_handler.py  # 文件I/O
│           ├── token_counter.py # Token计数
│           ├── output.py        # 输出格式化
│           └── console.py       # 控制台工具 (rich)
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # pytest配置
│   ├── unit/
│   │   ├── test_providers.py
│   │   ├── test_config.py
│   │   └── test_utils.py
│   └── integration/
│       └── test_cli.py
├── docs/
│   └── README_ask_llm.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 4. 详细重构步骤

### 步骤1: 依赖配置更新
- 添加 typer, loguru, tqdm, rich, pydantic
- 创建 pyproject.toml

### 步骤2: 数据模型定义 (Pydantic)
```python
class ProviderConfig(BaseModel):
    api_provider: str
    api_key: str
    api_base: str
    api_model: str
    models: List[str] = []
    api_temperature: float = 0.7
    api_top_p: Optional[float] = None

class AppConfig(BaseModel):
    default_provider: str
    providers: Dict[str, ProviderConfig]
```

### 步骤3: 重构配置模块
- 使用Pydantic验证配置
- 统一配置加载和验证

### 步骤4: 重构提供者模块
- 完善类型注解
- 统一异常处理

### 步骤5: 实现Typer CLI
```python
import typer
from typing_extensions import Annotated

app = typer.Typer(help="Ask LLM - 灵活的LLM API命令行工具")

@app.command()
def ask(
    input_file: Annotated[Optional[str], typer.Argument(help="输入文件或文本")] = None,
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="输出文件")] = None,
    # ... 其他参数
):
    """发送请求到LLM API"""
    pass

@app.command()
def chat(
    # ... 参数
):
    """进入交互式聊天模式"""
    pass
```

### 步骤6: 重构聊天模式
- 提取命令处理逻辑
- 使用rich美化输出

### 步骤7: 添加进度条和日志
- 文件读写使用tqdm
- 使用loguru记录日志

### 步骤8: 编写测试
- 单元测试: 配置加载、提供者、工具函数
- 集成测试: CLI完整流程

### 步骤9: 编写文档
- README_ask_llm.md
- 代码文档字符串

## 5. 关键技术选型

| 功能 | 原方案 | 新方案 |
|------|--------|--------|
| CLI框架 | argparse | typer + rich |
| 日志 | 自定义Logger | loguru |
| 配置验证 | 手动验证 | Pydantic |
| 进度条 | 无 | tqdm |
| 输出美化 | print | rich.console |
| 类型注解 | 部分 | 完整 + mypy |

## 6. 向后兼容性

- 保留原有的config.json格式
- 支持原有的命令行参数
- 添加迁移指南

## 7. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 功能回归 | 高 | 完善的集成测试 |
| 配置不兼容 | 中 | 保留原有配置格式 |
| 性能下降 | 低 | 基准测试对比 |
