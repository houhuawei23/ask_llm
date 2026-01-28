# Ask LLM 重构版本

现代化的命令行LLM API工具，使用Typer、Pydantic、Rich和Loguru构建。

## 目录

- [概述](#概述)
- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [配置](#配置)
- [使用方法](#使用方法)
- [主要接口](#主要接口)
- [测试](#测试)

## 概述

Ask LLM 是一个灵活的命令行工具，用于调用多个LLM API（DeepSeek、Qwen等）。重构版本采用了现代化的Python工具链，提供更好的用户体验和代码质量。

## 主要特性

### 专业性
- **Typer**: 现代化的CLI框架，支持类型提示和自动帮助生成
- **Pydantic**: 数据验证和序列化
- **Loguru**: 强大的日志记录
- **Rich**: 美观的控制台输出
- **tqdm**: 文件操作的进度条

### 功能特性
- 支持多个OpenAI兼容的API提供商
- 交互式聊天模式
- 灵活的提示模板
- 文件输入/输出支持
- 流式响应
- 详细的请求元数据

## 项目结构

```
ask_llm/
├── src/ask_llm/              # 主包
│   ├── __init__.py           # 版本信息
│   ├── __main__.py           # python -m 入口
│   ├── cli.py                # Typer CLI实现
│   ├── core/                 # 核心逻辑
│   │   ├── __init__.py
│   │   ├── models.py         # Pydantic数据模型
│   │   ├── processor.py      # 请求处理
│   │   └── chat.py           # 聊天会话
│   ├── providers/            # API提供商
│   │   ├── __init__.py
│   │   ├── base.py           # 抽象基类
│   │   └── openai_compatible.py
│   ├── config/               # 配置管理
│   │   ├── __init__.py
│   │   ├── loader.py         # 配置加载
│   │   └── manager.py        # 配置管理
│   └── utils/                # 工具模块
│       ├── __init__.py
│       ├── console.py        # 控制台工具(Rich)
│       ├── file_handler.py   # 文件I/O(tqdm)
│       └── token_counter.py  # Token计数
├── tests/                    # 测试
│   ├── unit/                 # 单元测试
│   └── integration/          # 集成测试
├── docs/
│   └── README_ask_llm.md     # 本文档
├── pyproject.toml            # 现代Python项目配置
├── requirements.txt          # 依赖
└── README.md                 # 主README
```

## 安装

### 开发安装

```bash
# 克隆仓库
git clone <repository-url>
cd ask_llm

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 生产安装

```bash
pip install -e .
```

## 配置

### 快速开始

1. 创建示例配置:
```bash
ask-llm config init
```

2. 编辑 `config.json`，添加你的API密钥:
```json
{
  "default_provider": "deepseek",
  "providers": {
    "deepseek": {
      "api_provider": "deepseek",
      "api_key": "sk-your-actual-api-key",
      "api_base": "https://api.deepseek.com/v1",
      "api_model": "deepseek-chat",
      "models": ["deepseek-chat", "deepseek-reasoner"],
      "api_temperature": 0.7
    }
  }
}
```

3. 验证配置:
```bash
ask-llm config show
ask-llm config test
```

## 使用方法

### 基本用法

```bash
# 处理文件
ask-llm input.txt

# 直接输入文本
ask-llm "Translate to Chinese: Hello world"

# 指定输出文件
ask-llm input.txt -o output.txt

# 使用自定义提示
ask-llm input.txt -p "Summarize this: {content}"

# 切换模型
ask-llm input.txt -m gpt-4 -t 0.5
```

### 聊天模式

```bash
# 启动交互式聊天
ask-llm chat

# 带初始上下文的聊天
ask-llm chat -i context.txt

# 设置系统提示
ask-llm chat -s "You are a helpful coding assistant"

# 指定模型
ask-llm chat -m deepseek-reasoner
```

聊天模式支持以下命令:
- `/help` - 显示帮助
- `/info` - 显示会话信息
- `/models` - 列出可用模型
- `/model <name>` - 切换模型
- `/save <file>` - 保存聊天记录
- `/clear` - 清除历史
- `/system <text>` - 设置系统提示
- `!command` - 执行shell命令

### 配置管理

```bash
# 显示配置
ask-llm config show

# 测试API连接
ask-llm config test

# 测试特定提供商
ask-llm config test -p deepseek

# 创建示例配置
ask-llm config init
```

## 主要接口

### ProviderConfig (Pydantic Model)

```python
from ask_llm.core.models import ProviderConfig

config = ProviderConfig(
    api_provider="deepseek",
    api_key="sk-...",
    api_base="https://api.deepseek.com/v1",
    api_model="deepseek-chat",
    api_temperature=0.7,
)
```

### OpenAICompatibleProvider

```python
from ask_llm.providers import OpenAICompatibleProvider
from ask_llm.core.models import ProviderConfig

config = ProviderConfig(...)
provider = OpenAICompatibleProvider(config)

# 简单调用
response = provider.call(prompt="Hello!")

# 流式响应
for chunk in provider.call(prompt="Hello!", stream=True):
    print(chunk, end="")

# 多轮对话
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"},
]
response = provider.call(messages=messages)
```

### RequestProcessor

```python
from ask_llm.core.processor import RequestProcessor

processor = RequestProcessor(provider)

# 处理内容
result = processor.process_with_metadata(
    content="Text to process",
    prompt_template="Process this: {content}",
    temperature=0.5,
)
print(result.content)
print(result.metadata.format())
```

### ChatSession

```python
from ask_llm.core.chat import ChatSession

session = ChatSession(
    provider=provider,
    temperature=0.7,
    model="deepseek-chat"
)
session.start()  # 启动交互式会话
```

### ConfigLoader & ConfigManager

```python
from ask_llm.config.loader import ConfigLoader
from ask_llm.config.manager import ConfigManager

# 加载配置
config = ConfigLoader.load("config.json")

# 管理配置
manager = ConfigManager(config)
manager.set_provider("qwen")
manager.apply_overrides(model="qwen-max", temperature=0.5)

provider_config = manager.get_provider_config()
```

### Console (Rich输出)

```python
from ask_llm.utils.console import console

# 设置
console.setup(quiet=False, debug=False)

# 各种输出
console.print("Regular message")
console.print_success("Success!")
console.print_error("Error!")
console.print_warning("Warning!")
console.print_info("Info")

# 表格
console.print_table(
    headers=["Name", "Value"],
    rows=[["key1", "value1"], ["key2", "value2"]]
)
```

### FileHandler (带进度条)

```python
from ask_llm.utils.file_handler import FileHandler

# 读取文件
content = FileHandler.read("large_file.txt", show_progress=True)

# 写入文件
FileHandler.write("output.txt", content, show_progress=True)

# 生成输出路径
output_path = FileHandler.generate_output_path("input.txt")
```

### TokenCounter

```python
from ask_llm.utils.token_counter import TokenCounter

# 计数
words = TokenCounter.count_words(text)
tokens = TokenCounter.count_tokens(text, model="gpt-4")

# 估算所有指标
stats = TokenCounter.estimate_tokens(text, model="gpt-4")
# stats = {"word_count": ..., "token_count": ..., "char_count": ...}
```

## 测试

### 运行测试

```bash
# 所有测试
pytest

# 仅单元测试
pytest tests/unit -v

# 仅集成测试
pytest tests/integration -v

# 带覆盖率
pytest --cov=src/ask_llm --cov-report=html
```

### 运行演示脚本

```bash
# 生成并运行演示
python -m pytest tests/integration/test_cli.py::TestDemoScript -v -s

# 或使用Typer CLI测试
python -m ask_llm --help
ask-llm --help
```

## 依赖说明

| 包 | 用途 | 版本 |
|---|---|---|
| typer | CLI框架 | >=0.9.0 |
| rich | 控制台美化 | >=13.0.0 |
| pydantic | 数据验证 | >=2.0.0 |
| loguru | 日志记录 | >=0.7.0 |
| tqdm | 进度条 | >=4.65.0 |
| openai | API客户端 | >=1.0.0 |
| tiktoken | Token计数 | >=0.5.0 |
| pytest | 测试框架 | >=7.0.0 |

## 迁移指南

从旧版本迁移:

1. 配置文件格式保持不变
2. 命令行参数基本兼容:
   - `python ask_llm.py input.txt` → `ask-llm input.txt`
   - `python ask_llm.py -i input.txt --chat` → `ask-llm chat -i input.txt`
3. 新增功能:
   - 使用 `ask-llm config init` 创建配置
   - 使用 `ask-llm config test` 测试API
   - 更美观的输出格式

## 许可证

MIT License
