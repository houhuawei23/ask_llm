# 批量处理命令使用指南

`batch` 命令用于批量处理文本内容并调用大语言模型进行问答。支持两种批量处理模式，具备多线程并发能力，并能够灵活处理不同服务商的模型调用。

## 基本用法

```bash
ask-llm batch <config_file> [OPTIONS]
```

### 必需参数

- `config_file`: 批量配置文件路径（YAML 格式）

### 可选参数

- `--output, -o`: 输出文件或目录路径（默认：自动生成）
- `--format, -f`: 输出格式（json, yaml, csv, markdown），默认：json
- `--threads, -t`: 并发线程数（默认：5，范围：1-50）
- `--retries, -r`: 失败任务的最大重试次数（默认：3，范围：0-10）
- `--config, -c`: 配置文件路径
- `--separate-files`: 按模型分组保存结果到不同文件

## 配置文件格式

### 模式 A：相同提示词 + 不同内容（prompt-contents.yml）

适用于使用相同提示词处理多个不同内容的场景。

```yaml
provider-models:  # 可选，如果未指定则交互式选择
  - provider: deepseek
    models:
      - model: deepseek-chat
      - model: deepseek-reasoner
        temperature: 1.0
        top_p: 0.9
  - provider: kimi
    models:
      - model: kimi-k2.5
prompt: 你是一个高中数学老师，请帮我解决以下问题：
contents:
  - 题目：已知函数f(x) = x^2 + 2x + 1，求f(x)的导数。
  - 题目：已知函数f(x) = x^2 + 3x + 1，求f(x)的导数。
  - 题目：已知函数f(x) = x^3 + 2x，求f(x)的导数。
```

**字段说明：**

- `provider-models`（可选）：指定要使用的模型列表
  - `provider`: 服务商名称
  - `models`: 模型列表
    - `model`: 模型名称（必需）
    - `temperature`: 温度参数（可选，0.0-2.0）
    - `top_p`: Top-p 采样参数（可选，0.0-1.0）
    - `max_tokens`: 最大生成 token 数（可选）
- `prompt`: 统一的提示词模板（必需）
- `contents`: 要处理的内容列表（必需，至少一项）

### 模式 B：提示词-内容对（prompt-content-pairs.yml）

适用于每个任务使用不同提示词的场景。

```yaml
---
prompt: 你是一个高中数学老师，请帮我解决以下问题：
content: 题目：已知函数f(x) = x^2 + 2x + 1，求f(x)的导数。
---
prompt: 你是一个英语老师，请帮我翻译以下内容：
content: Hello, how are you?
---
prompt: 你是一个代码审查专家，请审查以下代码：
content: def hello(): print("world")
```

**格式说明：**

- 使用 YAML 多文档格式（`---` 分隔符）
- 每个文档包含一对 `prompt` 和 `content`
- 第一个文档可以包含 `provider-models` 配置（可选）

## 使用示例

### 示例 1：基本批量处理

```bash
ask-llm batch batch-examples/prompt-contents.yml
```

### 示例 2：指定输出格式和文件

```bash
ask-llm batch batch-examples/prompt-contents.yml -o results.json -f json
```

### 示例 3：使用更多线程和重试

```bash
ask-llm batch batch-examples/prompt-contents.yml --threads 10 --retries 5
```

### 示例 4：按模型分组保存结果

```bash
ask-llm batch batch-examples/prompt-contents.yml --separate-files -o results/
```

这将为每个模型创建单独的结果文件：
- `results/batch_results_deepseek_deepseek-chat.json`
- `results/batch_results_deepseek_deepseek-reasoner.json`

### 示例 5：交互式选择模型

如果配置文件中未指定 `provider-models`，命令会提示您选择：

```bash
ask-llm batch my-config.yml
```

程序会：
1. 列出所有可用的服务商
2. 让您选择服务商
3. 列出该服务商的可用模型
4. 让您选择模型（可多选）
5. 检查并配置 API Key（如需要）

## 输出格式

### JSON 格式

结构化的 JSON 数据，包含完整的元数据：

```json
{
  "statistics": {
    "total_tasks": 3,
    "successful_tasks": 2,
    "failed_tasks": 1,
    "average_latency": 1.5,
    "total_input_tokens": 100,
    "total_output_tokens": 200
  },
  "results": [
    {
      "task_id": 1,
      "prompt": "...",
      "content": "...",
      "model_config": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "temperature": null,
        "top_p": null
      },
      "response": "...",
      "status": "success",
      "error": null,
      "metadata": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "input_tokens": 50,
        "output_tokens": 100,
        "latency": 1.5,
        "timestamp": "2024-01-15T10:30:00"
      },
      "timestamp": "2024-01-15T10:30:00",
      "retry_count": 0
    }
  ]
}
```

### YAML 格式

人类可读的结构化数据，格式与 JSON 相同但使用 YAML 语法。

### CSV 格式

表格格式，便于数据分析：

```csv
Task ID,Status,Provider,Model,Prompt,Content,Response,Error,Latency (s),Input Tokens,Output Tokens,Timestamp
1,success,deepseek,deepseek-chat,"...","...","...","",1.50,50,100,2024-01-15T10:30:00
```

### Markdown 格式

便于阅读的报告格式，包含：
- 统计信息摘要
- 按模型分组的结果
- 成功和失败任务的详细信息

## 错误处理和重试

### 自动重试

对于以下类型的错误，系统会自动重试：
- 网络超时
- 连接错误
- 速率限制（429）
- 服务器错误（500, 502, 503）

### 重试策略

- 使用指数退避策略
- 最大延迟限制为 10 秒
- 可配置最大重试次数（默认：3）

### 错误报告

失败的任务会在结果中记录详细的错误信息，包括：
- 错误消息
- 重试次数
- 任务状态

## API Key 配置

### 方式 1：环境变量

```bash
export DEEPSEEK_API_KEY="your-api-key"
export KIMI_API_KEY="your-api-key"
```

### 方式 2：配置文件

编辑 `providers.yml` 文件：

```yaml
providers:
  deepseek:
    api_key: your-api-key
    base_url: https://api.deepseek.com/v1
    default_model: deepseek-chat
```

### 方式 3：交互式输入

如果 API Key 未配置或无效，程序会提示您输入。

## 性能优化

### 并发线程数

- 默认：5 个线程
- 建议：根据 API 限流和网络条件调整
- 范围：1-50

### 批量处理建议

1. **小批量（< 10 个任务）**：使用默认线程数（5）
2. **中批量（10-100 个任务）**：使用 5-10 个线程
3. **大批量（> 100 个任务）**：使用 10-20 个线程，注意 API 限流

## 常见问题

### Q: 如何处理大量任务？

A: 使用 `--threads` 参数增加并发数，但要注意 API 限流。如果遇到限流错误，减少线程数或增加重试次数。

### Q: 如何只处理部分任务？

A: 编辑配置文件，只保留需要处理的内容。

### Q: 如何查看详细的处理进度？

A: 程序会显示进度条和实时统计信息。使用 `--debug` 标志可以查看更详细的日志。

### Q: 结果文件太大怎么办？

A: 使用 `--separate-files` 选项按模型分组保存，或使用 CSV 格式减少文件大小。

### Q: 如何处理 API Key 安全问题？

A: 推荐使用环境变量而不是配置文件存储 API Key。如果必须使用配置文件，确保文件权限设置为仅所有者可读（`chmod 600 providers.yml`）。

## 最佳实践

1. **测试配置**：先用少量任务测试配置是否正确
2. **保存结果**：始终指定输出文件，避免结果丢失
3. **监控进度**：关注成功率和平均延迟
4. **错误处理**：检查失败任务并适当调整重试策略
5. **API Key 安全**：使用环境变量或安全的密钥管理工具
