# Format 命令文档

## 概述

`format` 命令用于对 Markdown 文件的标题层级进行格式化。通过提取文档中的所有标题，使用 LLM API 推断正确的标题层级关系，然后将格式化后的标题应用回原文档。

## 主要功能

- **标题提取**：自动识别 Markdown 文档中的所有 ATX 风格标题（`# Title`）
- **智能格式化**：使用 LLM 根据标题编号（1, 1.1, 1.1.1）或上下文推断正确的标题层级
- **批量处理**：支持 glob 模式批量处理多个文件
- **保持内容**：只调整标题级别，不改变标题文本内容

## 使用方法

### 基本用法

```bash
# 格式化单个文件
ask-llm format document.md

# 格式化多个文件
ask-llm format *.md

# 指定输出目录
ask-llm format *.md -o formatted/

# 指定输出文件
ask-llm format doc.md -o formatted_doc.md
```

### 高级选项

```bash
# 指定模型和提供商
ask-llm format doc.md -m gpt-4 -a openai

# 使用自定义提示词模板
ask-llm format doc.md -p @prompts/custom-format.md

# 覆盖已存在的输出文件
ask-llm format doc.md -f

# 设置温度参数
ask-llm format doc.md -t 0.3
```

## 命令参数

| 参数 | 简写 | 类型 | 说明 |
|------|------|------|------|
| `files` | - | 位置参数 | 输入文件路径或 glob 模式 |
| `--output` | `-o` | 可选 | 输出文件或目录路径 |
| `--config` | `-c` | 可选 | 配置文件路径 |
| `--provider` | `-a` | 可选 | API 提供商名称 |
| `--model` | `-m` | 可选 | 模型名称 |
| `--temperature` | `-t` | 可选 | 采样温度 (0.0-2.0) |
| `--force` | `-f` | flag | 覆盖已存在的输出文件 |
| `--prompt` | `-p` | 可选 | 提示词模板文件路径（支持 @ 前缀） |

## 使用示例

### 示例 1：格式化单个文件

**输入文件** (`example.md`):
```markdown
# Title
Some content here.

# 1 title1
Content for section 1.

# 1.1 title1.1
Subsection content.

# 1.1.1 title1.1.1
Sub-subsection content.
```

**命令**:
```bash
ask-llm format example.md
```

**输出文件** (`example_formatted.md`):
```markdown
# Title
Some content here.

## 1 title1
Content for section 1.

### 1.1 title1.1
Subsection content.

#### 1.1.1 title1.1.1
Sub-subsection content.
```

### 示例 2：批量格式化

```bash
# 格式化所有 .md 文件到 formatted/ 目录
ask-llm format *.md -o formatted/
```

### 示例 3：使用自定义提示词

```bash
# 使用项目根目录下的自定义提示词
ask-llm format doc.md -p @prompts/custom-heading-format.md
```

## 工作原理

### 1. 标题提取

使用正则表达式 `^(#{1,6})\s+(.+)$` 提取所有 ATX 风格的标题：
- 识别 1-6 个 `#` 符号
- 提取标题文本
- 记录每个标题在原文档中的位置

### 2. LLM 格式化

将提取的标题列表发送给 LLM，LLM 根据以下规则推断正确的层级：
- 第一个标题通常是 `#` (h1)
- 编号 `1` → `##` (h2)
- 编号 `1.1` → `###` (h3)
- 编号 `1.1.1` → `####` (h4)
- 以此类推

### 3. 应用格式化

将格式化后的标题按位置替换回原文档，保持其他内容不变。

## 提示词模板

默认使用 `prompts/md-heading-format.md` 作为提示词模板。模板包含：
- 格式化规则说明
- 示例输入/输出
- 输出格式要求

可以通过 `--prompt` 参数指定自定义模板。

## 错误处理

### 常见错误

1. **文件不存在**
   ```
   Error: Input file not found: doc.md
   ```

2. **无标题**
   ```
   Warning: No headings found in doc.md. Skipping.
   ```

3. **输出文件已存在**
   ```
   Error: Output file already exists: doc_formatted.md. Use --force to overwrite.
   ```

4. **LLM API 调用失败**
   ```
   Error: API error: Connection timeout
   ```

### 处理策略

- 文件不存在：跳过并继续处理其他文件
- 无标题：跳过该文件
- 输出文件已存在：提示用户使用 `--force` 覆盖
- LLM 调用失败：记录错误日志，跳过该文件

## 限制

- **仅支持 Markdown**：目前只支持 `.md` 和 `.markdown` 文件
- **ATX 风格标题**：只识别 `# Title` 格式的标题，不支持 Setext 风格（`===`）
- **标题数量限制**：建议单文件标题数量不超过 100 个（取决于 LLM 上下文窗口）

## 最佳实践

1. **备份原文件**：格式化前建议备份原文件
2. **检查结果**：格式化后检查输出文件，确保标题层级正确
3. **批量处理**：使用 `-o` 指定输出目录，避免覆盖原文件
4. **自定义提示词**：对于特殊格式的文档，可以创建自定义提示词模板

## 技术实现

### 核心模块

- `ask_llm.core.md_heading_formatter.HeadingExtractor`：标题提取
- `ask_llm.core.md_heading_formatter.HeadingFormatter`：LLM 格式化
- `ask_llm.core.md_heading_formatter.HeadingApplier`：应用格式化结果

### 依赖

- `loguru`：日志记录
- `llm_engine`：LLM API 调用
- `typer`：CLI 框架
- `rich`：控制台输出

## 相关文档

- [主文档](README_ask_llm.md)
- [翻译命令文档](../README.md#trans-命令)
- [配置文件说明](../README.md#配置)
