# Using Ollama with ask-llm

ask-llm supports [Ollama](https://ollama.com/) — a local LLM server — as a provider.
This allows you to run models locally without any API key or network access.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running
- At least one model pulled (e.g., `ollama pull qwen3.6`)

## Configuration

Ollama is pre-configured in the default `providers.yml` and `default_config.yml`.
The default model is `qwen3.6` and the server URL is `http://localhost:11434/v1`.

No API key is required — Ollama uses `api_key: null`.

### Verify connectivity

```bash
ask-llm config test --provider ollama
```

## Usage

### Basic chat

```bash
ask-llm ask "你是谁？" --provider ollama --model qwen3.6
```

### Translation

```bash
ask-llm trans document.md --provider ollama --model qwen3.6
```

### Formatting

```bash
ask-llm format ./file.md --type body \
  -p @prompts/md-body-format.md \
  --provider ollama --model qwen3.5:9b
```

## Available Models

| Model | Size | Notes |
|-------|------|-------|
| `qwen3.6` | 23 GB (Q4_K_M) | Default; MoE 36B parameters |
| `qwen3.5:9b` | 6.6 GB | Good balance of speed/quality for format/trans |
| `qwen3.5:0.8b` | 1.0 GB | Fastest; suitable for simple formatting |

## Performance Tips

### 1. Pre-load the model

Ollama unloads models after idle timeout. Pre-load to avoid cold-start latency:

```bash
# Set keep_alive to 24 hours
export OLLAMA_KEEP_ALIVE=24h

# Or warm up with a ping request
ask-llm ask "ping" --provider ollama --model qwen3.6 --no-stream
```

### 2. Reduce concurrency

Local models don't have independent instances like cloud APIs.
Lower concurrency avoids GPU scheduling contention:

```bash
# For format body: reduce from default 4 to 1
ask-llm format ./file.md --type body --body-concurrency 1 --provider ollama

# For translation: reduce from default 20 to 2-4
ask-llm trans ./doc.md --threads 4 --provider ollama
```

### 3. Use smaller models for simple tasks

Formatting and simple translations don't need 36B parameters:

```bash
# Use 0.8B model for markdown formatting
ask-llm format ./file.md --type body \
  -p @prompts/md-body-format.md \
  --provider ollama --model qwen3.5:0.8b
```

### 4. Cold start numbers (reference)

On 4× H100 GPUs, loading qwen3.6 (23 GB) from disk takes ~20-30 seconds.
Once loaded, inference runs at 50-100 tokens/second for MoE models.

## Troubleshooting

**404 page not found**: Ensure Ollama is running and the `base_url` ends with `/v1`
(the OpenAI SDK constructs URLs as `{base_url}/chat/completions`).

**Slow first request**: Model is loading from disk. Pre-load or use a smaller model.

**Connection refused**: Ensure Ollama server is running on the expected port (default 11434).
