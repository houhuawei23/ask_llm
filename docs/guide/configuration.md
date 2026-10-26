# Configuration

Ask LLM uses a unified YAML configuration file (`default_config.yml`). The configuration loader merges settings in this priority order:

1. CLI arguments (`--provider`, `--model`, etc.)
2. Environment variables (`ASK_LLM_*`)
3. User configuration file (`default_config.yml`)
4. Package built-in defaults

## Configuration Search Paths

When no `--config` argument is provided, the loader searches in this order:

- `./default_config.yml`
- `~/.config/ask_llm/default_config.yml`
- `/etc/ask_llm/default_config.yml`
- Package built-in default

## Initialize a Configuration File

```bash
ask-llm config init
```

By default, this writes to `~/.config/ask_llm/default_config.yml`. You can specify a different path:

```bash
ask-llm config init -o ./my_config.yml
```

## Provider Configuration

A minimal provider block looks like this:

```yaml
default_provider: deepseek
default_model: deepseek-chat
providers:
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key: ${DEEPSEEK_API_KEY}
    default_model: deepseek-chat
    models:
      - name: deepseek-chat
      - name: deepseek-reasoner
    api_temperature: 0.7
```

Use `${VAR_NAME}` syntax to reference environment variables so API keys are not stored in plain text.

## Provider Runtime Catalog (`providers.yml`)

Provider runtime settings (base URL, API key, model list) and per-model metadata
(`reasoning`, `token_per_character`, `functions.json_output`) are read from
`providers.yml`, searched in this order:

- `$ASK_LLM_PROVIDERS_YML`
- `~/.config/ask_llm/providers.yml`
- packaged copy (`ask_llm/config/providers.yml`)

The working directory is deliberately **not** on this list, and the per-model
metadata handed to the LLM engine is resolved from the same file, so `ask-llm`
behaves identically no matter which directory it is launched from.

Mark thinking models with `reasoning: false` when the translation/format
pipelines use them. Without it, a thinking model can spend 30-170s streaming
`reasoning_content` before any translatable content appears, which looks like a
hang (and can exhaust `max_tokens` before the body is produced).

```yaml
providers:
  deepseek:
    models:
      - name: deepseek-v4-flash
        reasoning: false   # disable CoT for translation/format pipelines
        context_length: 1_000_000
```

## Environment Variables

Supported `ASK_LLM_*` variables include:

| Variable | Config Path |
|----------|-------------|
| `ASK_LLM_DEFAULT_PROVIDER` | `default_provider` |
| `ASK_LLM_DEFAULT_MODEL` | `default_model` |
| `ASK_LLM_TRANSLATION_TARGET_LANGUAGE` | `translation.target_language` |
| `ASK_LLM_TRANSLATION_THREADS` | `translation.threads` |
| `ASK_LLM_TRANSLATION_MAX_CHUNK_TOKENS` | `translation.max_chunk_tokens` |
| `ASK_LLM_BATCH_THREADS` | `batch.threads` |
| `ASK_LLM_BATCH_RETRIES` | `batch.retries` |

See `src/ask_llm/config/loader.py` for the complete mapping.

## Testing Configuration

Display the loaded configuration:

```bash
ask-llm config show
```

Test API connectivity for all configured providers:

```bash
ask-llm config test
```

Test a specific provider:

```bash
ask-llm config test -p deepseek
```
