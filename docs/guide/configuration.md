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
