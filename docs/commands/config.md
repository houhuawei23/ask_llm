# config

Manage configuration, display settings, and test API connections.

## Usage

```bash
ask-llm config [ACTION] [OPTIONS]
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `ACTION` | `show` | Action to perform: `show`, `test`, or `init` |

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Configuration file path |
| `--provider` | `-p` | Provider to test (only with `test` action) |
| `--output` | `-o` | Output path for `init` (default: `~/.config/ask_llm/default_config.yml`) |

## Actions

### show

Display the loaded configuration including default provider, available providers, models, and API key status.

```bash
ask-llm config show
```

### test

Test API connectivity for configured providers.

```bash
# Test all providers
ask-llm config test

# Test a specific provider
ask-llm config test -p deepseek
```

### init

Create an example `default_config.yml`.

```bash
# Default location
ask-llm config init

# Custom location
ask-llm config init -o ./my_config.yml
```
