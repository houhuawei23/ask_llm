# Ask LLM

A flexible command-line tool for calling multiple LLM APIs (DeepSeek, Qwen, etc.) with customizable configuration.

## Features

- Support for multiple LLM API providers (OpenAI Compatible)
  - DeepSeek
  - Qwen
  - Easy to extend for more providers
- Flexible configuration via JSON config file
- Command-line argument overrides
- Custom prompt templates
- File input/output support (.txt and .md)
- Debug and quiet modes
- Comprehensive error handling

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API keys in `config.json`:
```json
{
  "default_provider": "deepseek",
  "providers": {
    "deepseek": {
      "api_provider": "deepseek",
      "api_key": "your-api-key-here",
      "api_base": "https://api.deepseek.com/v1",
      "api_model": "deepseek-chat",
      "api_temperature": 0.7,
      "api_top_p": 0.95
    },
    "qwen": {
      "api_provider": "qwen",
      "api_key": "your-api-key-here",
      "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_model": "qwen-turbo",
      "api_temperature": 0.7,
      "api_top_p": 0.8
    }
  }
}
```

## Usage

### Basic Usage

```bash
python ask_llm.py -i input.txt
```

This will:
- Read content from `input.txt`
- Use the default provider and model from config.json
- Save output to `input_output.txt`

### Advanced Usage

```bash
# Specify output file
python ask_llm.py -i input.txt -o output.txt

# Use custom prompt template
python ask_llm.py -i input.txt -p prompt.txt

# Override model and temperature
python ask_llm.py -i input.txt -m deepseek-chat -t 0.5

# Use different API provider
python ask_llm.py -i input.txt -a qwen

# Use custom config file
python ask_llm.py -i input.txt -c custom_config.json

# Force overwrite output file
python ask_llm.py -i input.txt -o output.txt -f

# Debug mode
python ask_llm.py -i input.txt -d

# Quiet mode (only errors)
python ask_llm.py -i input.txt -q
```

## Command Line Options

### Required Arguments
- `-i, --input`: Input file path (txt or md)

### Optional Arguments
- `-o, --output`: Output file path (default: `input_name_output.txt/md`)
- `-p, --prompt`: Prompt template file (default: built-in prompt)
- `-m, --model`: Model name (overrides config default)
- `-a, --api_provider`: API provider name (overrides config default)
- `-t, --temperature`: Temperature parameter (overrides config default)
- `-c, --config`: Configuration file path (default: `config.json`)
- `-f, --force`: Force overwrite output file if it exists
- `-d, --debug`: Enable debug logging
- `-q, --quiet`: Quiet mode (only show errors)
- `-h, --help`: Show help message
- `-v, --version`: Show version information

## Prompt Templates

You can create custom prompt templates. The template should contain a `{content}` placeholder where the input file content will be inserted.

Example prompt template (`prompt.txt`):
```
You are a helpful assistant. Please analyze the following text and provide a summary:

{content}
```

If the template doesn't contain `{content}`, it will be automatically appended to the end.

## Configuration

The configuration file (`config.json`) supports multiple API providers. Each provider should have:
- `api_provider`: Provider identifier
- `api_key`: Your API key
- `api_base`: API base URL
- `api_model`: Default model name
- `api_temperature`: Default temperature (0.0-2.0)
- `api_top_p`: Optional top_p parameter

## Adding New Providers

To add a new provider:

1. Create a new file in `providers/` directory (e.g., `providers/new_provider.py`)
2. Implement the `BaseProvider` class:

```python
from .base import BaseProvider
from openai import OpenAI

class NewProviderProvider(BaseProvider):
    def validate_config(self):
        # Validate required config keys
        required_keys = ['api_key', 'api_base', 'api_model']
        for key in required_keys:
            if key not in self.config or not self.config[key]:
                raise ValueError(f"Missing required config key: {key}")
    
    def call(self, prompt, temperature=None, model=None, **kwargs):
        # Implement API call logic
        client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['api_base']
        )
        # ... API call implementation
```

3. Register the provider in `providers/__init__.py`
4. Add it to the provider map in `ask_llm.py` (`get_provider_class` function)
5. Add configuration to `config.json`

## Project Structure

```
ask_llm/
├── ask_llm.py          # Main CLI script
├── config.json         # Configuration file
├── requirements.txt    # Python dependencies
├── providers/          # API provider modules
│   ├── __init__.py
│   ├── base.py         # Base provider class
│   ├── deepseek.py     # DeepSeek implementation
│   └── qwen.py         # Qwen implementation
├── utils/              # Utility modules
│   ├── __init__.py
│   ├── file_handler.py # File I/O utilities
│   ├── config_loader.py # Configuration loader
│   └── logger.py       # Logging utilities
└── README.md           # This file
```

## License

This project is provided as-is for personal use.

