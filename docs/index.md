---
layout: home

hero:
  name: "Ask LLM"
  text: "A flexible CLI for LLM APIs"
  tagline: Send prompts, translate documents, explain papers, and chat interactively with multiple LLM providers.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: Commands
      link: /commands/

features:
  - title: Multiple Providers
    details: Supports OpenAI-compatible APIs including DeepSeek, Qwen, and more via llm-engine.
  - title: Interactive Chat
    details: Multi-turn conversations with shell command execution, history search, and meta commands.
  - title: Translation
    details: Translate Markdown, plain text, and Jupyter notebooks with glossary support and chunking.
  - title: Paper Explanation
    details: Split academic papers by headings and generate section-by-section explanations.
  - title: Batch Processing
    details: Run the same prompt across multiple inputs or models concurrently with retry logic.
  - title: Markdown Formatting
    details: Use LLMs to normalize Markdown heading levels across files and directories.
---

## Quick Start

Install and configure Ask LLM in minutes:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Create a configuration file
ask-llm config init

# Edit the file with your API keys, then verify
ask-llm config test

# Ask a question
ask-llm "What is the capital of France?"
```

## License

Ask LLM is released under the [MIT License](https://github.com/houhuawei23/ask_llm/blob/main/LICENSE).
