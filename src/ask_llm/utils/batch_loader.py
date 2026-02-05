"""Batch configuration file loader."""

from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger

from ask_llm.core.batch import BatchTask, ModelConfig
from ask_llm.utils.file_handler import FileHandler


class BatchConfigLoader:
    """Load and parse batch configuration files."""

    @classmethod
    def load(cls, config_path: str) -> Dict[str, Any]:
        """
        Load batch configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary with keys:
            - 'mode': 'prompt-contents' or 'prompt-content-pairs'
            - 'provider_models': List of ModelConfig (optional)
            - 'tasks': List of BatchTask

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        file_path = Path(config_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Batch config file not found: {config_path}")

        logger.debug(f"Loading batch config from: {config_path}")

        # Read file content
        content = FileHandler.read(config_path)

        # Try to detect format
        if content.strip().startswith("---"):
            # Multi-document format (prompt-content-pairs)
            return cls._load_prompt_content_pairs(content, config_path)
        else:
            # Single document format (prompt-contents)
            return cls._load_prompt_contents(content, config_path)

    @classmethod
    def _load_prompt_contents(cls, content: str, _config_path: str) -> Dict[str, Any]:
        """
        Load prompt-contents format.

        Format:
        ```yaml
        provider-models:  # optional
          - provider: deepseek
            models:
              - model: deepseek-chat
              - model: deepseek-reasoner
                temperature: 1.0
                top_p: 0.9
        prompt: <统一提示词>
        contents:
          - <内容1>
          - <内容2>
        ```

        Args:
            content: YAML content
            config_path: Config file path (for error messages)

        Returns:
            Configuration dictionary
        """
        try:
            data = yaml.safe_load(content)
            if not data:
                raise ValueError("Configuration file is empty")

            # Extract prompt
            if "prompt" not in data:
                raise ValueError("Missing required field: 'prompt'")

            prompt = data["prompt"]
            if not isinstance(prompt, str):
                raise ValueError("'prompt' must be a string")

            # Extract contents
            if "contents" not in data:
                raise ValueError("Missing required field: 'contents'")

            contents = data["contents"]
            if not isinstance(contents, list):
                raise ValueError("'contents' must be a list")

            if not contents:
                raise ValueError("'contents' list cannot be empty")

            # Extract provider-models (optional)
            provider_models: List[ModelConfig] = []
            if "provider-models" in data:
                provider_models = cls._parse_provider_models(data["provider-models"])

            # Create tasks
            tasks = []
            for idx, content_item in enumerate(contents):
                if not isinstance(content_item, str):
                    raise ValueError(f"Content item {idx} must be a string")
                tasks.append(
                    BatchTask(
                        task_id=idx + 1,
                        prompt=prompt,
                        content=content_item,
                    )
                )

            return {
                "mode": "prompt-contents",
                "provider_models": provider_models,
                "tasks": tasks,
            }

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}") from e

    @classmethod
    def _load_prompt_content_pairs(cls, content: str, _config_path: str) -> Dict[str, Any]:
        """
        Load prompt-content-pairs format.

        Format:
        ```yaml
        ---
        prompt: <提示词1>
        content: <内容1>
        ---
        prompt: <提示词2>
        content: <内容2>
        ```

        Args:
            content: YAML content
            config_path: Config file path (for error messages)

        Returns:
            Configuration dictionary
        """
        try:
            documents = list(yaml.safe_load_all(content))
            if not documents:
                raise ValueError("Configuration file is empty")

            # Extract provider-models from first document if present
            provider_models: List[ModelConfig] = []
            first_doc = documents[0]
            if isinstance(first_doc, dict) and "provider-models" in first_doc:
                provider_models = cls._parse_provider_models(first_doc["provider-models"])
                # Remove provider-models from first doc to avoid confusion
                documents = documents[1:] if len(documents) > 1 else []

            # Parse each document as a task
            tasks = []
            for idx, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    raise ValueError(f"Document {idx + 1} must be a dictionary")

                if "prompt" not in doc:
                    raise ValueError(f"Document {idx + 1} missing required field: 'prompt'")

                if "content" not in doc:
                    raise ValueError(f"Document {idx + 1} missing required field: 'content'")

                prompt = doc["prompt"]
                content_item = doc["content"]

                if not isinstance(prompt, str):
                    raise ValueError(f"Document {idx + 1}: 'prompt' must be a string")

                if not isinstance(content_item, str):
                    raise ValueError(f"Document {idx + 1}: 'content' must be a string")

                tasks.append(
                    BatchTask(
                        task_id=idx + 1,
                        prompt=prompt,
                        content=content_item,
                    )
                )

            if not tasks:
                raise ValueError("No valid tasks found in configuration")

            return {
                "mode": "prompt-content-pairs",
                "provider_models": provider_models,
                "tasks": tasks,
            }

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}") from e

    @classmethod
    def _parse_provider_models(cls, provider_models_data: Any) -> List[ModelConfig]:
        """
        Parse provider-models configuration.

        Args:
            provider_models_data: Provider models data from YAML

        Returns:
            List of ModelConfig objects
        """
        if not isinstance(provider_models_data, list):
            raise ValueError("'provider-models' must be a list")

        model_configs: List[ModelConfig] = []

        for provider_item in provider_models_data:
            if not isinstance(provider_item, dict):
                raise ValueError("Each provider-models item must be a dictionary")

            if "provider" not in provider_item:
                raise ValueError("Provider item missing 'provider' field")

            provider_name = provider_item["provider"]
            if not isinstance(provider_name, str):
                raise ValueError("'provider' must be a string")

            if "models" not in provider_item:
                raise ValueError(f"Provider '{provider_name}' missing 'models' field")

            models = provider_item["models"]
            if not isinstance(models, list):
                raise ValueError(f"Provider '{provider_name}': 'models' must be a list")

            for model_item in models:
                if isinstance(model_item, str):
                    # Simple string format: just model name
                    model_configs.append(ModelConfig(provider=provider_name, model=model_item))
                elif isinstance(model_item, dict):
                    # Dictionary format: model with parameters
                    if "model" not in model_item:
                        raise ValueError(
                            f"Model item in provider '{provider_name}' missing 'model' field"
                        )

                    model_name = model_item["model"]
                    if not isinstance(model_name, str):
                        raise ValueError("'model' must be a string")

                    model_config = ModelConfig(
                        provider=provider_name,
                        model=model_name,
                        temperature=model_item.get("temperature"),
                        top_p=model_item.get("top_p"),
                        max_tokens=model_item.get("max_tokens"),
                    )
                    model_configs.append(model_config)
                else:
                    raise ValueError(f"Invalid model item format in provider '{provider_name}'")

        return model_configs
