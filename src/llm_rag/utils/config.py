"""Configuration management for the LLM-RAG system.

This module provides utilities for loading and validating configuration from
various sources including environment variables, configuration files, and
command-line arguments. It supports hierarchical configurations with defaults
and overrides.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError

from llm_rag.utils.errors import ConfigurationError, ErrorCode
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for configuration model types
T = TypeVar("T", bound=BaseModel)


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        ConfigurationError: If the file doesn't exist or can't be parsed

    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                error_code=ErrorCode.FILE_NOT_FOUND,
            )

        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ConfigurationError(
                f"Invalid YAML configuration format in {file_path}",
                error_code=ErrorCode.INVALID_FILE_FORMAT,
            )

        logger.debug(f"Loaded configuration from {file_path}")
        return config_dict

    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Error parsing YAML configuration: {str(e)}",
            error_code=ErrorCode.INVALID_FILE_FORMAT,
            original_exception=e,
        )
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Error loading configuration from {file_path}",
            original_exception=e,
        )


def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        ConfigurationError: If the file doesn't exist or can't be parsed

    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                error_code=ErrorCode.FILE_NOT_FOUND,
            )

        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        logger.debug(f"Loaded configuration from {file_path}")
        return config_dict

    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Error parsing JSON configuration: {str(e)}",
            error_code=ErrorCode.INVALID_FILE_FORMAT,
            original_exception=e,
        )
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Error loading configuration from {file_path}",
            original_exception=e,
        )


def load_env_config(prefix: str, lowercase_keys: bool = True) -> Dict[str, str]:
    """Load configuration from environment variables with the given prefix.

    Args:
        prefix: Prefix for environment variables to include
        lowercase_keys: Whether to convert keys to lowercase

    Returns:
        Dictionary containing the configuration from environment variables

    """
    config_dict = {}
    prefix_upper = prefix.upper()

    for key, value in os.environ.items():
        if key.startswith(prefix_upper):
            # Remove prefix and separator
            config_key = (
                key[len(prefix_upper) + 1 :] if key.startswith(f"{prefix_upper}_") else key[len(prefix_upper) :]
            )

            if lowercase_keys:
                config_key = config_key.lower()

            # Convert keys with double underscore to nested dictionaries
            if "__" in config_key:
                parts = config_key.split("__")
                current = config_dict

                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                current[parts[-1]] = value
            else:
                config_dict[config_key] = value

    logger.debug(f"Loaded {len(config_dict)} environment variables with prefix {prefix}")
    return config_dict


def validate_config(config: Dict[str, Any], model_cls: type[T]) -> T:
    """Validate configuration against a Pydantic model.

    Args:
        config: Configuration dictionary to validate
        model_cls: Pydantic model class for validation

    Returns:
        Validated configuration model instance

    Raises:
        ConfigurationError: If validation fails

    """
    try:
        return model_cls(**config)
    except ValidationError as e:
        raise ConfigurationError(
            f"Configuration validation failed: {str(e)}",
            error_code=ErrorCode.INVALID_CONFIG,
            original_exception=e,
            details={"errors": e.errors()},
        )


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Later configurations override earlier ones for top-level keys.
    For nested dictionaries, the dictionaries are merged recursively.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary

    """
    result: Dict[str, Any] = {}

    for config in configs:
        for key, value in config.items():
            # If both are dictionaries, merge recursively
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                # Otherwise, override the value
                result[key] = value

    return result


def load_config(
    config_paths: Optional[Union[str, Path, list[Union[str, Path]]]] = None,
    env_prefix: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
    model_cls: Optional[type[T]] = None,
) -> Union[Dict[str, Any], T]:
    """Load and merge configuration from multiple sources.

    Configuration is loaded in the following order, with later sources
    overriding earlier ones:
    1. Default configuration
    2. Configuration files (in the order provided)
    3. Environment variables

    Args:
        config_paths: Path(s) to configuration files
        env_prefix: Prefix for environment variables to include
        default_config: Default configuration dictionary
        model_cls: Optional Pydantic model for validation

    Returns:
        Merged configuration dictionary or validated model instance

    Raises:
        ConfigurationError: If any configuration source can't be loaded or validation fails

    """
    configs = []

    # Start with default config if provided
    if default_config:
        configs.append(default_config)

    # Load from config files
    if config_paths:
        if not isinstance(config_paths, list):
            config_paths = [config_paths]

        for path in config_paths:
            path_str = str(path)
            if path_str.endswith((".yml", ".yaml")):
                configs.append(load_yaml_config(path))
            elif path_str.endswith(".json"):
                configs.append(load_json_config(path))
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {path}",
                    error_code=ErrorCode.INVALID_FILE_FORMAT,
                )

    # Load from environment variables
    if env_prefix:
        configs.append(load_env_config(env_prefix))

    # Merge all configs
    merged_config = merge_configs(*configs)

    # Validate against model if provided
    if model_cls:
        return validate_config(merged_config, model_cls)

    return merged_config


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get an environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Whether the variable is required

    Returns:
        Environment variable value or default

    Raises:
        ConfigurationError: If the variable is required but not set

    """
    value = os.environ.get(key, default)

    if required and value is None:
        raise ConfigurationError(
            f"Required environment variable not set: {key}",
            error_code=ErrorCode.ENV_VAR_NOT_SET,
        )

    return value
