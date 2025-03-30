"""Unit tests for the configuration management module."""

import json
import tempfile
from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel, Field

from llm_rag.utils.config import (
    get_env,
    load_config,
    load_env_config,
    load_json_config,
    load_yaml_config,
    merge_configs,
    validate_config,
)
from llm_rag.utils.errors import ConfigurationError, ErrorCode


class TestConfigModel(BaseModel):
    """Test configuration model."""

    name: str
    value: int
    optional: Optional[str] = None
    nested: Dict[str, Any] = Field(default_factory=dict)


class TestConfigUtils:
    """Test cases for configuration utilities."""

    def test_load_yaml_config_valid(self):
        """Test loading valid YAML configuration."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp:
            temp.write("name: test\nvalue: 123\n")
            temp.flush()

            config = load_yaml_config(temp.name)

            assert config["name"] == "test"
            assert config["value"] == 123

    def test_load_yaml_config_file_not_found(self):
        """Test loading YAML from non-existent file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_config("/path/to/nonexistent.yaml")

        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND

    def test_load_yaml_config_invalid_format(self):
        """Test loading invalid YAML format."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as temp:
            temp.write("invalid: yaml:\n  - missing\n  indentation")
            temp.flush()

            with pytest.raises(ConfigurationError) as exc_info:
                load_yaml_config(temp.name)

            assert exc_info.value.error_code == ErrorCode.INVALID_FILE_FORMAT

    def test_load_json_config_valid(self):
        """Test loading valid JSON configuration."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as temp:
            json.dump({"name": "test", "value": 123}, temp)
            temp.flush()

            config = load_json_config(temp.name)

            assert config["name"] == "test"
            assert config["value"] == 123

    def test_load_json_config_file_not_found(self):
        """Test loading JSON from non-existent file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_json_config("/path/to/nonexistent.json")

        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND

    def test_load_json_config_invalid_format(self):
        """Test loading invalid JSON format."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as temp:
            temp.write('{"invalid": "json",}')  # Extra comma makes this invalid JSON
            temp.flush()

            with pytest.raises(ConfigurationError) as exc_info:
                load_json_config(temp.name)

            assert exc_info.value.error_code == ErrorCode.INVALID_FILE_FORMAT

    def test_load_env_config(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Setup environment variables
        monkeypatch.setenv("TEST_NAME", "test_name")
        monkeypatch.setenv("TEST_VALUE", "123")
        monkeypatch.setenv("TEST_NESTED__KEY", "nested_value")
        monkeypatch.setenv("DIFFERENT_PREFIX", "ignored")

        # Test loading with the TEST prefix
        config = load_env_config("TEST")

        assert config["name"] == "test_name"
        assert config["value"] == "123"
        assert config["nested"]["key"] == "nested_value"
        assert "different_prefix" not in config

    def test_load_env_config_no_lowercase(self, monkeypatch):
        """Test loading environment variables without lowercase conversion."""
        monkeypatch.setenv("TEST_NAME", "test_name")

        config = load_env_config("TEST", lowercase_keys=False)

        assert "NAME" in config
        assert "name" not in config

    def test_validate_config_valid(self):
        """Test validating valid configuration."""
        config_dict = {
            "name": "test",
            "value": 123,
            "optional": "optional_value",
        }

        validated = validate_config(config_dict, TestConfigModel)

        assert isinstance(validated, TestConfigModel)
        assert validated.name == "test"
        assert validated.value == 123
        assert validated.optional == "optional_value"

    def test_validate_config_invalid(self):
        """Test validating invalid configuration."""
        config_dict = {
            "name": "test",
            # Missing required 'value' field
            "optional": "optional_value",
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_dict, TestConfigModel)

        assert exc_info.value.error_code == ErrorCode.INVALID_CONFIG

    def test_merge_configs_flat(self):
        """Test merging flat configuration dictionaries."""
        config1 = {"name": "test1", "value": 1}
        config2 = {"name": "test2", "extra": "extra"}

        merged = merge_configs(config1, config2)

        assert merged["name"] == "test2"  # Overridden by config2
        assert merged["value"] == 1  # From config1
        assert merged["extra"] == "extra"  # From config2

    def test_merge_configs_nested(self):
        """Test merging nested configuration dictionaries."""
        config1 = {
            "name": "test1",
            "nested": {
                "key1": "value1",
                "key2": "value2",
            },
        }
        config2 = {
            "name": "test2",
            "nested": {
                "key2": "new_value2",
                "key3": "value3",
            },
        }

        merged = merge_configs(config1, config2)

        assert merged["name"] == "test2"
        assert merged["nested"]["key1"] == "value1"  # From config1
        assert merged["nested"]["key2"] == "new_value2"  # Overridden by config2
        assert merged["nested"]["key3"] == "value3"  # From config2

    def test_merge_configs_multiple(self):
        """Test merging multiple configuration dictionaries."""
        config1 = {"name": "test1"}
        config2 = {"value": 2}
        config3 = {"name": "test3", "extra": "extra"}

        merged = merge_configs(config1, config2, config3)

        assert merged["name"] == "test3"
        assert merged["value"] == 2
        assert merged["extra"] == "extra"

    def test_load_config_with_default(self):
        """Test loading config with default config."""
        default_config = {"name": "default", "value": 0}

        config = load_config(default_config=default_config)

        assert config["name"] == "default"
        assert config["value"] == 0

    def test_load_config_with_files(self):
        """Test loading config from files."""
        with (
            tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as yaml_file,
            tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as json_file,
        ):
            # Create YAML config
            yaml_file.write("name: yaml\nvalue: 1\n")
            yaml_file.flush()

            # Create JSON config
            json.dump({"name": "json", "extra": "from_json"}, json_file)
            json_file.flush()

            # Load with both files
            config = load_config(config_paths=[yaml_file.name, json_file.name])

            assert config["name"] == "json"  # From JSON, overrides YAML
            assert config["value"] == 1  # From YAML
            assert config["extra"] == "from_json"  # From JSON

    def test_load_config_with_env(self, monkeypatch):
        """Test loading config with environment variables."""
        default_config = {"name": "default", "value": 0}

        # Setup environment variables
        monkeypatch.setenv("TEST_NAME", "from_env")

        config = load_config(default_config=default_config, env_prefix="TEST")

        assert config["name"] == "from_env"  # From env, overrides default
        assert config["value"] == 0  # From default

    def test_load_config_with_validation(self):
        """Test loading config with validation."""
        default_config = {"name": "default", "value": 0}

        config = load_config(default_config=default_config, model_cls=TestConfigModel)

        assert isinstance(config, TestConfigModel)
        assert config.name == "default"
        assert config.value == 0

    def test_load_config_full_chain(self, monkeypatch):
        """Test loading config from all sources."""
        default_config = {"name": "default", "value": 0}

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as yaml_file:
            # Create YAML config
            yaml_file.write("name: yaml\nvalue: 1\n")
            yaml_file.flush()

            # Setup environment variables
            monkeypatch.setenv("TEST_NAME", "from_env")

            # Load with all sources
            config = load_config(
                config_paths=yaml_file.name, env_prefix="TEST", default_config=default_config, model_cls=TestConfigModel
            )

            assert isinstance(config, TestConfigModel)
            assert config.name == "from_env"  # From env, highest priority
            assert config.value == 1  # From YAML, overrides default

    def test_get_env_existing(self, monkeypatch):
        """Test getting existing environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        value = get_env("TEST_VAR")

        assert value == "test_value"

    def test_get_env_with_default(self):
        """Test getting non-existent environment variable with default."""
        value = get_env("NONEXISTENT_VAR", default="default_value")

        assert value == "default_value"

    def test_get_env_required(self, monkeypatch):
        """Test getting required environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        value = get_env("TEST_VAR", required=True)

        assert value == "test_value"

    def test_get_env_required_missing(self):
        """Test getting missing required environment variable."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_env("NONEXISTENT_VAR", required=True)

        assert "Required environment variable" in str(exc_info.value)
