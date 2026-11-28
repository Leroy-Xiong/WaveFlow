"""Configuration loading and command-line override utilities."""

import yaml


def load_config(config_path='./configs/configs.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config, overrides):
    """
    Apply command-line overrides to a config dict.

    Overrides are strings in the form 'section.key=value' with dot-path nesting.
    Values are auto-typed: int, float, bool, 'null' -> None, otherwise str.

    Example:
        overrides = ['sampling.ens_lead_time=24', 'work_dir.prefix=my_prefix']
    """
    if not overrides:
        return config
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'.")
        path, value_str = override.split('=', 1)
        keys = path.strip().split('.')
        value = _parse_override_value(value_str.strip())

        # Navigate to the nested dict and set the value
        d = config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    return config


def _parse_override_value(value_str):
    """Parse a string value into the appropriate Python type."""
    # Handle null/None
    if value_str.lower() in ('null', 'none', ''):
        return None
    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    # Try bool
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    # Fallback to string
    return value_str
