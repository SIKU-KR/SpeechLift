"""API key management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speechlift.ui.protocols import UserInterface


class ApiKeyManager:
    """Manages OpenAI API key retrieval and storage."""

    def __init__(self, config_file: Path) -> None:
        self._config_file = config_file

    def get_api_key(self, ui: UserInterface) -> str:
        """
        Get OpenAI API key from environment, config file, or user input.

        Args:
            ui: User interface for prompting if needed.

        Returns:
            The API key string.

        Raises:
            SystemExit: If the user provides an invalid key.
        """
        # 1. Check environment variable
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key

        # 2. Check config file
        if self._config_file.exists():
            try:
                config = json.loads(self._config_file.read_text())
                if config.get("api_key"):
                    return config["api_key"]
            except (json.JSONDecodeError, IOError):
                pass

        # 3. Prompt user for API key
        api_key = ui.prompt_api_key()

        if not api_key.startswith("sk-"):
            ui.display_error("Invalid API key format (should start with 'sk-')")
            raise SystemExit(1)

        # 4. Ask how to save
        save_method = ui.prompt_save_method()
        self._save_api_key(api_key, save_method, ui)

        return api_key

    def _save_api_key(self, api_key: str, method: str, ui: UserInterface) -> None:
        """Save the API key using the specified method."""
        if method == "1":
            # Save to config file
            self._config_file.write_text(json.dumps({"api_key": api_key}, indent=2))
            self._config_file.chmod(0o600)
            ui.display_success(f"API key saved to {self._config_file}")

        elif method == "2":
            # Add to shell profile
            shell = os.environ.get("SHELL", "/bin/zsh")
            if "zsh" in shell:
                profile_path = Path.home() / ".zshrc"
            else:
                profile_path = Path.home() / ".bashrc"

            export_line = f'\nexport OPENAI_API_KEY="{api_key}"\n'

            with open(profile_path, "a") as f:
                f.write(export_line)

            ui.display_success(f"Added to {profile_path}")
            ui.display_warning(f"Run 'source {profile_path}' or restart terminal to apply")

        else:
            ui.display_warning("API key will only be used for this session")
