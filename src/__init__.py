from __future__ import annotations

"""
Project package initializer.

Loads environment variables from a local .env file (if present) so that
secrets like the Basketball Reference cookie do not need to be hard-coded
in the code.
"""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)


