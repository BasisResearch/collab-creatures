from pathlib import Path


def find_repo_root() -> Path:
    return Path(__file__).parent.parent.parent
