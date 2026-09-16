import re
from pathlib import Path

import bolt

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    # Read rather than tomllib-parse: tomllib is 3.11+, and CI covers 3.10
    match = re.search(r'^version = "(.+)"$', PYPROJECT.read_text(), re.M)
    assert match is not None, f"no version found in {PYPROJECT}"
    return match.group(1)


def test_version_matches_pyproject() -> None:
    assert bolt.__version__ == _pyproject_version()
