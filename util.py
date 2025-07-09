import re
from typing import Optional

from rich import print


def combine_test_with_code(main_code: str, test_code: str) -> str:
    """Merge code followed by test, so test can be executed."""
    return f"# Your solution:\n{main_code}\n\n# QA test:\n{test_code}"
