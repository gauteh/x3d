"""The code blocks of README.md run as written (against the public RGW store)."""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.rgw

README = (Path(__file__).parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, re.S)


def test_readme_has_snippets():
    assert BLOCKS


@pytest.mark.parametrize("code", BLOCKS, ids=lambda c: c.splitlines()[1][:40])
def test_readme_snippet_runs(code, rgw_url):
    ns = {}
    exec(compile(code, "README.md", "exec"), ns)
    assert "hs" in ns["ds"]
    assert ns["ds"].hs.isel(time=0).size > 0
