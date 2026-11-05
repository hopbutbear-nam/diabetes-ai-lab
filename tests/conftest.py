import json, pathlib
import pytest

@pytest.fixture
def sample_profile():
    p = pathlib.Path(__file__).parent / "fixtures" / "sample_profile.json"
    return json.loads(p.read_text())
