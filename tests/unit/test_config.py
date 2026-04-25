"""Sanity check: the config singleton loads with a stub API key.

If this fails, the conftest env stub isn't taking effect — anything
downstream that imports `horcrux.config` will fail the same way.
"""

import pytest

pytestmark = pytest.mark.unit


def test_settings_loads_with_stub_key():
    from horcrux.config import settings

    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-stub"


def test_default_corpus_path_points_at_data_lake():
    from horcrux.config import settings

    assert settings.corpus_path == "data_lake/corpus.pdf"


def test_qdrant_defaults():
    from horcrux.config import settings

    assert settings.qdrant.host == "localhost"
    assert settings.qdrant.port == 6333
    assert settings.qdrant.paragraphs_collection == "hp_paragraphs"
    assert settings.qdrant.chapters_collection == "hp_chapters"


def test_litellm_aliases():
    from horcrux.config import settings

    assert settings.litellm.haiku_alias == "haiku"
    assert settings.litellm.sonnet_alias == "sonnet"


def test_secret_str_does_not_leak_in_repr():
    from horcrux.config import settings

    assert "sk-ant-test-stub" not in repr(settings)
    assert "sk-ant-test-stub" not in settings.model_dump_json()
