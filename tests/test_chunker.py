from unittest.mock import patch

from tadbench.utils.chunker import _ensure_punkt


def test_punkt_bootstrap_checks_existing_data_once():
    _ensure_punkt.cache_clear()
    try:
        with patch("nltk.data.find") as find, patch("nltk.download") as download:
            _ensure_punkt()
            _ensure_punkt()
        find.assert_called_once()
        download.assert_not_called()
    finally:
        _ensure_punkt.cache_clear()
