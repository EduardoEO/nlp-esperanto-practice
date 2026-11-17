from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def nlp():
    import esperanto_nlp
    return esperanto_nlp.load()


@pytest.fixture(scope="session")
def mini_dataset_path():
    return FIXTURES / "mini_dataset.csv"
