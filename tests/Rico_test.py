import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "Rico.py"


@pytest.mark.parametrize(
    argnames=("dataset_task"),
    argvalues=(
        # "ui-screenshots-and-view-hierarchies",
        # "ui-layout-vectors",
        "interaction-traces",
        # "animations",
        # "ui-screenshots-and-hierarchies-with-semantic-annotations",
        "ui-metadata",
        "play-store-metadata",
    ),
)
def test_load_dataset(dataset_path: str, dataset_task: str):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_task)
