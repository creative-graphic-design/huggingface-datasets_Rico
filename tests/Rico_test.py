import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "Rico.py"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames=(
        "dataset_task",
        "expected_num_train",
        "expected_num_valid",
        "expected_num_test",
    ),
    argvalues=(
        ("ui-screenshots-and-view-hierarchies", 56322, 3314, 6625),
        ("ui-layout-vectors", 61288, 3606, 7209),
        ("interaction-traces", 8749, 513, 1030),
        # "animations",
        ("ui-screenshots-and-hierarchies-with-semantic-annotations", 56322, 3314, 6625),
    ),
)
def test_load_dataset(
    dataset_path: str,
    dataset_task: str,
    expected_num_train: int,
    expected_num_valid: int,
    expected_num_test: int,
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_task)
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["test"].num_rows == expected_num_test


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames=("dataset_task", "expected_num_data"),
    argvalues=(
        ("ui-metadata", 66261),
        ("play-store-metadata", 9384 - 1),  # There is one invalid data
    ),
)
def test_load_metadata(dataset_path: str, dataset_task: str, expected_num_data: int):
    metadata = ds.load_dataset(path=dataset_path, name=dataset_task)
    assert metadata["metadata"].num_rows == expected_num_data
