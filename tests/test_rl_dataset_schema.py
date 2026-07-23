import datasets
import pandas as pd

from verl.utils.dataset.rl_dataset import _validate_dual_search_rl_schema


def _rl_row(**overrides):
    query_images = [
        {
            "image_index": 1,
            "dataset_image_id": "101",
            "image_key": "inaturalist:101",
            "image": "/tmp/101.jpg",
            "source_file_name": "train/101.jpg",
            "source_split": "train",
        },
        {
            "image_index": 2,
            "dataset_image_id": "102",
            "image_key": "inaturalist:102",
            "image": "/tmp/102.jpg",
            "source_file_name": "train/102.jpg",
            "source_split": "train",
        },
    ]
    row = {
        "schema_version": 2,
        "data_source": "dual_search",
        "sample_id": "evqa:parent",
        "category_key": "inaturalist:category-10",
        "dataset_category_id": "10",
        "query_images": query_images,
        "image_keys": ["inaturalist:101", "inaturalist:102"],
        "image_count": 2,
        "images": [{"image": "/tmp/101.jpg"}, {"image": "/tmp/102.jpg"}],
    }
    row.update(overrides)
    return row


def test_rl_loader_schema_accepts_ordered_multi_image_v2():
    _validate_dual_search_rl_schema(datasets.Dataset.from_list([_rl_row()]))


def test_rl_loader_schema_accepts_v2_parquet_roundtrip(tmp_path):
    path = tmp_path / "rl_v2.parquet"
    pd.DataFrame([_rl_row()]).to_parquet(path, index=False)
    dataset = datasets.load_dataset("parquet", data_files=str(path), split="train")

    _validate_dual_search_rl_schema(dataset)


def test_rl_loader_schema_rejects_v1():
    dataset = datasets.Dataset.from_list([_rl_row(schema_version=1)])

    try:
        _validate_dual_search_rl_schema(dataset)
    except ValueError as exc:
        assert "schema_version must be 2" in str(exc)
        assert "Rerun split and corpus" in str(exc)
    else:
        raise AssertionError("legacy DualSearch RL rows must be rejected")


def test_rl_loader_schema_rejects_misaligned_images():
    dataset = datasets.Dataset.from_list(
        [
            _rl_row(
                image_keys=["inaturalist:102", "inaturalist:101"],
            )
        ]
    )

    try:
        _validate_dual_search_rl_schema(dataset)
    except ValueError as exc:
        assert "image position 1" in str(exc)
        assert "Rerun split and corpus" in str(exc)
    else:
        raise AssertionError("misaligned DualSearch RL image lists must be rejected")


def test_rl_loader_schema_rejects_obsolete_scalar_image_identity():
    dataset = datasets.Dataset.from_list(
        [_rl_row(image_key="inaturalist:legacy-scalar")]
    )

    try:
        _validate_dual_search_rl_schema(dataset)
    except ValueError as exc:
        assert "obsolete scalar image fields" in str(exc)
        assert "image_key" in str(exc)
        assert "Rerun split and corpus" in str(exc)
    else:
        raise AssertionError("schema v2 RL rows must not retain scalar image identity")
