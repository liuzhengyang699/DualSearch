import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROMPT_TEMPLATE = """<image>
Answer the question about the image. You can reason inside <think>...</think>. If you need information about the image, call <vision_search>image=1</vision_search>. If you need more textual knowledge after receiving vision information, call <search>your query</search>. Return the final answer inside <answer>...</answer>.
Question: {question}"""


def split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def dedup(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def load_existing_images(vision_corpus_path: Path) -> dict[str, str]:
    image_paths: dict[str, str] = {}
    with vision_corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_id = str(item.get("image_id", "")).strip()
            image_path = str(item.get("image", "")).strip()
            if image_id and image_path and Path(image_path).exists():
                image_paths[image_id] = image_path
    return image_paths


def choose_first_existing_image(image_ids: list[str], existing_images: dict[str, str]) -> tuple[str, str] | None:
    for image_id in image_ids:
        image_path = existing_images.get(image_id)
        if image_path:
            return image_id, image_path
    return None


def make_sample(row: pd.Series, source_index: int, image_id: str, image_path: str) -> dict[str, Any]:
    answers = dedup(split_pipe(row.get("answer", "")))
    question = str(row.get("question", "")).strip()
    prompt = DEFAULT_PROMPT_TEMPLATE.format(question=question)

    return {
        "data_source": "dual_search",
        "prompt": [{"role": "user", "content": prompt}],
        "images": [{"image": image_path}],
        "ability": "vision-search",
        "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
        "extra_info": {
            "source_row_index": int(source_index),
            "dataset_name": str(row.get("dataset_name", "")),
            "question_type": str(row.get("question_type", "")),
            "wikipedia_title": str(row.get("wikipedia_title", "")),
            "wikipedia_url": str(row.get("wikipedia_url", "")),
            "image_id": image_id,
            "all_image_ids": split_pipe(row.get("dataset_image_ids", "")),
            "answer": str(row.get("answer", "")),
            "evidence": str(row.get("evidence", "")),
            "evidence_section_id": str(row.get("evidence_section_id", "")),
            "evidence_section_title": str(row.get("evidence_section_title", "")),
        },
    }


def stratified_split(samples: list[dict[str, Any]], seed: int, test_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_type[sample["extra_info"].get("question_type", "")].append(sample)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for question_type in sorted(by_type):
        group = by_type[question_type]
        rng.shuffle(group)
        test_count = int(round(len(group) * test_ratio))
        if group and test_count == 0:
            test_count = 1
        test_rows.extend(group[:test_count])
        train_rows.extend(group[test_count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def attach_split_metadata(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    for index, row in enumerate(rows):
        row["extra_info"]["split"] = split
        row["extra_info"]["index"] = index
    return rows


def validate_sample(row: dict[str, Any]) -> None:
    prompt_text = "".join(message.get("content", "") for message in row["prompt"])
    image_count = len(row["images"])
    placeholder_count = prompt_text.count("<image>")
    if image_count != placeholder_count:
        raise ValueError(f"image count {image_count} does not match <image> count {placeholder_count}")
    targets = row["reward_model"]["ground_truth"]["target"]
    if not targets:
        raise ValueError("reward_model.ground_truth.target is empty")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        validate_sample(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def build_parquets(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv_path)
    vision_corpus_path = Path(args.vision_corpus_path)
    output_dir = Path(args.output_dir)

    existing_images = load_existing_images(vision_corpus_path)
    df = pd.read_csv(csv_path)

    samples = []
    skipped_no_image = 0
    skipped_wrong_dataset = 0
    for source_index, row in df.iterrows():
        if row.get("dataset_name") != "inaturalist":
            skipped_wrong_dataset += 1
            continue
        image_ids = split_pipe(row.get("dataset_image_ids", ""))
        chosen = choose_first_existing_image(image_ids, existing_images)
        if chosen is None:
            skipped_no_image += 1
            continue
        image_id, image_path = chosen
        samples.append(make_sample(row, int(source_index), image_id, image_path))

    train_rows, test_rows = stratified_split(samples, seed=args.seed, test_ratio=args.test_ratio)
    train_rows = attach_split_metadata(train_rows, "train")
    test_rows = attach_split_metadata(test_rows, "test")

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    write_parquet(train_rows, train_path)
    write_parquet(test_rows, test_path)

    return {
        "csv_path": str(csv_path),
        "vision_corpus_path": str(vision_corpus_path),
        "output_dir": str(output_dir),
        "available_image_ids": len(existing_images),
        "samples": len(samples),
        "train": len(train_rows),
        "test": len(test_rows),
        "skipped_wrong_dataset": skipped_wrong_dataset,
        "skipped_no_existing_image": skipped_no_image,
        "question_type_train": pd.Series([row["extra_info"]["question_type"] for row in train_rows]).value_counts().to_dict(),
        "question_type_test": pd.Series([row["extra_info"]["question_type"] for row in test_rows]).value_counts().to_dict(),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EVQA-iNaturalist veRL parquet files.")
    parser.add_argument("--csv_path", default="/data2/lzy/data/raw/val.csv")
    parser.add_argument("--vision_corpus_path", default="/data2/lzy/data/processed/evqa_val_vision_corpus.jsonl")
    parser.add_argument("--output_dir", default="data/evqa_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    report = build_parquets(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
