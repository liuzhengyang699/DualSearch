import argparse
import csv
import io
import json
import os
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


INAT_VAL_TAR_SIZE = 8931661582


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url.rstrip("/")


def split_pipe(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def row_wiki_pairs(row: dict[str, str]) -> list[tuple[str, str]]:
    titles = split_pipe(row.get("wikipedia_title", ""))
    urls = split_pipe(row.get("wikipedia_url", ""))
    pairs = []
    for idx, url in enumerate(urls):
        title = titles[idx] if idx < len(titles) else (titles[0] if titles else "")
        pairs.append((title, url))
    if not pairs and titles:
        pairs.append((titles[0], ""))
    return pairs


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_evqa_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_inat_metadata(val_json_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    with val_json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    images = {str(item["id"]): item for item in data["images"]}
    categories = {str(item["id"]): item for item in data["categories"]}

    image_to_category: dict[str, str] = {}
    for ann in data.get("annotations", []):
        image_to_category[str(ann["image_id"])] = str(ann["category_id"])
    for image_id, image in images.items():
        category_id = image_to_category.get(image_id)
        image["category_id"] = category_id
        image["category"] = categories.get(category_id, {})
    return images, categories


def choose_inat_tar(raw_dir: Path) -> Path:
    final = raw_dir / "inat2021" / "val.tar.gz"
    incomplete = raw_dir / "inat2021" / "val.tar.gz.incomplete"
    if final.exists():
        return final
    if incomplete.exists() and incomplete.stat().st_size == INAT_VAL_TAR_SIZE:
        return incomplete
    raise FileNotFoundError(
        "Could not find a complete iNat val tar. Expected "
        f"{final} or a complete {incomplete}."
    )


def choose_kb_zip(raw_dir: Path, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            raw_dir / "encyclopedic_kb_wiki.zip.incomplete",
            raw_dir / "encyclopedic_kb_wiki.zip",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                if names and names[0].endswith(".json"):
                    return path
        except zipfile.BadZipFile:
            continue
    raise FileNotFoundError("Could not find a valid encyclopedic_kb_wiki zip file.")


def build_inat_vision_rows(
    evqa_rows: list[dict[str, str]],
    inat_images: dict[str, dict[str, Any]],
    image_root: Path,
    members_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_image: dict[str, dict[str, Any]] = {}
    title_by_image: dict[str, set[str]] = defaultdict(set)
    wiki_pairs_by_image: dict[str, set[tuple[str, str]]] = defaultdict(set)
    missing_ids: set[str] = set()
    referenced = 0

    for row in evqa_rows:
        if row.get("dataset_name") != "inaturalist":
            continue
        wiki_pairs = row_wiki_pairs(row)
        for image_id in split_pipe(row.get("dataset_image_ids", "")):
            referenced += 1
            image = inat_images.get(image_id)
            if not image:
                missing_ids.add(image_id)
                continue
            per_image.setdefault(image_id, image)
            for title, url in wiki_pairs:
                if title:
                    title_by_image[image_id].add(title)
                if url:
                    wiki_pairs_by_image[image_id].add((title, url))

    members = sorted({image["file_name"] for image in per_image.values()})
    members_path.parent.mkdir(parents=True, exist_ok=True)
    members_path.write_text("\n".join(members) + "\n", encoding="utf-8")

    rows = []
    for image_id, image in sorted(per_image.items(), key=lambda item: int(item[0])):
        category = image.get("category") or {}
        titles = sorted(title_by_image.get(image_id) or [])
        wiki_pairs = sorted(wiki_pairs_by_image.get(image_id) or [])
        primary_title = category.get("name") or (titles[0] if titles else image_id)
        image_path = image_root / image["file_name"]
        primary_url = ""
        for title, url in wiki_pairs:
            if title == primary_title:
                primary_url = url
                break
        if not primary_url and wiki_pairs:
            primary_url = wiki_pairs[0][1]

        taxonomy_parts = [
            category.get("kingdom"),
            category.get("phylum"),
            category.get("class"),
            category.get("order"),
            category.get("family"),
            category.get("genus"),
            category.get("specific_epithet"),
        ]
        taxonomy = " > ".join(part for part in taxonomy_parts if part)
        caption_bits = [
            f"Entity: {primary_title}.",
        ]
        if category.get("name"):
            caption_bits.append(f"Scientific name: {category['name']}.")
        if category.get("common_name"):
            caption_bits.append(f"Common name: {category['common_name']}.")
        if taxonomy:
            caption_bits.append(f"Taxonomy: {taxonomy}.")
        caption = " ".join(caption_bits)

        rows.append(
            {
                "id": f"inaturalist:{image_id}",
                "dataset_name": "inaturalist",
                "image_id": image_id,
                "image": str(image_path),
                "source_file_name": image["file_name"],
                "title": primary_title,
                "wikipedia_url": primary_url,
                "all_wikipedia_titles": titles,
                "caption": caption,
                "contents": f'"{primary_title}"\n{caption}',
            }
        )

    report = {
        "inat_reference_count": referenced,
        "inat_unique_images": len(per_image),
        "inat_missing_image_ids": sorted(missing_ids),
        "tar_member_list": str(members_path),
    }
    return rows, report


class JsonCharReader:
    def __init__(self, stream: io.TextIOBase, chunk_size: int = 1 << 20):
        self.stream = stream
        self.chunk_size = chunk_size
        self.buf = ""
        self.pos = 0

    def read_char(self) -> str:
        if self.pos >= len(self.buf):
            self.buf = self.stream.read(self.chunk_size)
            self.pos = 0
            if not self.buf:
                return ""
        ch = self.buf[self.pos]
        self.pos += 1
        return ch


def skip_ws(reader: JsonCharReader) -> str:
    while True:
        ch = reader.read_char()
        if not ch or not ch.isspace():
            return ch


def read_string(reader: JsonCharReader) -> str:
    raw = ['"']
    escape = False
    while True:
        ch = reader.read_char()
        if not ch:
            raise ValueError("Unexpected EOF inside JSON string.")
        raw.append(ch)
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            return json.loads("".join(raw))


def skip_json_string(reader: JsonCharReader):
    escape = False
    while True:
        ch = reader.read_char()
        if not ch:
            raise ValueError("Unexpected EOF inside JSON string.")
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            return


def skip_value(reader: JsonCharReader, first: str):
    if first == '"':
        skip_json_string(reader)
        return
    if first not in "{[":
        while True:
            ch = reader.read_char()
            if not ch or ch in ",}]":
                reader.pos -= 1
                return

    stack = [first]
    while stack:
        ch = reader.read_char()
        if not ch:
            raise ValueError("Unexpected EOF while skipping JSON value.")
        if ch == '"':
            skip_json_string(reader)
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack[-1] == "[":
            stack.pop()


def collect_value(reader: JsonCharReader, first: str) -> str:
    out = [first]
    if first == '"':
        escape = False
        while True:
            ch = reader.read_char()
            if not ch:
                raise ValueError("Unexpected EOF inside JSON string.")
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                return "".join(out)

    if first not in "{[":
        while True:
            ch = reader.read_char()
            if not ch or ch in ",}]":
                reader.pos -= 1
                return "".join(out)
            out.append(ch)

    stack = [first]
    while stack:
        ch = reader.read_char()
        if not ch:
            raise ValueError("Unexpected EOF while collecting JSON value.")
        out.append(ch)
        if ch == '"':
            escape = False
            while True:
                ch = reader.read_char()
                if not ch:
                    raise ValueError("Unexpected EOF inside JSON string.")
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    break
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack[-1] == "[":
            stack.pop()
    return "".join(out)


def iter_selected_top_level_items(zip_path: Path, selected_norm_urls: set[str]):
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            reader = JsonCharReader(text)
            first = skip_ws(reader)
            if first != "{":
                raise ValueError("KB JSON must be a top-level object.")

            while True:
                ch = skip_ws(reader)
                if ch == "}":
                    return
                if ch != '"':
                    raise ValueError(f"Expected object key, got {ch!r}.")
                url = read_string(reader)
                ch = skip_ws(reader)
                if ch != ":":
                    raise ValueError(f"Expected ':', got {ch!r}.")
                first_value = skip_ws(reader)
                norm = normalize_url(url)
                if norm in selected_norm_urls:
                    yield url, json.loads(collect_value(reader, first_value))
                else:
                    skip_value(reader, first_value)

                ch = skip_ws(reader)
                if ch == ",":
                    continue
                if ch == "}":
                    return
                raise ValueError(f"Expected ',' or closing brace, got {ch!r}.")


def build_text_rows(evqa_rows: list[dict[str, str]], kb_zip: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    title_by_norm_url = {}
    original_by_norm_url = {}
    for row in evqa_rows:
        for title, url in row_wiki_pairs(row):
            norm = normalize_url(url)
            if not norm:
                continue
            title_by_norm_url.setdefault(norm, title)
            original_by_norm_url.setdefault(norm, url.strip())

    selected = set(title_by_norm_url)
    found = set()
    rows = []

    for url, page in iter_selected_top_level_items(kb_zip, selected):
        norm = normalize_url(url)
        found.add(norm)
        title = page.get("title") or title_by_norm_url.get(norm) or url.rsplit("/", 1)[-1]
        section_texts = page.get("section_texts") or []
        section_titles = page.get("section_titles") or []

        for section_id, text in enumerate(section_texts):
            text = (text or "").strip()
            if not text:
                continue
            section_title = ""
            if section_id < len(section_titles):
                section_title = (section_titles[section_id] or "").strip()
            display_title = title if not section_title else f"{title} :: {section_title}"
            rows.append(
                {
                    "id": f"{norm}#section-{section_id}",
                    "url": url,
                    "title": title,
                    "section_id": section_id,
                    "section_title": section_title,
                    "contents": f'"{display_title}"\n{text}',
                }
            )

    report = {
        "requested_wiki_urls": len(selected),
        "found_wiki_urls": len(found),
        "missing_wiki_urls": [original_by_norm_url[url] for url in sorted(selected - found)],
        "text_sections": len(rows),
    }
    return rows, report


def extract_inat_images(tar_path: Path, image_root: Path, members_path: Path):
    image_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tar",
        "-xzf",
        str(tar_path),
        "-C",
        str(image_root),
        "-T",
        str(members_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare EVQA retrieval corpora for DualSearch.")
    parser.add_argument("--data_root", type=Path, default=Path("/data2/lzy/data"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--kb_zip", default=None)
    parser.add_argument("--extract_inat_images", action="store_true")
    parser.add_argument("--skip_text_corpus", action="store_true")
    parser.add_argument("--skip_vision_corpus", action="store_true")
    args = parser.parse_args()

    raw_dir = args.data_root / "raw"
    processed_dir = args.data_root / "processed"
    image_root = processed_dir / "images" / "inat2021_val"

    evqa_csv = raw_dir / f"{args.split}.csv"
    evqa_rows = load_evqa_rows(evqa_csv)
    report: dict[str, Any] = {"split": args.split, "evqa_rows": len(evqa_rows)}

    if not args.skip_vision_corpus:
        inat_images, _ = load_inat_metadata(raw_dir / "inat2021" / "val.json")
        members_path = processed_dir / f"evqa_{args.split}_inat_needed_tar_members.txt"
        vision_rows, vision_report = build_inat_vision_rows(
            evqa_rows,
            inat_images,
            image_root,
            members_path,
        )
        report.update(vision_report)
        vision_path = processed_dir / f"evqa_{args.split}_vision_corpus.jsonl"
        report["vision_corpus_path"] = str(vision_path)
        report["vision_corpus_rows"] = write_jsonl(vision_path, vision_rows)

        if args.extract_inat_images:
            tar_path = choose_inat_tar(raw_dir)
            extract_inat_images(tar_path, image_root, members_path)
            existing = sum(1 for row in vision_rows if Path(row["image"]).exists())
            report["inat_extracted_image_root"] = str(image_root)
            report["vision_corpus_existing_images"] = existing

    if not args.skip_text_corpus:
        kb_zip = choose_kb_zip(raw_dir, args.kb_zip)
        text_rows, text_report = build_text_rows(evqa_rows, kb_zip)
        text_path = processed_dir / f"evqa_{args.split}_text_corpus.jsonl"
        report.update(text_report)
        report["kb_zip_path"] = str(kb_zip)
        report["text_corpus_path"] = str(text_path)
        report["text_corpus_rows"] = write_jsonl(text_path, text_rows)

    report_path = processed_dir / f"evqa_{args.split}_retrieval_prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
