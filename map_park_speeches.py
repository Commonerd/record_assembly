"""Apply House-specific member metadata to the existing Park-period speech CSV."""

import csv
import os
import shutil
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

from extract_hwp_v2_park import (
    INPUT_DIR,
    MEMBER_CSV_SUFFIX,
    map_speaker_metadata,
    normalize_name,
    parse_assembly_number,
    read_member_csv,
)

INPUT_CSV = Path("hwp_record_assembly_speeches_박정희시기.csv")
METADATA_COLUMNS = {
    "정당": "party",
    "지역": "region",
    "성별": "gender",
    "당선횟수": "election_count",
    "당선방법": "election_method",
}


def load_member_mappings(input_dir: str = INPUT_DIR) -> dict[int, dict]:
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"의원-당적 입력 폴더가 없습니다: {input_path.resolve()}")

    suffix = unicodedata.normalize("NFC", MEMBER_CSV_SUFFIX).casefold()
    mappings = {}
    for path in sorted(input_path.iterdir()):
        normalized_name = unicodedata.normalize("NFC", path.name)
        if not path.is_file() or not normalized_name.casefold().endswith(suffix):
            continue

        assembly_no = parse_assembly_number(normalized_name)
        if assembly_no is None:
            continue
        mappings[assembly_no] = read_member_csv(path)

    if not mappings:
        raise ValueError(f"의원-당적 CSV를 찾지 못했습니다: {input_path.resolve()}")
    return mappings


def remap_speeches(input_csv: Path = INPUT_CSV) -> tuple[int, int, Path]:
    if not input_csv.is_file():
        raise FileNotFoundError(f"발언 CSV가 없습니다: {input_csv.resolve()}")

    mappings = load_member_mappings()
    temporary_path = None
    backup_path = input_csv.with_name(
        f"{input_csv.name}.{datetime.now():%Y%m%d_%H%M%S}.bak"
    )
    changed_rows = 0
    matched_rows = 0

    try:
        with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, None)
            if not header:
                raise ValueError(f"CSV 헤더를 읽지 못했습니다: {input_csv}")

            indices = {normalize_name(name): i for i, name in enumerate(header)}
            missing = [name for name in ("Title", "Speaker") if name not in indices]
            if missing:
                raise ValueError("필수 열이 없습니다: " + ", ".join(missing))

            added_columns = [
                name for name in METADATA_COLUMNS if name not in indices
            ]
            output_header = header + added_columns
            indices.update(
                {name: len(header) + offset for offset, name in enumerate(added_columns)}
            )

            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{input_csv.stem}_",
                suffix=".tmp.csv",
                dir=str(input_csv.parent),
            )
            os.close(fd)
            temporary_path = Path(temporary_name)

            with temporary_path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.writer(
                    target,
                    quoting=csv.QUOTE_MINIMAL,
                    lineterminator="\n",
                )
                writer.writerow(output_header)

                for row in reader:
                    if len(row) < len(header):
                        row.extend([""] * (len(header) - len(row)))
                    row.extend([""] * len(added_columns))

                    title = row[indices["Title"]]
                    speaker = row[indices["Speaker"]]
                    assembly_no = parse_assembly_number(title)
                    metadata = map_speaker_metadata(speaker, assembly_no, mappings)

                    has_match = any(metadata.values())
                    if has_match:
                        matched_rows += 1

                    row_changed = False
                    for column, metadata_key in METADATA_COLUMNS.items():
                        index = indices[column]
                        if not row[index].strip() and metadata[metadata_key]:
                            row[index] = metadata[metadata_key]
                            row_changed = True
                    if row_changed:
                        changed_rows += 1

                    writer.writerow(row)

        shutil.copy2(input_csv, backup_path)
        os.replace(temporary_path, input_csv)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return matched_rows, changed_rows, backup_path


def main():
    matched_rows, changed_rows, backup_path = remap_speeches()
    print(f"의원 정보와 매칭된 발언 행: {matched_rows:,}")
    print(f"빈 의원 정보가 채워진 행: {changed_rows:,}")
    print(f"백업 파일: {backup_path}")
    print(f"갱신 파일: {INPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
