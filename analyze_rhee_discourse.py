import csv
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

# Python csv 모듈 기본 필드 길이 제한(131,072자) 해제
csv.field_size_limit(1024 * 1024 * 1024)

# ============================================================
# 설정
# ============================================================

INPUT_CSV = "hwp_record_assembly_speeches_이승만시기.csv"

# ============================================================
# 분석 대상 키워드
# ============================================================

DISCOURSE_KEYWORDS = {
    "발전": {
        "기술": [
            "산업", "공업", "생산", "기계", "기계화", "동력",
            "전력", "발전소", "개발", "건설", "부흥", "재건", "시설",
        ],
        "소득": [
            "물가", "생활비", "임금", "급여", "봉급", "물자", "생계",
            "경제", "재정", "예산", "국민소득", "세입",
        ],
        "노동": [
            "노무자", "근로자", "근로", "노동조합", "노조", "실업",
            "취업", "고용", "근로기준", "노동력", "노동자",
        ],
    },
    "경계": {
        "이민": [
            "교포", "동포", "이주민", "이민자", "해외동포", "재외국민",
            "귀환동포", "재일교포", "재중교포",
        ],
        "이주": [
            "이동", "인구이동", "유입", "유출", "정착", "피난민",
            "난민", "월남민", "실향민",
        ],
        "외국인": [
            "외인", "외래", "화교", "왜인", "미군", "미인", "미국인",
            "양인", "이방인", "외지인",
        ],
    },
}

# 기존 CSV 맨 뒤에 추가되는 열
ADDED_COLUMNS = [
    "발전담론",
    "발전계열",
    "발전키워드",
    "발전키워드수",
    "경계담론",
    "경계계열",
    "경계키워드",
    "경계키워드수",
    "담론분류",
]


def normalize_for_matching(text: str) -> str:
    """키워드 비교용 최소 정규화."""
    if text is None:
        return ""
    text = str(text).replace("\ufeff", "").replace("\u200b", "")
    return re.sub(r"\s+", "", text)


def analyze_content(content: str) -> dict:
    """
    Content만 대상으로 키워드를 탐색한다.

    같은 키워드가 여러 번 나와도 '발전키워드/경계키워드'에는 한 번만 기록한다.
    따라서 키워드수는 '고유하게 검출된 키워드 종류의 수'다.
    """
    normalized = normalize_for_matching(content)

    matched_families = {"발전": [], "경계": []}
    matched_keywords = {"발전": [], "경계": []}

    for discourse_name, families in DISCOURSE_KEYWORDS.items():
        for family_name, keywords in families.items():
            family_hit = False

            for keyword in keywords:
                keyword_norm = normalize_for_matching(keyword)
                if keyword_norm and keyword_norm in normalized:
                    matched_keywords[discourse_name].append(keyword)
                    family_hit = True

            if family_hit:
                matched_families[discourse_name].append(family_name)

    for discourse_name in ("발전", "경계"):
        matched_keywords[discourse_name] = list(dict.fromkeys(matched_keywords[discourse_name]))
        matched_families[discourse_name] = list(dict.fromkeys(matched_families[discourse_name]))

    development_hit = bool(matched_keywords["발전"])
    boundary_hit = bool(matched_keywords["경계"])

    return {
        "발전담론": "예" if development_hit else "아니오",
        "발전계열": "; ".join(matched_families["발전"]),
        "발전키워드": "; ".join(matched_keywords["발전"]),
        "발전키워드수": str(len(matched_keywords["발전"])),
        "경계담론": "예" if boundary_hit else "아니오",
        "경계계열": "; ".join(matched_families["경계"]),
        "경계키워드": "; ".join(matched_keywords["경계"]),
        "경계키워드수": str(len(matched_keywords["경계"])),
        "담론분류": (
            "발전+경계" if development_hit and boundary_hit
            else "발전" if development_hit
            else "경계" if boundary_hit
            else "미분류"
        ),
    }


def augment_csv(input_csv: str = INPUT_CSV) -> tuple[int, str]:
    """
    기존 이승만 시기 CSV를 직접 갱신한다.
    갱신 전 원본을 날짜가 붙은 .bak 파일로 보존한다.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = input_path.with_name(f"{input_path.name}.{timestamp}.bak")
    shutil.copy2(input_path, backup_path)

    fd, temp_path = tempfile.mkstemp(
        prefix=f".{input_path.stem}_",
        suffix=".tmp.csv",
        dir=str(input_path.parent),
    )
    os.close(fd)

    processed = 0

    try:
        with open(input_path, "r", encoding="utf-8-sig", newline="") as src, \
             open(temp_path, "w", encoding="utf-8-sig", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

            header = next(reader, None)
            if not header:
                raise ValueError("CSV 헤더를 읽지 못했습니다.")

            if "Content" not in header:
                raise ValueError("Content 열을 찾지 못했습니다.")

            content_index = header.index("Content")
            existing = set(header)
            new_columns = [c for c in ADDED_COLUMNS if c not in existing]

            # 이미 분석 열이 일부 존재하더라도 중복 열을 만들지 않는다.
            writer.writerow(header + new_columns)

            for row in reader:
                if not row:
                    continue

                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))

                analysis = analyze_content(row[content_index])
                writer.writerow(row + [analysis[col] for col in new_columns])
                processed += 1

        os.replace(temp_path, input_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return processed, str(backup_path)


def main():
    print("=" * 60)
    print("🚀 이승만 시기 회의록 발전·경계 담론 분석 시작")
    print(f"📄 대상: {INPUT_CSV}")
    print("=" * 60)

    processed, backup_path = augment_csv(INPUT_CSV)

    print(f"✅ 분석 완료: {processed:,}개 행")
    print(f"🛡️ 원본 백업: {backup_path}")
    print("📌 추가 열:")
    for column in ADDED_COLUMNS:
        print(f"   - {column}")


if __name__ == "__main__":
    main()
