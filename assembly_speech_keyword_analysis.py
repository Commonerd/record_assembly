import polars as pl
import glob

# ==============================================================================
# 1. Load and prepare legislator metadata (제헌 ~ 4대 국회 통합)
# ==============================================================================
# 각 대수별 원본 파일은 컬럼명이 서로 다르므로(한글 컬럼명이 대수마다 다름)
# 하나의 공통 영어 스키마로 통일한 뒤 세로로 합친다.

def find_file(prefix: str) -> str:
    matches = glob.glob(f"input/{prefix}*.csv")
    if not matches:
        raise FileNotFoundError(f"No file starting with '{prefix}' found in current directory")
    return matches[0]


META_FILES = [
    (find_file("제헌"), "제헌"),
    (find_file("2대"), "제2대"),
    (find_file("3대"), "제3대"),
    (find_file("4대"), "제4대"),
]

# 원본 CSV의 한글 컬럼명 -> 영어 컬럼명 매핑.
# 제헌국회 파일(Assembly_Num/Speaker/Party)과 2~4대 파일(대수/의원명/정당)이
# 서로 다른 표기를 쓰므로 둘 다 흡수할 수 있게 매핑을 넉넉히 잡는다.
RENAME_MAP = {
    "번호": "No",
    "대수": "Assembly_Num",
    "Assembly_Num": "Assembly_Num",
    "의원명": "Speaker",
    "Speaker": "Speaker",
    "정당": "Party",
    "Party": "Party",
    "소속위원회": "Committee",
    "지역": "Region",
    "성별": "Gender",
    "당선횟수": "Election_Count",
    "당선방법": "Election_Method",
}

meta_frames = []
for path, assembly_label in META_FILES:
    df = pl.read_csv(path)
    df = df.rename({k: v for k, v in RENAME_MAP.items() if k in df.columns})
    # Assembly_Num 표기를 "제헌", "제2대", "제3대", "제4대"로 통일
    df = df.with_columns(pl.lit(assembly_label).alias("Assembly_Num"))
    meta_frames.append(
        df.select([
            "Assembly_Num", "Speaker", "Party", "Committee", "Region",
            "Gender", "Election_Count", "Election_Method",
        ])
    )

meta_df = pl.concat(meta_frames, how="vertical_relaxed")

# 의원이 여러 대수에 걸쳐 당적을 바꾼 경우가 있으므로, 동일 (Speaker, Assembly_Num)
# 조합이 중복되면 첫 번째 값만 사용한다 (join 시 row 폭발 방지).
meta_df = meta_df.unique(subset=["Assembly_Num", "Speaker"], keep="first")

# ==============================================================================
# 2. 회의록 파일 스캔 및 발언자 단위 분할 함수
# ==============================================================================

# Title 컬럼 예시: "제1대국회 제1회(임시회) 제1차 국회본회의(전체회의) (1948.05.31.)"
# -> 맨 앞의 "제N대국회"에서 대수를 뽑아 meta_df 표기(제헌/제2대/제3대/제4대)로 변환
ASSEMBLY_NUM_KOREAN = {1: "제헌", 2: "제2대", 3: "제3대", 4: "제4대"}


def title_to_assembly_num(title_expr: pl.Expr) -> pl.Expr:
    num_str = title_expr.str.extract(r"^제(\d+)대국회", 1)
    return (
        num_str.cast(pl.Int64, strict=False)
        .replace_strict(ASSEMBLY_NUM_KOREAN, default=None)
        .alias("Assembly_Num")
    )


# 직책/호칭 패턴 정의
# 주의: "국회선거위원회사무총장"처럼 [단체명]+위원회+사무총장 형태의 복합 직책도
# 있으므로, "위원회"를 포함하는 조합과 "총장"류 접미사도 함께 인식하도록 넓게 잡는다.
titles_pattern = (
    r"(?:[가-힣]*위원회[가-힣]*(?:위원장|사무총장|부위원장)"
    r"|[가-힣]*위원장"
    r"|[가-힣]*장관"
    r"|[가-힣]*부장"
    r"|[가-힣]*국장"
    r"|임시의장|부의장|의장"
    r"|사무총장|국무총리|정부위원|위원|의원|증인|참고인)"
)


def parse_assembly_speeches(file_path: str) -> pl.LazyFrame:
    lazy_df = pl.scan_csv(file_path)

    parsed_lazy = (
        lazy_df
        .with_columns([
            title_to_assembly_num(pl.col("Title")),
            pl.col("Content").str.split("◯").alias("Speech_Chunk"),
        ])
        .explode("Speech_Chunk")
        .with_columns(pl.col("Speech_Chunk").str.strip_chars().alias("Speech_Chunk"))
        .filter(pl.col("Speech_Chunk") != "")
    )

    parsed_lazy = parsed_lazy.with_columns([
        # 1. 직책(Title_Role) 추출
        pl.coalesce([
            # Pattern A: [직책] + [이름] (예: "사회보건위원장 이진수" -> 직책: "사회보건위원장")
            pl.col("Speech_Chunk").str.extract(r"^(" + titles_pattern + r")\s+[가-힣]{2,4}", 1),

            # Pattern B: [이름] + [직책] (예: "김철수 의원" -> 직책: "의원")
            pl.col("Speech_Chunk").str.extract(r"^[가-힣]{2,4}\s*(" + titles_pattern + r")", 1),

            # Pattern C: 이름 없이 직책만 나온 경우 (예: "증인", "사회보건위원장")
            pl.col("Speech_Chunk").str.extract(r"^(" + titles_pattern + r")", 1),
        ]).fill_null(pl.lit("")).alias("Title_Role"),

        # 2. 이름(Speaker) 추출
        pl.coalesce([
            # Pattern A: [직책] + [이름] -> 이름만 추출
            pl.col("Speech_Chunk").str.extract(r"^" + titles_pattern + r"\s+([가-힣]{2,4})", 1),

            # Pattern B: [이름] + [직책] -> 이름만 추출
            pl.col("Speech_Chunk").str.extract(r"^([가-힣]{2,4})\s*" + titles_pattern, 1),

            # Pattern C: 단독 이름 (예: "신익희 ...")
            pl.col("Speech_Chunk").str.extract(r"^([가-힣]{2,4})(?:\s+|$)", 1),
        ]).fill_null(pl.lit("")).alias("Speaker"),

        # 3. 본문(Speech) 추출 (서두의 직책/이름 헤더 제거)
        pl.col("Speech_Chunk")
          .str.replace(
              r"^(?:" + titles_pattern + r"\s+[가-힣]{2,4}|[가-힣]{2,4}\s*" + titles_pattern + r"|" + titles_pattern + r"|[가-힣]{2,4})\s*",
              "",
          )
          .alias("Speech"),
    ]).drop("Speech_Chunk")

    return parsed_lazy


# ==============================================================================
# 3. 데이터 실행 및 정당 메타데이터 결합 (Join)
# ==============================================================================
speech_lazy = parse_assembly_speeches("record_assembly_sorted_rhee_keywords.csv")

# Left Join 실행 및 매칭되지 않은 Party는 "미상/직책자"로 채움
final_lazy = speech_lazy.join(
    meta_df.lazy(),
    on=["Speaker", "Assembly_Num"],
    how="left",
).with_columns(
    pl.col("Party").fill_null(pl.lit("미상/직책자")),
)

# ==============================================================================
# 4. 키워드 빈도 집계 (정당별 & 의원/직책별)
# ==============================================================================
# 키워드 컬럼명을 영어로 통일: 기술/소득/노동/이민/이주/외국인
KEYWORD_MAP = {
    "기술": "Technology",
    "소득": "Income",
    "노동": "Labor",
    "이민": "Immigration",
    "이주": "Migration",
    "외국인": "Foreigner",
}
keywords_kr = list(KEYWORD_MAP.keys())
keywords_en = list(KEYWORD_MAP.values())

kw_exprs = [
    pl.col("Speech").str.count_matches(kr).alias(en)
    for kr, en in KEYWORD_MAP.items()
]

# LazyFrame에 키워드 표현식 연산 추가
final_with_kw = final_lazy.with_columns(kw_exprs).with_columns(
    pl.sum_horizontal(keywords_en).alias("Keyword_Total")
)

final_collected = final_with_kw.collect()

# ==============================================================================
# 5. 결과물 1: 의원별 발언 정리 (발언 단위 상세 데이터, 시간 흐름순 정렬)
# ==============================================================================
# Date 컬럼이 "1948.05.31." 같은 문자열이라 그대로 정렬하면 자릿수 차이(예:
# "1948.5.9." vs "1948.10.1.")로 실제 시간 순서와 어긋날 수 있다. 따라서 실제
# Date 타입으로 변환한 뒤, 같은 날짜에 여러 차수 회의가 있을 경우를 대비해
# Title(회차 정보 포함)을 보조 정렬 기준으로 사용해 정확한 시간 흐름순을 만든다.
# 최종 Date 표기는 컴퓨터가 읽기 쉬운 YYYY-MM-DD(ISO 8601) 형식으로 저장한다.
speaker_speech_detail = (
    final_collected
    .with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y.%m.%d.", strict=False).alias("Date_Parsed")
    )
    .with_columns(
        pl.col("Date_Parsed").dt.strftime("%Y-%m-%d").alias("Date")
    )
    .select([
        "Assembly_Num", "Title", "Date", "Date_Parsed", "Speaker", "Title_Role", "Party",
        "Committee", "Region", "Speech", *keywords_en, "Keyword_Total",
    ])
    .sort(["Date_Parsed", "Title"])
    .drop("Date_Parsed")
)

speaker_speech_detail.write_csv("output/speaker_speech_detail.csv")

# ==============================================================================
# 6. 결과물 2: 의원/직책별 키워드 집계
# ==============================================================================
speaker_keyword_summary = (
    final_collected
    .group_by(["Assembly_Num", "Speaker", "Title_Role", "Party"])
    .agg([pl.col(en).sum() for en in keywords_en] + [pl.len().alias("Speech_Count")])
    .sort(by="Labor", descending=True)
)

speaker_keyword_summary.write_csv("output/speaker_keyword_summary.csv")

# ==============================================================================
# 7. 결과물 3: 정당별 키워드 집계
# ==============================================================================
party_keyword_summary = (
    final_collected
    .group_by(["Assembly_Num", "Party"])
    .agg([pl.col(en).sum() for en in keywords_en] + [pl.len().alias("Speech_Count")])
    .sort(by="Labor", descending=True)
)

party_keyword_summary.write_csv("output/party_keyword_summary.csv")

# ==============================================================================
# 8. 결과 콘솔 출력
# ==============================================================================
print("=== 의원별 발언 정리 (상위 5건) ===")
print(speaker_speech_detail.head(5))

print("\n=== 정당별 키워드 집계 ===")
print(party_keyword_summary)

print("\n=== 의원/직책별 키워드 집계 (상위 10건) ===")
print(speaker_keyword_summary.head(10))
