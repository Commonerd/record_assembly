"""
국회 회의록 발언자-당적 매핑 및 키워드 분석 파이프라인
======================================================

[측정 기준 - 팀 논의 전까지 아래 기본값으로 계산됨. 변경 시 CONFIG 섹션만 수정]

1. 원자료(raw count)와 정규화 지표를 모두 산출하여, 연구질문에 따라 나중에
   선택할 수 있도록 함 (raw count만으로 결론 내리지 않음).
   - raw_count       : 발화(◯ 단위) 전체에서 키워드가 등장한 총 횟수
   - 회기수(session presence) : 해당 화자/정당이 그 키워드를 1회 이상 언급한
     "회의록(세션)"의 개수 (한 회기에 10번 말해도 1로 카운트)
   - 정규화 빈도(rate_per_10k) : raw_count / 총 발화 어절수 * 10,000
     (발화량이 다른 화자·정당 간 비교 시 이 지표를 우선 사용 권장)

2. 급증(surge) 판정: 직전 SURGE_WINDOW개 세션의 이동평균 + SURGE_K * 이동표준편차를
   초과하는 세션을 급증으로 플래그. 최소 MIN_SURGE_HISTORY개 이전 세션 데이터가
   있어야 계산 (부족하면 null). look-ahead bias 방지를 위해 "직전" 세션들만 사용.

3. 동명이인(같은 대수에 이름이 같은 서로 다른 의원)은 임의로 하나를 고르지 않고
   "동명이인(확인필요): A / B" 형태로 모든 값을 남겨, 분석 시 사람이 직접 확인하게 함.

※ 초당적 그룹 탐지(co-occurrence network / community detection)는 이번 버전에는
   포함하지 않음. 회기·발언자 단위 co-occurrence 테이블을 먼저 쌓은 뒤 별도로
   networkx 등을 붙이는 방식을 제안함 (다음 단계).
"""

import polars as pl
import re

# ==============================================================================
# 0. CONFIG
# ==============================================================================
META_FILES = {
    "제헌": "input/제헌국회 의원-당적.csv",
    "제2대": "input/2대국회 의원-당적.csv",
    "제3대": "input/3대국회 의원-당적.csv",
    "제4대": "input/4대국회 의원-당적.csv",
}
RECORD_FILE = "record_assembly_sorted_rhee_keywords.csv"

KEYWORDS = ["기술", "소득", "노동", "이민", "이주", "외국인"]

RATE_PER_WORDS = 10_000     # 정규화 빈도 분모 기준 (어절 1만 개당)
SURGE_WINDOW = 5            # 급증 판정용 이동통계 윈도우(직전 세션 수)
SURGE_K = 2.0               # 이동평균 + K * 이동표준편차 초과 시 급증
MIN_SURGE_HISTORY = 3       # 이동통계 계산에 필요한 최소 직전 세션 수

TITLES_PATTERN = (
    r"(?:[가-힣]*위원장|[가-힣]*장관|[가-힣]*부장|[가-힣]*국장|"
    r"임시의장|부의장|의장|사무총장|국무총리|정부위원|위원|의원|증인|참고인)"
)

META_EXTRA_COLS = ["Party", "지역", "소속위원회", "당선횟수", "성별", "당선방법"]
STD_COLS = ["Assembly_Num", "Speaker"] + META_EXTRA_COLS


# ==============================================================================
# 1. 의원 메타데이터 로드 및 표준화 (제헌 / 2대 / 3대 / 4대 통합)
#    - 파일마다 컬럼명이 다름(제헌: 영문 / 2~4대: 한글) -> 표준 컬럼명으로 통일
#    - Assembly_Num 표기는 함수 호출 시 지정한 값으로 강제 통일
# ==============================================================================
def load_meta(path: str, assembly_num: str) -> pl.DataFrame:
    df = pl.read_csv(path)

    rename_map = {}
    if "대수" in df.columns:
        rename_map["대수"] = "Assembly_Num"
    if "의원명" in df.columns:
        rename_map["의원명"] = "Speaker"
    if "정당" in df.columns:
        rename_map["정당"] = "Party"
    if rename_map:
        df = df.rename(rename_map)

    df = df.with_columns(pl.lit(assembly_num).alias("Assembly_Num"))
    return df.select(STD_COLS)


meta_df = pl.concat([load_meta(path, num) for num, path in META_FILES.items()])

# 동명이인(같은 대수 + 같은 이름, 서로 다른 인물) 처리
# -> 값이 여러 개면 " / "로 합쳐 "동명이인(확인필요)"로 표시, 값이 하나면 그대로 사용
dup_agg = meta_df.group_by(["Assembly_Num", "Speaker"]).agg(
    [pl.col(c).unique().alias(f"{c}__list") for c in META_EXTRA_COLS]
).with_columns(pl.col("Party__list").list.len().alias("n_dup"))

ambiguous_names = dup_agg.filter(pl.col("n_dup") > 1)
if ambiguous_names.height > 0:
    print("[경고] 동명이인으로 추정되는 의원 (대수 내 이름 중복, 정당 상이):")
    print(ambiguous_names.select(["Assembly_Num", "Speaker", "Party__list"]))

meta_lookup = dup_agg.with_columns([
    pl.when(pl.col("n_dup") > 1)
    .then(pl.lit("동명이인(확인필요): ") + pl.col(f"{c}__list").list.join(" / "))
    .otherwise(pl.col(f"{c}__list").list.first())
    .alias(c)
    for c in META_EXTRA_COLS
]).select(["Assembly_Num", "Speaker"] + META_EXTRA_COLS)

# 대수별 정당 소속 의원 수 (실제 발언 여부와 무관 - 정당 규모 정규화용)
party_member_count = (
    meta_df.group_by(["Assembly_Num", "Party"])
    .agg(pl.col("Speaker").n_unique().alias("정당_소속의원수"))
)


# ==============================================================================
# 2. 회의록 발언자 단위 분할
#    - Session_ID: 원본 회의록(행) 단위 고유 ID 부여 (세션 단위 집계/타임라인용)
#    - Title(회의록 제목)에서 "제N대국회" 패턴으로 대수 자동 판별
#    - 회의록 제목(Session_Title)과 발언자 직책(직책) 컬럼명이 겹치지 않도록 분리
# ==============================================================================
def parse_assembly_speeches(file_path: str) -> pl.LazyFrame:
    lazy_df = (
        pl.scan_csv(file_path)
        .rename({"Title": "Session_Title"})
        .with_row_index("Session_ID")
    )

    parsed_lazy = (
        lazy_df
        .with_columns(pl.col("Session_Title").str.extract(r"제(\d+)대국회", 1).alias("_num"))
        .with_columns(
            pl.when(pl.col("_num") == "1").then(pl.lit("제헌"))
            .when(pl.col("_num").is_not_null()).then(pl.lit("제") + pl.col("_num") + pl.lit("대"))
            .otherwise(pl.lit("")).alias("Assembly_Num")
        )
        .drop("_num")
        .with_columns(pl.col("Content").str.split("◯").alias("Speech_Chunk"))
        .explode("Speech_Chunk")
        .with_columns(pl.col("Speech_Chunk").str.strip_chars().alias("Speech_Chunk"))
        .filter(pl.col("Speech_Chunk") != "")
    )

    parsed_lazy = parsed_lazy.with_columns([
        # 1. 직책 추출
        pl.coalesce([
            # Pattern A: [직책] + [이름] (예: "사회보건위원장 이진수" -> 직책: "사회보건위원장")
            pl.col("Speech_Chunk").str.extract(r"^(" + TITLES_PATTERN + r")\s+[가-힣]{2,4}", 1),
            # Pattern B: [이름] + [직책] (예: "김철수 의원" -> 직책: "의원")
            # 이름-직책 사이 공백 필수(\s+). \s*로 두면 "국회선거위원회사무총장"처럼
            # 조직명 안의 "위원/의원" 문자열이 이름+직책으로 오탐되는 버그가 있었음.
            pl.col("Speech_Chunk").str.extract(r"^[가-힣]{2,4}\s+(" + TITLES_PATTERN + r")", 1),
            # Pattern C: 이름 없이 직책만 나온 경우 (예: "증인", "위원장")
            pl.col("Speech_Chunk").str.extract(r"^(" + TITLES_PATTERN + r")", 1),
        ]).fill_null(pl.lit("")).alias("직책"),

        # 2. 이름(Speaker) 추출
        pl.coalesce([
            pl.col("Speech_Chunk").str.extract(r"^" + TITLES_PATTERN + r"\s+([가-힣]{2,4})", 1),
            pl.col("Speech_Chunk").str.extract(r"^([가-힣]{2,4})\s+" + TITLES_PATTERN, 1),
            pl.col("Speech_Chunk").str.extract(r"^([가-힣]{2,4})(?:\s+|$)", 1),
        ]).fill_null(pl.lit("")).alias("Speaker"),

        # 3. 본문(Speech) 추출 (서두의 직책/이름 헤더 제거)
        pl.col("Speech_Chunk")
        .str.replace(
            r"^(?:" + TITLES_PATTERN + r"\s+[가-힣]{2,4}|[가-힣]{2,4}\s+" + TITLES_PATTERN + r"|"
            + TITLES_PATTERN + r"|[가-힣]{2,4})\s*",
            ""
        )
        .alias("Speech")
    ]).drop("Speech_Chunk")

    return parsed_lazy


# ==============================================================================
# 3. 데이터 실행 + 정당/인적사항 메타 결합 + 키워드/어절수 계산
# ==============================================================================
speech_lazy = parse_assembly_speeches(RECORD_FILE)

final_lazy = (
    speech_lazy
    .join(meta_lookup.lazy(), on=["Speaker", "Assembly_Num"], how="left")
    .with_columns([pl.col(c).fill_null(pl.lit("")) for c in META_EXTRA_COLS])
    .with_columns([pl.col("Speech").str.count_matches(kw).alias(kw) for kw in KEYWORDS])
    .with_columns(
        pl.col("Speech").str.split(" ").list.len().alias("어절수")
    )
)

final_df = final_lazy.collect()

# ==============================================================================
# 4-1. speaker_speech_detail.csv : 발언자-당적-발언내용 (매핑된 원자료)
# ==============================================================================
speech_detail = final_df.select(
    ["Assembly_Num", "Session_ID", "Date", "Session_Title", "Speaker", "직책",
     "Party", "지역", "소속위원회", "당선횟수", "Speech", "어절수"] + KEYWORDS
).sort(["Assembly_Num", "Date", "Session_ID"])

speech_detail.write_csv("speaker_speech_detail.csv")

# ==============================================================================
# 4-2. speaker_keyword_summary.csv : 발언자-키워드 빈도
#      raw_count + 회기수(session presence) + 정규화 빈도(rate_per_10k)
# ==============================================================================
# (a) 세션 단위로 먼저 합산 -> 회기수(presence) 계산의 기준 단위
speaker_session_kw = (
    final_df.group_by(["Assembly_Num", "Speaker", "Session_ID"])
    .agg([pl.col(kw).sum().alias(kw) for kw in KEYWORDS])
)
speaker_session_presence = (
    speaker_session_kw
    .with_columns([(pl.col(kw) > 0).cast(pl.Int32).alias(f"{kw}_회기수") for kw in KEYWORDS])
    .group_by(["Assembly_Num", "Speaker"])
    .agg([pl.col(f"{kw}_회기수").sum() for kw in KEYWORDS])
)

# (b) 발언자별 raw 합계 + 어절수 총합 + 발언횟수
speaker_raw = (
    final_df.group_by(["Assembly_Num", "Speaker", "직책", "Party", "지역", "소속위원회", "당선횟수"])
    .agg(
        [pl.col(kw).sum().alias(kw) for kw in KEYWORDS]
        + [pl.len().alias("발언횟수"), pl.col("어절수").sum().alias("총어절수")]
    )
)

speaker_keyword_summary = speaker_raw.join(
    speaker_session_presence, on=["Assembly_Num", "Speaker"], how="left"
)

# (c) 정규화 빈도 컬럼 추가 (어절 1만 개당)
speaker_keyword_summary = speaker_keyword_summary.with_columns([
    pl.when(pl.col("총어절수") > 0)
    .then(pl.col(kw) / pl.col("총어절수") * RATE_PER_WORDS)
    .otherwise(0.0)
    .round(2)
    .alias(f"{kw}_빈도_1만어절당")
    for kw in KEYWORDS
]).sort(by="노동", descending=True)

speaker_keyword_summary.write_csv("speaker_keyword_summary.csv")

# ==============================================================================
# 4-3. party_keyword_summary.csv : 정당-키워드 빈도
#      raw_sum + 정당 소속의원수 대비 1인당 평균 + 정규화 빈도(rate_per_10k)
# ==============================================================================
party_raw = (
    final_df.group_by(["Assembly_Num", "Party"])
    .agg(
        [pl.col(kw).sum().alias(kw) for kw in KEYWORDS]
        + [pl.len().alias("발언횟수"), pl.col("어절수").sum().alias("총어절수")]
    )
)

party_keyword_summary = party_raw.join(
    party_member_count, on=["Assembly_Num", "Party"], how="left"
).with_columns(pl.col("정당_소속의원수").fill_null(0))

party_keyword_summary = party_keyword_summary.with_columns(
    [
        pl.when(pl.col("총어절수") > 0)
        .then(pl.col(kw) / pl.col("총어절수") * RATE_PER_WORDS)
        .otherwise(0.0)
        .round(2)
        .alias(f"{kw}_빈도_1만어절당")
        for kw in KEYWORDS
    ] + [
        pl.when(pl.col("정당_소속의원수") > 0)
        .then(pl.col(kw) / pl.col("정당_소속의원수"))
        .otherwise(0.0)
        .round(2)
        .alias(f"{kw}_1인당평균")
        for kw in KEYWORDS
    ]
).sort(by="노동", descending=True)

party_keyword_summary.write_csv("party_keyword_summary.csv")

# ==============================================================================
# 4-4. session_keyword_timeline.csv : 서사(narrative) 패턴 지원용
#      회기(세션) 단위 키워드 총량 + 이동평균/이동표준편차 + 급증(surge) 플래그
# ==============================================================================
session_timeline = (
    final_df.group_by(["Assembly_Num", "Session_ID", "Date", "Session_Title"])
    .agg([pl.col(kw).sum().alias(kw) for kw in KEYWORDS])
    .sort(["Assembly_Num", "Session_ID"])
)

for kw in KEYWORDS:
    session_timeline = session_timeline.with_columns([
        pl.col(kw).shift(1).over("Assembly_Num")
        .rolling_mean(window_size=SURGE_WINDOW, min_samples=MIN_SURGE_HISTORY)
        .alias(f"{kw}_이동평균"),
        pl.col(kw).shift(1).over("Assembly_Num")
        .rolling_std(window_size=SURGE_WINDOW, min_samples=MIN_SURGE_HISTORY)
        .alias(f"{kw}_이동표준편차"),
    ])
    session_timeline = session_timeline.with_columns(
        (
            pl.col(f"{kw}_이동평균").is_not_null()
            & (pl.col(kw) > (pl.col(f"{kw}_이동평균") + SURGE_K * pl.col(f"{kw}_이동표준편차").fill_null(0.0)))
        ).alias(f"{kw}_급증")
    )

session_timeline.write_csv("session_keyword_timeline.csv")

# ==============================================================================
# 5. 콘솔 요약 출력
# ==============================================================================
print("\n=== speaker_speech_detail.csv (상위 5행) ===")
print(speech_detail.head(5))

print("\n=== speaker_keyword_summary.csv (상위 5행, 노동 기준 정렬) ===")
print(speaker_keyword_summary.head(5))

print("\n=== party_keyword_summary.csv ===")
print(party_keyword_summary)

print("\n=== session_keyword_timeline.csv ===")
print(session_timeline)
