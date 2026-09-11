"""
국회 회의록 발언자 파싱 · 키워드 집계 (v2)
==============================================================================
v1(assembly_modifired.py) 대비 변경점 요약
------------------------------------------------------------------------------
unmatched_speaker_review.csv 5,229건을 역추적한 결과, 매칭 실패 원인은
'애매한 데이터' 때문이 아니라 규칙 3개의 구조적 결함 때문이었다.

  [원인 1] 복합 직책 미등록 ................................ 3,492건 (66.8%)
      "위원장대리 송방용", "국무총리서리 허정", "사무총장대리 정홍섭"에서
      titles_pattern이 "위원장"까지만 먹고 "대리"가 남아
      `직책\\s+이름` 이 깨졌다.  -> TITLE_SUFFIX 로 흡수.

  [원인 2] 직책과 이름 사이 공백 없음 ........................ 264건 (5.0%)
      "의장이승만", "부의장신익희", "사무총장전규홍".
      제헌국회 회의록은 원래 붙여쓰기 조판이라 공백이 없는 게 정상이다.
      v1은 `\\s+`(1개 이상)를 요구해서 전부 실패.  -> `\\s*`(0개 이상) 허용.
      단, 공백 0개를 허용하면 이름 경계를 알 수 없으므로 반드시
      '이름 사전(gazetteer)' 과 함께 써야 한다. (아래 참조)

  [원인 3] dehyphenate_header 가 정상 데이터를 파괴 ........ 1,473건 (28.2%)
      v1의 `([가-힣]{1,2})\\s+([가-힣]{1,2})` 규칙은 "2글자 이름 + 공백 +
      본문 첫 단어"를 이름 하나로 붙여버린다.
          "운영위원장 조순 그러면…"    -> Speaker="조순그러"
          "법무부장관 이인 국적법…"    -> Speaker="이인국적"
          "교통부장관 허정 인사의…"    -> Speaker="허정인사"
      실제로 이 1,473건은 전부 2글자 이름(조순·이인·이호·허정·장면·김훈·
      정준) 발언자였고, 99.7%가 조인 실패로 이어졌다.
      즉 "이스 니" 같은 희귀 케이스를 잡으려다 2글자 이름 발언자를
      통째로 잃은 것. -> 이 휴리스틱은 폐기하고, 사전 기반으로 대체한다.

핵심 설계 변경: 규칙 추측 -> 이름 사전(gazetteer) 매칭
------------------------------------------------------------------------------
"직책 뒤 2~4글자를 이름으로 본다"는 규칙은 이름 경계를 원리적으로 알 수 없다.
대신 이미 갖고 있는 정답지(제헌~4대 국회의원 명부)를 사전으로 쓰고,
사전에 있는 이름만 매칭한다.

  - 사전 이름은 글자 사이 `\\s*`를 허용해 생성한다.
    -> "이 승만", "이스 니" 처럼 공백이 낀 이름도 잡히고,
       사전에 없는 임의 문자열은 절대 이름으로 오인되지 않는다.
       (회의에서 지적된 '단어 중간 공백' 문제를 오탐 없이 해결)
  - 국무위원·정부위원 등 명부에 없는 인사는 2-pass로 데이터에서 학습한다.
    (Pass 1에서 "직책은 잡혔는데 이름을 못 찾은" 잔여 문자열을 모아,
     직책별로 반복 출현하는 접두사를 이름으로 승격. 3번째 글자의 분산도로
     2글자 이름/3글자 이름을 판별한다. 예: 이인+{요,시,법,지,국} -> "이인",
     송방+{용,용,용} -> "송방용")
  - 사전에도 없고 학습도 안 된 경우에만 기존 방식(직책 뒤 2~4글자)으로
    보수적으로 추정하고 Speaker_Source="fallback" 으로 표시해 검수 대상에 남긴다.

출력 컬럼 변경
------------------------------------------------------------------------------
  Speaker_Match_Flag / Speaker_Dehyphen_Flag  ->  Speaker_Source 로 통합
      "member"   : 국회의원 명부에서 확정 (신뢰도 최상)
      "learned"  : 데이터에서 학습한 직책자 이름 (신뢰도 상)
      "fallback" : 규칙 추정 (검수 필요)
      "none"     : 이름 파싱 실패 (검수 필요 / 발언이 아닌 행일 가능성)
  Speaker_Type : 의원 / 정부·기타 직책자 / 미상

실행:  python assembly_v2.py
자가검증: SELFTEST=1 python assembly_v2.py   (데이터 없이 파서 규칙만 테스트)
"""

import glob
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl

INPUT_DIR = "input"
OUTPUT_DIR = "output"
SPEECH_FILE = "record_assembly_sorted_rhee_keywords.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 0. 직책 / 이름 / 헤더 정규식 구성 요소
# ==============================================================================
# 주의: polars는 Rust regex 엔진이라 look-around(?=, ?<=)를 못 쓴다.
#       모든 패턴은 look-around 없이, 소비(consume) 방식으로만 작성한다.
#       또 alternation은 leftmost-first 이므로 "긴 것 먼저" 순서가 중요하다.

TITLE_CORE = (
    r"(?:[가-힣]{0,10}위원회[가-힣]{0,6}(?:위원장|부위원장|사무총장|간사|위원)"
    r"|[가-힣]{0,10}위원장"          # 운영위원장, 법제사법위원장 …
    r"|[가-힣]{0,8}장관"             # 법무부장관, 교통부장관 …  (부장보다 먼저!)
    r"|[가-힣]{0,8}차관"
    r"|[가-힣]{0,10}부장"
    r"|[가-힣]{0,10}국장"            # 육군본부인사국장 …
    r"|[가-힣]{0,10}청장"
    r"|[가-힣]{0,10}실장"
    r"|[가-힣]{0,10}총장"            # 사무총장 …
    r"|임시의장|부의장|의장"
    r"|국무총리|국무위원|정부위원|전문위원"
    r"|위원|의원|증인|참고인|서기)"
)

# [원인 1] 해결: "위원장대리", "국무총리서리", "사무총장대리" 등 복합 직책.
TITLE_SUFFIX = r"(?:직무대리|권한대행|직무대행|대리|서리|대행)?"
TITLE_PAT = "(?:" + TITLE_CORE + TITLE_SUFFIX + ")"

# [원인 2] 해결: 구분자 0개 허용(`*`). 대신 이름은 사전 매칭으로만 인정한다.
SEP = r"[\s,.·․‧・:：]*"
# 직책/이름 뒤에 한자나 소속이 괄호로 붙는 경우: "김철수(金哲洙)", "위원장(자유당)"
PAREN = r"(?:\([^)]{0,30}\))?"
# 헤더의 끝 경계. 공백/구두점을 소비하거나 문자열 끝.
BOUND = r"(?:[\s,.·、，]+|$)"
# fallback 전용 구분자: 공백이 반드시 1개 이상 있어야 함(본문 침범 방지)
SEP_STRICT = r"[\s,·]+"
GENERIC_NAME = r"[가-힣]{2,4}"


def name_to_regex(name: str) -> str:
    r"""이름 글자 사이에 선택적 공백을 허용. "이승만" -> "이\s*승\s*만"

    회의에서 지적된 "이스 니"(이름 중간에 공백이 낀 경우)를 잡기 위한 것.
    v1처럼 임의의 짧은 조각을 붙이는 게 아니라 '아는 이름'에만 적용하므로
    본문을 이름으로 오인할 위험이 없다.
    """
    return r"\s*".join(re.escape(ch) for ch in name)


def build_name_alternation(names) -> str:
    """이름 사전을 하나의 alternation 정규식으로. 긴 이름 우선(leftmost-first)."""
    ns = sorted({n for n in names if n and len(n) >= 2}, key=lambda s: (-len(s), s))
    if not ns:
        # 사전이 비어 있으면 절대 매칭되지 않는 패턴을 반환 (파이프라인 보호)
        return r"(?!)" if False else r"(?:\x00NEVER\x00)"
    return "(?:" + "|".join(name_to_regex(n) for n in ns) + ")"


def build_header_patterns(name_alt: str):
    """헤더 인식용 정규식 세트를 만든다.

    반환: dict(v1_title, v1_name, v2_name, v2_title, v3_title, v4_name, removal)
    우선순위:
      V1  [직책] [사전이름]      예) "위원장대리 송방용", "의장이승만"
      V2  [사전이름] [직책]      예) "김철수 의원"
      V3  [직책]만               예) "위원장대리" (이름은 2단계에서 재시도)
      V4  [사전이름]만           예) "조순 그러면…"
    """
    T, N = TITLE_PAT, name_alt
    pats = {
        "v1": r"^(" + T + r")" + PAREN + SEP + r"(" + N + r")" + PAREN + BOUND,
        "v2": r"^(" + N + r")" + PAREN + SEP + r"(" + T + r")" + PAREN + BOUND,
        "v3": r"^(" + T + r")" + PAREN + BOUND,
        "v4": r"^(" + N + r")" + PAREN + BOUND,
    }
    pats["removal"] = "^(?:" + "|".join([
        T + PAREN + SEP + N + PAREN + BOUND,
        N + PAREN + SEP + T + PAREN + BOUND,
        T + PAREN + BOUND,
        N + PAREN + BOUND,
    ]) + ")"
    return pats


# 2단계(fallback): 사전에 없는 이름을 보수적으로 추정.
#   반드시 [직책] + 공백 + [2~4글자] 형태여야 하며, 아래 불용어는 제외한다.
FALLBACK_NAME_RE = r"^(" + GENERIC_NAME + r")" + PAREN + r"(?:[\s,.·]+|$)"

# 직책 뒤에 자주 오는 '이름이 아닌' 첫 단어들. fallback 오탐 차단용.
NAME_STOPWORDS = {
    "그러", "그러면", "그런데", "그리고", "그것", "그거", "지금", "이것", "이거",
    "저는", "우리", "오늘", "어제", "내일", "다음", "이번", "지난", "여러",
    "회의", "의사", "국회", "본회", "의장", "위원", "의원", "정부", "대한",
    "명단", "출석", "결석", "전문", "조사", "감사", "보고", "동의", "재청",
    "개회", "폐회", "산회", "속개", "표결", "가결", "부결", "제안", "설명",
    "질의", "답변", "발언", "말씀", "지금부", "잠깐", "먼저", "다시", "아까",
    "여기", "거기", "저기", "이상", "이하", "제일", "제이", "제삼", "무슨",
    "어떤", "어떻", "왜냐", "만일", "만약", "물론", "사실", "실은", "특히",
}


# ==============================================================================
# 1. 국회의원 명부 로드 (제헌 ~ 4대) — 이름 사전의 1차 소스
# ==============================================================================

def find_file(prefix: str) -> str:
    matches = glob.glob(os.path.join(INPUT_DIR, f"{prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"'{prefix}'로 시작하는 CSV를 {INPUT_DIR}/ 에서 찾지 못했습니다"
        )
    return matches[0]


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

ASSEMBLY_NUM_KOREAN = {1: "제헌", 2: "제2대", 3: "제3대", 4: "제4대"}


def load_meta() -> pl.DataFrame:
    meta_files = [
        (find_file("제헌"), "제헌"),
        (find_file("2대"), "제2대"),
        (find_file("3대"), "제3대"),
        (find_file("4대"), "제4대"),
    ]
    frames = []
    for path, label in meta_files:
        df = pl.read_csv(path)
        df = df.rename({k: v for k, v in RENAME_MAP.items() if k in df.columns})
        df = df.with_columns(pl.lit(label).alias("Assembly_Num"))
        frames.append(df.select([
            "Assembly_Num", "Speaker", "Party", "Committee", "Region",
            "Gender", "Election_Count", "Election_Method",
        ]))
    meta = pl.concat(frames, how="vertical_relaxed")
    # 조인 키 정규화: 명부 쪽 이름에 들어간 공백/특수문자 제거
    meta = meta.with_columns(
        pl.col("Speaker").str.strip_chars().str.replace_all(r"\s+", "").alias("Speaker")
    )
    # 동일 (대수, 이름) 중복 시 첫 행만 (조인 시 row 폭증 방지)
    return meta.unique(subset=["Assembly_Num", "Speaker"], keep="first")


# 사용자가 직접 보강할 수 있는 직책자 이름 목록 (선택).
#   input/officer_names.csv  형식: Assembly_Num,Title_Role,Speaker
def load_manual_officers() -> list[str]:
    path = os.path.join(INPUT_DIR, "officer_names.csv")
    if not os.path.exists(path):
        return []
    df = pl.read_csv(path)
    if "Speaker" not in df.columns:
        return []
    return [s for s in df["Speaker"].to_list() if s]


# ==============================================================================
# 2. 발언 청크 분할 및 헤더 파싱
# ==============================================================================

def title_to_assembly_num(title_expr: pl.Expr) -> pl.Expr:
    num_str = title_expr.str.extract(r"^제(\d+)대국회", 1)
    return (
        num_str.cast(pl.Int64, strict=False)
        .replace_strict(ASSEMBLY_NUM_KOREAN, default=None)
        .alias("Assembly_Num")
    )


# ------------------------------------------------------------------------------
# Title 컬럼 해체 (v3 추가)
# ------------------------------------------------------------------------------
# 예: "제1대국회 제18회(임시회) 제3차 국회본회의(전체회의) (1949.01.10.)"
#      -> 대수=제헌, 회기=18, 회기구분=임시회, 차수=3, 회의체=국회본회의
# v2까지는 Title을 통문자열로만 들고 있어서 회의에서 요구된
# "회기별 출현 빈도"(참석자 2, 36:25)를 집계할 축이 아예 없었다.
TITLE_PARTS = [
    pl.col("Title").str.extract(r"제(\d+)회", 1).cast(pl.Int32, strict=False)
      .alias("Session_No"),                                   # 회기 (제N회)
    pl.col("Title").str.extract(r"제\d+회\(([^)]+)\)", 1)
      .alias("Session_Type"),                                 # 임시회/정기회
    pl.col("Title").str.extract(r"제(\d+)차", 1).cast(pl.Int32, strict=False)
      .alias("Sitting_No"),                                   # 차수 (제N차)
    pl.col("Title").str.extract(r"제\d+차\s+([^(（]+)", 1)
      .str.strip_chars().alias("Meeting_Body"),               # 국회본회의 / OO위원회
]


# 직책 문자열은 표기 흔들림이 심하다. 실제 데이터에 "법제사법위원장"과
# "법제사법위원회위원장"이 공존한다. 원본 Title_Role은 그대로 보존하되,
# 집계·필터용으로 쓸 수 있는 대분류 Role_Group을 따로 만든다.
def role_group(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col == "").then(pl.lit("미상"))
        .when(col.str.contains(r"국무총리|국무위원|장관|차관")).then(pl.lit("국무위원"))
        .when(col.str.contains(r"의장")).then(pl.lit("의장단"))
        .when(col.str.contains(r"위원장")).then(pl.lit("위원장"))
        .when(col.str.contains(r"정부위원|청장|국장|실장|부장")).then(pl.lit("정부위원"))
        .when(col.str.contains(r"사무총장|서기")).then(pl.lit("사무처"))
        .when(col.str.contains(r"증인|참고인")).then(pl.lit("증인·참고인"))
        .when(col.str.contains(r"의원|^위원$|전문위원")).then(pl.lit("의원"))
        .otherwise(pl.lit("기타"))
        .alias("Role_Group")
    )


def split_chunks(file_path: str) -> pl.LazyFrame:
    """회의록 1행 -> 발언 1건 단위로 분해."""
    return (
        pl.scan_csv(file_path)
        .with_columns([title_to_assembly_num(pl.col("Title"))] + TITLE_PARTS)
        .with_columns(pl.col("Content").str.split("◯").alias("Speech_Chunk"))
        # empty_as_null: polars 2.0 기본값 변경 예고에 대비해 명시 지정.
        # 빈 리스트(내용 없는 회의록)는 null 행으로 만든 뒤 아래에서 걸러낸다.
        .explode("Speech_Chunk", empty_as_null=True)
        .with_columns(pl.col("Speech_Chunk").str.strip_chars())
        .filter(
            pl.col("Speech_Chunk").is_not_null() & (pl.col("Speech_Chunk") != "")
        )
        # 회의 내 발언 순번. 이게 있어야 발언을 고유하게 다시 지목할 수 있고
        # "누가 먼저 말했나" 같은 순서 분석도 가능해진다.
        .with_columns(
            pl.int_range(pl.len()).over(["Title"]).add(1).cast(pl.Int32)
            .alias("Speech_Seq")
        )
    )


def parse_header(lazy: pl.LazyFrame, name_alt: str) -> pl.LazyFrame:
    """사전 기반으로 Title_Role / Speaker / Speech 를 추출한다.

    v1과 달리 원문(Speech_Chunk)을 사전에 변형하지 않는다. 헤더 인식과
    본문 절단을 같은 정규식으로 수행하므로 본문은 원문 그대로 보존된다.
    """
    p = build_header_patterns(name_alt)
    C = pl.col("Speech_Chunk")
    return lazy.with_columns([
        pl.coalesce([
            C.str.extract(p["v1"], 1),   # [직책] [이름]
            C.str.extract(p["v2"], 2),   # [이름] [직책]
            C.str.extract(p["v3"], 1),   # [직책]만
        ]).fill_null("").alias("Title_Role"),
        pl.coalesce([
            C.str.extract(p["v1"], 2),
            C.str.extract(p["v2"], 1),
            C.str.extract(p["v4"], 1),   # [이름]만
        ])
        .fill_null("")
        # "이 승만" 처럼 공백이 낀 채 잡힌 이름을 조인 키 형태로 정규화
        .str.replace_all(r"\s+", "")
        .alias("Speaker"),
        C.str.replace(p["removal"], "").str.strip_chars().alias("Speech"),
    ])


def apply_fallback(lazy: pl.LazyFrame) -> pl.LazyFrame:
    """사전 매칭 실패분에 한해 [직책] 뒤 2~4글자를 이름으로 추정(검수 대상)."""
    need = (pl.col("Title_Role") != "") & (pl.col("Speaker") == "")
    cand = pl.col("Speech").str.extract(FALLBACK_NAME_RE, 1)
    ok = need & cand.is_not_null() & ~cand.is_in(list(NAME_STOPWORDS))
    return lazy.with_columns([
        pl.when(ok).then(cand).otherwise(pl.col("Speaker")).alias("Speaker"),
        pl.when(ok)
        .then(pl.col("Speech").str.replace(FALLBACK_NAME_RE, "").str.strip_chars())
        .otherwise(pl.col("Speech"))
        .alias("Speech"),
        ok.alias("_fallback_used"),
    ])


# ==============================================================================
# 2-1. Pass 1: 명부에 없는 직책자 이름을 데이터에서 학습
# ==============================================================================

# 복성(複姓): 이 성씨로 시작할 때만 4글자 이름을 허용한다.
COMPOUND_SURNAMES = {"남궁", "황보", "선우", "독고", "사공", "서문", "제갈"}


def learn_officer_names(residual_counts: Counter, min_count: int = 5,
                        min_share: float = 0.03, concentration: float = 0.7):
    """직책별 잔여 문자열에서 이름을 추정한다.

    residual_counts: Counter({"이인국적법": 3, "이인지금질": 12, ...})
        (직책을 떼어낸 뒤 공백 제거하고 앞 5글자만 남긴 문자열의 빈도)

    판정 로직
      1) 앞 2글자 접두사 중 충분히 자주 나오는 것을 이름 후보로 본다.
      2) 그 후보 뒤 3번째 글자의 분포를 본다.
         - 한 글자가 압도적(>=concentration)  -> 3글자 이름 ("송방"+"용")
         - 여러 글자로 흩어짐                 -> 2글자 이름 ("이인")
         (본문 첫 단어는 매번 달라지므로 흩어지고, 이름의 3번째 글자는 고정된다)
      3) 한국인 이름은 2~3글자가 사실상 전부이므로 3글자에서 끊는다.
         복성(남궁·황보 등)일 때만 4글자까지 허용.
         (이 상한이 없으면 발언 수가 적고 본문 첫 단어가 늘 같은 직책자에서
          "이철원" + "낭"(낭독하겠읍니다) -> "이철원낭" 처럼 과확장된다)

    반환: {이름: (근거 빈도, 서로 다른 잔여문자열 수)}  — 사람이 검수할 근거 포함
    """
    total = sum(residual_counts.values())
    if total == 0:
        return {}

    by_prefix2 = Counter()
    for s, n in residual_counts.items():
        if len(s) >= 2:
            by_prefix2[s[:2]] += n

    learned = {}
    for p2, n2 in by_prefix2.items():
        if n2 < min_count or n2 / total < min_share:
            continue
        if not re.fullmatch(r"[가-힣]{2}", p2):
            continue
        max_len = 4 if p2 in COMPOUND_SURNAMES else 3
        name = prefix = p2
        while len(prefix) < max_len:
            nxt = Counter()
            for s, n in residual_counts.items():
                if s.startswith(prefix) and len(s) > len(prefix):
                    nxt[s[len(prefix)]] += n
            t = sum(nxt.values())
            if not t:
                break
            ch, cnt = nxt.most_common(1)[0]
            if cnt / t >= concentration and re.fullmatch(r"[가-힣]", ch):
                prefix = name = prefix + ch
            else:
                break
        # 불용어가 그대로 이름으로 승격되는 것만 차단한다.
        # (2글자 불용어로 시작하는 3글자 실명 — 예 "이상직" — 은 살린다)
        if name in NAME_STOPWORDS:
            continue
        distinct = sum(1 for s in residual_counts if s.startswith(p2))
        learned[name] = (n2, distinct)
    return learned


def collect_officer_candidates(lazy: pl.LazyFrame) -> dict:
    """Pass 1 결과에서 '직책은 잡혔으나 이름 미상'인 행의 잔여 문자열을 모은다."""
    resid = (
        lazy.filter((pl.col("Title_Role") != "") & (pl.col("Speaker") == ""))
        .select([
            pl.col("Assembly_Num"),
            pl.col("Title_Role"),
            pl.col("Speech")
            .str.replace_all(r"\s+", "")
            .str.slice(0, 5)
            .alias("Residual"),
        ])
        .group_by(["Assembly_Num", "Title_Role", "Residual"])
        .agg(pl.len().alias("n"))
    )
    df = _collect(resid)

    buckets = defaultdict(Counter)
    for row in df.iter_rows(named=True):
        if row["Residual"]:
            buckets[(row["Assembly_Num"], row["Title_Role"])][row["Residual"]] += row["n"]
    return buckets


def _collect(lazy: pl.LazyFrame) -> pl.DataFrame:
    """대용량(수 GB) 대비 streaming 엔진 우선 사용."""
    try:
        return lazy.collect(engine="streaming")
    except TypeError:
        return lazy.collect(streaming=True)
    except Exception:
        return lazy.collect()


# ==============================================================================
# 3. 텍스트 정규화 / 키워드 유틸 (v1 유지 + 오탐 진단 추가)
# ==============================================================================

def normalize_text(col: pl.Expr) -> pl.Expr:
    result = (
        col.str.replace_all(r"[\r\n\t]+", " ")
        .str.replace_all(r"-{3,}|_{3,}", " ")
    )
    # 낱글자로 쪼개진 2~5글자 영문 약어 복원: "u n" -> "un"
    for _ in range(4):
        result = result.str.replace_all(r"\b([A-Za-z])\s+([A-Za-z]\b)", r"${1}${2}")
    return result.str.replace_all(r"\s{2,}", " ").str.strip_chars()


def normalize_for_keyword_matching(col: pl.Expr) -> pl.Expr:
    normalized = normalize_text(col)
    try:  # polars >= 1.x 네이티브 NFKC (map_elements보다 훨씬 빠름)
        return normalized.str.normalize("NFKC")
    except AttributeError:
        return normalized.map_elements(
            lambda s: unicodedata.normalize("NFKC", s) if s is not None else s,
            return_dtype=pl.Utf8,
        )


def fix_legacy_spelling(col: pl.Expr) -> pl.Expr:
    legacy_map = {
        r"있읍니다": "있습니다",
        r"없읍니다": "없습니다",
        r"하였읍니다": "하였습니다",
        r"되었읍니다": "되었습니다",
        r"드리겠읍니다": "드리겠습니다",
        r"됩니다마는": "됩니다만",
    }
    for pattern, repl in legacy_map.items():
        col = col.str.replace_all(pattern, repl)
    return col


def build_keyword_regex(keyword: str) -> str:
    r"""글자 사이 공백을 허용: "노동" -> "노\s*동"

    주의(v2 추가): 이 완화는 오탐을 만든다. 예를 들어 "이민"의 완화형
    `이\s*민`은 "…하니 이 민족은…"의 "이 민"에도 걸린다. 그래서 아래
    집계에서는 엄격형(공백 불허) 카운트도 함께 내고, 그 차이를
    Keyword_Spaced_Extra 로 남겨 오탐 규모를 검수할 수 있게 한다.
    """
    return r"\s*".join(re.escape(ch) for ch in keyword)


KEYWORD_GROUPS = {
    "Technology": ("development", ["기술"]),
    "Income": ("development", ["소득"]),
    "Labor": ("development", ["노동"]),
    "Immigration": ("boundary", ["이민"]),
    "Migration": ("boundary", ["이주"]),
    "Foreigner": ("boundary", ["외국인", "외인", "타국인"]),
}
KEYWORDS_EN = list(KEYWORD_GROUPS.keys())
DISCOURSE_MAP = {en: g for en, (g, _) in KEYWORD_GROUPS.items()}


# ==============================================================================
# 자가검증 (데이터 없이 파서 규칙만 확인)
# ==============================================================================

SELFTEST_CASES = [
    # (입력, 기대 직책, 기대 이름)
    ("의장이승만 지금제3차회의를개회하겠읍 니다.", "의장", "이승만"),
    ("사무총장전규홍 곧개회를하겠읍니다.", "사무총장", "전규홍"),
    ("부의장신익희 시방말씀하신", "부의장", "신익희"),
    ("위원장대리 송방용 이 안건에 대해서", "위원장대리", "송방용"),
    ("위원장대리백남식 그러면 다음", "위원장대리", "백남식"),
    ("사무총장대리 정홍섭 그 다음", "사무총장대리", "정홍섭"),
    ("국무총리서리 허정 지금 말씀드리겠읍니다", "국무총리서리", "허정"),
    ("법무부장관 이인 국적법 초안을 여러분에게", "법무부장관", "이인"),
    ("교통부장관 허정 인사의 말씀이 늦어서", "교통부장관", "허정"),
    ("운영위원장 조순 그러면 다음으로", "운영위원장", "조순"),
    ("위원 허정 의원선출 방법은 의장이", "위원", "허정"),
    ("조순 그러면 회의를 속개하겠습니다", "", "조순"),
    ("김철수 의원 이것은 말이 안 됩니다", "의원", "김철수"),
    ("의장 이 승만 그러면", "의장", "이승만"),          # 이름 중간 공백
    ("법제사법위원장 이인 지금 보고드립니다", "법제사법위원장", "이인"),
    ("육군본부인사국장 김만수 저도 그렇게", "육군본부인사국장", "김만수"),
    ("증인 이원등 저는 그때", "증인", "이원등"),
    ("김철수(金哲洙) 의원 말씀드립니다", "의원", "김철수"),
]

SELFTEST_NAMES = [
    "이승만", "신익희", "전규홍", "허정", "이인", "조순", "송방용", "김두진",
    "박흥규", "백남식", "변영태", "이호", "김훈", "장면", "정준", "김식",
    "이석", "김철수", "정홍섭", "김만수", "이원등",
]


def run_selftest() -> int:
    alt = build_name_alternation(SELFTEST_NAMES)
    df = pl.DataFrame({"Speech_Chunk": [c[0] for c in SELFTEST_CASES]})
    got = parse_header(df.lazy(), alt).collect()
    fails = 0
    for (src, exp_t, exp_n), t, n, sp in zip(
        SELFTEST_CASES, got["Title_Role"], got["Speaker"], got["Speech"]
    ):
        ok = (t == exp_t) and (n == exp_n)
        fails += 0 if ok else 1
        print(f"[{'OK ' if ok else 'FAIL'}] {src[:34]:<34} -> 직책={t!r:<18} 이름={n!r:<8} 본문={sp[:22]!r}")
        if not ok:
            print(f"        기대: 직책={exp_t!r} 이름={exp_n!r}")
    print(f"\n{len(SELFTEST_CASES) - fails}/{len(SELFTEST_CASES)} 통과")
    return 1 if fails else 0


if os.environ.get("SELFTEST"):
    sys.exit(run_selftest())


# ==============================================================================
# 4. 실행: 2-pass 파싱
# ==============================================================================
print("[1/6] 국회의원 명부 로드 …")
meta_df = load_meta()
member_names = set(meta_df["Speaker"].to_list())
manual_names = set(load_manual_officers())
print(f"      명부 이름 {len(member_names)}개, 수동 보강 {len(manual_names)}개")

print("[2/6] Pass 1: 명부 사전으로 1차 파싱 · 직책자 이름 학습 …")
base = split_chunks(SPEECH_FILE)
pass1 = parse_header(base, build_name_alternation(member_names | manual_names))
buckets = collect_officer_candidates(pass1)

learned_rows = []          # (대수, 직책, 이름, 근거빈도, 서로다른잔여문자열수)
learned_names = set()
for (asm, role), counter in sorted(buckets.items()):
    for name, (freq, distinct) in sorted(learn_officer_names(counter).items()):
        if name in member_names:
            continue
        learned_rows.append((asm, role, name, freq, distinct))
        learned_names.add(name)
print(f"      학습된 직책자 이름 {len(learned_names)}개: "
      f"{', '.join(sorted(learned_names)[:20])}{' …' if len(learned_names) > 20 else ''}")

print("[3/6] Pass 2: 확장 사전으로 재파싱 …")
all_names = member_names | manual_names | learned_names
final_lazy = parse_header(base, build_name_alternation(all_names))
final_lazy = apply_fallback(final_lazy)

# 이름 출처 분류
final_lazy = final_lazy.with_columns(
    pl.when(pl.col("Speaker") == "").then(pl.lit("none"))
    .when(pl.col("_fallback_used")).then(pl.lit("fallback"))
    .when(pl.col("Speaker").is_in(list(member_names))).then(pl.lit("member"))
    .when(pl.col("Speaker").is_in(list(learned_names | manual_names))).then(pl.lit("learned"))
    .otherwise(pl.lit("fallback"))
    .alias("Speaker_Source")
).drop("_fallback_used")


# ==============================================================================
# 5. 명부 조인
# ==============================================================================
print("[4/6] 정당 메타데이터 조인 …")
final_lazy = final_lazy.join(
    meta_df.lazy(), on=["Speaker", "Assembly_Num"], how="left"
).with_columns([
    pl.col("Party").fill_null(pl.lit("미상/직책자")),
    pl.when(pl.col("Speaker") == "").then(pl.lit("미상"))
    .when(pl.col("Party").is_not_null()).then(pl.lit("의원"))
    .otherwise(pl.lit("정부/기타 직책자"))
    .alias("Speaker_Type"),
])


# ==============================================================================
# 6. 키워드 집계
# ==============================================================================
print("[5/6] 키워드 집계 …")
final_lazy = final_lazy.with_columns(
    fix_legacy_spelling(normalize_for_keyword_matching(pl.col("Speech")))
    .alias("Speech_Norm")
)

kw_exprs, strict_exprs = [], []
for en, (_, kr_list) in KEYWORD_GROUPS.items():
    loose = [
        pl.col("Speech_Norm").str.count_matches(build_keyword_regex(kr))
        for kr in kr_list
    ]
    strict = [
        pl.col("Speech_Norm").str.count_matches(re.escape(kr)) for kr in kr_list
    ]
    kw_exprs.append(pl.sum_horizontal(loose).alias(en))
    strict_exprs.append(pl.sum_horizontal(strict).alias(f"_{en}_strict"))

final_with_kw = (
    final_lazy.with_columns(kw_exprs + strict_exprs)
    .with_columns([
        pl.sum_horizontal(KEYWORDS_EN).alias("Keyword_Total"),
        # 공백 허용으로 '추가로' 잡힌 양 = 오탐 위험 구간 (검수용)
        (
            pl.sum_horizontal(KEYWORDS_EN)
            - pl.sum_horizontal([f"_{en}_strict" for en in KEYWORDS_EN])
        ).alias("Keyword_Spaced_Extra"),
        pl.sum_horizontal(
            [en for en in KEYWORDS_EN if DISCOURSE_MAP[en] == "development"]
        ).alias("Discourse_Development"),
        pl.sum_horizontal(
            [en for en in KEYWORDS_EN if DISCOURSE_MAP[en] == "boundary"]
        ).alias("Discourse_Boundary"),
    ])
    .drop(["Speech_Norm"] + [f"_{en}_strict" for en in KEYWORDS_EN])
)

final_collected = _collect(final_with_kw)


# ==============================================================================
# 7. 산출물
# ==============================================================================
# v3 설계 원칙
#   (1) 발언 원장(detail)은 '분석의 원천'이고, 요약 3종은 '바로 읽는 표'다.
#       원장에는 재현에 필요한 식별자를 전부 넣고, 요약에는 절대량과
#       정규화 지표를 함께 넣는다.
#   (2) 절대 빈도만으로는 발언 독점 편향이 걸린다(회의 논의). 그래서 모든
#       요약에 분모를 두 개 둔다: 발언 건수(Speech_Count)와 발언 분량
#       (Total_Chars). 실제 정규화는 '발언 1건당'이 아니라 '1,000자당'이
#       옳다. 발언 길이가 한 줄짜리부터 수천 자까지 편차가 크기 때문이다.
#   (3) 사람이 스프레드시트로 열 파일과 기계가 읽을 파일을 분리한다.
print("[6/6] 산출물 작성 …")


def write_out(df: pl.DataFrame, name: str, parquet: bool = False) -> None:
    """CSV는 Excel 한글 깨짐 방지를 위해 UTF-8 BOM으로 저장.
    원장처럼 큰 파일은 parquet도 같이 남긴다(Excel로는 어차피 못 여는 크기)."""
    df.write_csv(f"{OUTPUT_DIR}/{name}.csv", include_bom=True)
    if parquet:
        df.write_parquet(f"{OUTPUT_DIR}/{name}.parquet")


# ------------------------------------------------------------------------------
# 7-1. 발언 원장  speaker_speech_detail
# ------------------------------------------------------------------------------
# v2 대비 변경
#   + Speech_ID       : 안정적 고유키. 이게 없으면 특정 발언을 다시 지목할 수 없다.
#   + Session_No / Sitting_No / Session_Type / Meeting_Body : 회기·차수 축
#   + Year            : 시계열 집계용
#   + Role_Group      : 직책 대분류(표기 흔들림 흡수)
#   + Speech_Chars    : 정규화 분모
#   * 컬럼 순서 재배치 : 식별자 → 발언자 → 지표 → 원문(맨 뒤).
#                       원문이 중간에 있으면 스프레드시트에서 사실상 못 본다.
detail = (
    final_collected
    .with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y.%m.%d.", strict=False)
        .alias("Date_Parsed")
    )
    .with_columns([
        pl.col("Date_Parsed").dt.strftime("%Y-%m-%d").alias("Date"),
        pl.col("Date_Parsed").dt.year().cast(pl.Int32).alias("Year"),
        pl.col("Speech").str.len_chars().cast(pl.Int32).alias("Speech_Chars"),
        role_group(pl.col("Title_Role")),
    ])
    .sort(["Date_Parsed", "Title", "Speech_Seq"])
    .with_columns(
        # 예: 제헌-1948-06-02-S001-M003-0007  (대수-일자-회기-차수-발언순번)
        pl.concat_str([
            pl.col("Assembly_Num"),
            pl.col("Date").fill_null("0000-00-00"),
            pl.lit("S") + pl.col("Session_No").fill_null(0).cast(pl.Utf8).str.zfill(3),
            pl.lit("M") + pl.col("Sitting_No").fill_null(0).cast(pl.Utf8).str.zfill(3),
            pl.col("Speech_Seq").cast(pl.Utf8).str.zfill(4),
        ], separator="-").alias("Speech_ID")
    )
    .select([
        # --- 식별자 ---
        "Speech_ID", "Assembly_Num", "Year", "Date",
        "Session_No", "Session_Type", "Sitting_No", "Meeting_Body", "Speech_Seq",
        # --- 발언자 ---
        "Speaker", "Speaker_Type", "Speaker_Source",
        "Title_Role", "Role_Group", "Party", "Committee", "Region",
        # --- 지표 ---
        "Speech_Chars", *KEYWORDS_EN, "Keyword_Total",
        "Discourse_Development", "Discourse_Boundary", "Keyword_Spaced_Extra",
        # --- 원문(맨 뒤) ---
        "Title", "Speech",
    ])
)
write_out(detail, "speaker_speech_detail", parquet=True)

# 검수 대상: 이름을 못 찾았거나(none) 규칙으로 추정한(fallback) 행만
review = detail.filter(pl.col("Speaker_Source").is_in(["none", "fallback"]))
write_out(review, "unmatched_speaker_review")

# 학습된 직책자 이름 목록.
# Evidence_Count(근거 빈도)와 Distinct_Contexts(서로 다른 본문 시작 패턴 수)를
# 함께 남긴다. Distinct_Contexts가 1~2로 작으면 이름 길이 판정이 불안정할 수
# 있으니 눈으로 확인한 뒤 input/officer_names.csv 로 옮겨 고정하는 것을 권장.
if learned_rows:
    write_out(
        pl.DataFrame(
            learned_rows,
            schema=["Assembly_Num", "Title_Role", "Speaker",
                    "Evidence_Count", "Distinct_Contexts"],
            orient="row",
        ),
        "learned_officer_names",
    )


# ------------------------------------------------------------------------------
# 공통 집계 헬퍼
# ------------------------------------------------------------------------------
def agg_exprs(extra=None):
    base = (
        [pl.col(en).sum() for en in KEYWORDS_EN]
        + [
            pl.col("Discourse_Development").sum().alias("Discourse_Development"),
            pl.col("Discourse_Boundary").sum().alias("Discourse_Boundary"),
            pl.col("Keyword_Total").sum().alias("Keyword_Total"),
            pl.len().alias("Speech_Count"),
            pl.col("Speech_Chars").sum().alias("Total_Chars"),
        ]
    )
    return base + (extra or [])


def add_rates(df: pl.DataFrame, over: str = "Assembly_Num") -> pl.DataFrame:
    """1,000자당 정규화 지표 + 발언 분량 점유율.

    Share_of_Chars 는 회의에서 지적된 '소수 발언자의 발언 독점' 을 한 컬럼으로
    확인하기 위한 것이다. Keyword_Total 이 큰데 Share_of_Chars 도 크면
    그냥 말을 많이 한 것이고, Share 는 작은데 Per_1k 가 높으면 실제로 그
    주제에 집중한 것이다.
    """
    return df.with_columns([
        (pl.col("Keyword_Total") / pl.col("Total_Chars") * 1000)
        .round(3).alias("Keyword_Per_1k_Chars"),
        (pl.col("Discourse_Development") / pl.col("Total_Chars") * 1000)
        .round(3).alias("Development_Per_1k_Chars"),
        (pl.col("Discourse_Boundary") / pl.col("Total_Chars") * 1000)
        .round(3).alias("Boundary_Per_1k_Chars"),
        (pl.col("Total_Chars") / pl.col("Total_Chars").sum().over(over))
        .round(5).alias("Share_of_Chars"),
    ])


# ------------------------------------------------------------------------------
# 7-2. 발언자별 요약  speaker_keyword_summary
# ------------------------------------------------------------------------------
# [v2의 버그] group_by 에 Title_Role 이 들어가 있어서 같은 사람이 직책마다
#   다른 행으로 쪼개졌다. 실제 데이터에서 제헌 '이인' 은
#   법무부장관 / 법제사법위원장 / 법제사법위원회위원장 / 의장 → 4행,
#   제4대 '조순' 은 예산결산위원장 / 운영위원장 / 위원장 → 3행이었다.
#   ("법제사법위원장" 과 "법제사법위원회위원장" 은 같은 직책의 표기 차이다.)
#   회의에서 지적된 "직책 때문에 카운팅에 오류가 날 수 있다" 가 정확히 이 현상.
# [수정] 키를 (Assembly_Num, Speaker) 로만 잡아 1인 1행을 보장하고,
#   직책 정보는 Primary_Role(최빈 직책) / Role_Groups(대분류 목록) 로 보존한다.
speaker_keyword_summary = (
    detail
    .filter(pl.col("Speaker") != "")          # 파싱 실패 행은 집계에서 제외
    .group_by(["Assembly_Num", "Speaker"])
    .agg(agg_exprs([
        pl.col("Party").drop_nulls().mode().first().alias("Party"),
        pl.col("Speaker_Type").mode().first().alias("Speaker_Type"),
        pl.col("Title_Role").filter(pl.col("Title_Role") != "")
          .mode().first().alias("Primary_Role"),
        pl.col("Role_Group").unique().sort().str.join("; ").alias("Role_Groups"),
        pl.col("Speaker_Source").mode().first().alias("Speaker_Source"),
        pl.col("Session_No").n_unique().alias("Active_Sessions"),
        pl.col("Date").min().alias("First_Speech_Date"),
        pl.col("Date").max().alias("Last_Speech_Date"),
    ]))
    .pipe(add_rates)
    .select([
        "Assembly_Num", "Speaker", "Party", "Speaker_Type",
        "Primary_Role", "Role_Groups", "Speaker_Source",
        "Speech_Count", "Total_Chars", "Share_of_Chars", "Active_Sessions",
        "First_Speech_Date", "Last_Speech_Date",
        *KEYWORDS_EN,
        "Discourse_Development", "Discourse_Boundary", "Keyword_Total",
        "Development_Per_1k_Chars", "Boundary_Per_1k_Chars",
        "Keyword_Per_1k_Chars",
    ])
    .sort(["Assembly_Num", "Keyword_Total"], descending=[False, True])
)
write_out(speaker_keyword_summary, "speaker_keyword_summary")


# ------------------------------------------------------------------------------
# 7-3. 정당별 요약  party_keyword_summary
# ------------------------------------------------------------------------------
# [v2의 문제] 국무위원(장관)·증인·파싱 실패 행이 전부 "미상/직책자" 라는
#   가짜 정당으로 묶여 정당 비교를 오염시켰다. 장관 발언은 정당 담론이 아니라
#   정부 담론이므로 성격이 다르다.
# [수정] 정당 집계는 Speaker_Type == "의원" 인 행만 쓴다. 나머지는
#   nonparty_speaker_summary.csv 로 분리해 따로 볼 수 있게 한다.
party_keyword_summary = (
    detail
    .filter(pl.col("Speaker_Type") == "의원")
    .with_columns(
        pl.col("Party").str.strip_chars().str.replace_all(r"\s+", "").alias("Party")
    )
    .group_by(["Assembly_Num", "Party"])
    .agg(agg_exprs([
        pl.col("Speaker").n_unique().alias("Distinct_Speakers"),
    ]))
    .pipe(add_rates)
    .with_columns(
        (pl.col("Speech_Count") / pl.col("Distinct_Speakers"))
        .round(2).alias("Speeches_Per_Speaker")
    )
    .select([
        "Assembly_Num", "Party", "Distinct_Speakers",
        "Speech_Count", "Speeches_Per_Speaker", "Total_Chars", "Share_of_Chars",
        *KEYWORDS_EN,
        "Discourse_Development", "Discourse_Boundary", "Keyword_Total",
        "Development_Per_1k_Chars", "Boundary_Per_1k_Chars",
        "Keyword_Per_1k_Chars",
    ])
    .sort(["Assembly_Num", "Keyword_Per_1k_Chars"], descending=[False, True])
)
write_out(party_keyword_summary, "party_keyword_summary")

# 정당 집계에서 뺀 비의원 발언자(국무위원·증인·사무처 등)를 따로 남긴다.
nonparty_summary = (
    detail
    .filter((pl.col("Speaker_Type") != "의원") & (pl.col("Speaker") != ""))
    .group_by(["Assembly_Num", "Role_Group", "Speaker"])
    .agg(agg_exprs([
        pl.col("Title_Role").mode().first().alias("Primary_Role"),
    ]))
    .pipe(add_rates)
    .sort(["Assembly_Num", "Keyword_Total"], descending=[False, True])
)
write_out(nonparty_summary, "nonparty_speaker_summary")


# ------------------------------------------------------------------------------
# 7-4. (신규) 회기·연도별 시계열  session_keyword_summary / year_keyword_summary
# ------------------------------------------------------------------------------
# 회의에서 명시적으로 요구된 두 가지:
#   - "그 회기에서 출연 빈도, 회기와 발언자에 따른 출연 빈도" (36:25)
#   - "이 두 담론이 어떤 관계 속에 있는지, 시기 안에서 변동 양상" (18:03)
# v2에는 시간 축 집계가 아예 없어서 Date로 직접 group_by 해야 했다.
session_keyword_summary = (
    detail
    .group_by(["Assembly_Num", "Session_No", "Session_Type"])
    .agg(agg_exprs([
        pl.col("Speaker").n_unique().alias("Distinct_Speakers"),
        pl.col("Date").min().alias("Start_Date"),
        pl.col("Date").max().alias("End_Date"),
    ]))
    .pipe(add_rates)
    .with_columns(
        # 두 담론의 상대 비중. 경계 담론이 언제 튀는지 한 컬럼으로 본다.
        (
            pl.col("Discourse_Boundary")
            / (pl.col("Discourse_Development") + pl.col("Discourse_Boundary"))
        ).round(4).alias("Boundary_Ratio")
    )
    .sort(["Assembly_Num", "Session_No"])
)
write_out(session_keyword_summary, "session_keyword_summary")

year_keyword_summary = (
    detail
    .group_by(["Year"])
    .agg(agg_exprs([pl.col("Speaker").n_unique().alias("Distinct_Speakers")]))
    .pipe(add_rates, over="Year")
    .with_columns(
        (
            pl.col("Discourse_Boundary")
            / (pl.col("Discourse_Development") + pl.col("Discourse_Boundary"))
        ).round(4).alias("Boundary_Ratio")
    )
    .drop("Share_of_Chars")     # 연도별은 자기 자신이 분모라 의미 없음
    .sort("Year")
)
write_out(year_keyword_summary, "year_keyword_summary")


# ==============================================================================
# 8. 진단 리포트
# ==============================================================================
src_counts = (
    final_collected.group_by("Speaker_Source").agg(pl.len().alias("n")).sort("n", descending=True)
)
total_rows = final_collected.height

print("\n=== 발언자 파싱 진단 ===")
print(f"총 발언 chunk 수: {total_rows:,}")
for row in src_counts.iter_rows(named=True):
    label = {
        "member": "명부 확정(신뢰도 최상)",
        "learned": "직책자 학습(신뢰도 상)",
        "fallback": "규칙 추정(검수 필요)",
        "none": "이름 미확인(검수 필요)",
    }.get(row["Speaker_Source"], row["Speaker_Source"])
    print(f"  {label:<24} {row['n']:>9,}  ({row['n'] / total_rows * 100:5.2f}%)")
print(f"  -> 검수 대상은 {OUTPUT_DIR}/unmatched_speaker_review.csv ({review.height:,}행)")

spaced_extra = final_collected["Keyword_Spaced_Extra"].sum()
kw_total = final_collected["Keyword_Total"].sum()
print("\n=== 키워드 카운팅 진단 ===")
print(f"총 키워드 매칭: {kw_total:,}")
print(f"그중 '글자 사이 공백 허용'으로 추가 확보: {spaced_extra:,} "
      f"({spaced_extra / kw_total * 100 if kw_total else 0:.2f}%)")
print("  (예: '이 민족' 이 '이민'으로 잘못 잡힐 수 있음. 비율이 크면 "
      "Keyword_Spaced_Extra > 0 인 행을 표본 검수할 것)")

print("\n=== 산출물 ===")
for name, df in [
    ("speaker_speech_detail", detail),
    ("speaker_keyword_summary", speaker_keyword_summary),
    ("party_keyword_summary", party_keyword_summary),
    ("nonparty_speaker_summary", nonparty_summary),
    ("session_keyword_summary", session_keyword_summary),
    ("year_keyword_summary", year_keyword_summary),
    ("unmatched_speaker_review", review),
]:
    print(f"  {name+'.csv':<34} {df.height:>9,} 행 x {df.width:>2} 열")

print("\n=== 정당별 (의원 발언만, 1,000자당 정규화) ===")
print(party_keyword_summary.select([
    "Assembly_Num", "Party", "Distinct_Speakers", "Speech_Count",
    "Share_of_Chars", "Development_Per_1k_Chars", "Boundary_Per_1k_Chars",
]))

print("\n=== 회기별 담론 추이 ===")
print(session_keyword_summary.select([
    "Assembly_Num", "Session_No", "Distinct_Speakers", "Total_Chars",
    "Discourse_Development", "Discourse_Boundary", "Boundary_Ratio",
]).head(12))

print("\n=== 발언자별 (상위 10, 1인 1행 보장) ===")
print(speaker_keyword_summary.select([
    "Assembly_Num", "Speaker", "Party", "Primary_Role", "Speech_Count",
    "Share_of_Chars", "Keyword_Total", "Keyword_Per_1k_Chars",
]).head(10))
