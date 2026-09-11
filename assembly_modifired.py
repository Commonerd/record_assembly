import polars as pl
import glob
import re
import unicodedata

# ==============================================================================
# 0. 회의(2026.08.08) 논의 기반 데이터 클렌징 유틸리티
# ==============================================================================
# 회의에서 확인된 원본 데이터 문제:
#   1) 맞춤법/표기 불일치 - 예: "UN" vs "u n"(알파벳 사이에 공백이 낀 경우),
#      "이스니" vs "이스 니"(한 단어 중간에 공백이 낀 경우) 등 OCR/PDF->CSV
#      변환 과정에서 생긴 것으로 추정되는 불필요한 공백 삽입.
#      -> 키워드(노동/기술/이민 등)가 이런 식으로 쪼개져 있으면 str.count_matches
#         가 이를 놓쳐 키워드 빈도가 과소 집계된다.
#   2) 옛 표기법/오타 잔존 - 예: "있읍니다" 등 현대 맞춤법과 다른 표기가 섞여 있음.
#   3) 직책-이름 매칭 오류 - 예: "OO위원장 홍길동"에서 이름이 헤더로 안 빠지고
#      본문 안에 "홍길동"이 남아있는 경우가 "종종" 있다고 언급됨.
#   4) 발언자-메타데이터(Speaker x Assembly_Num) 조인이 매칭되지 않는 경우가
#      꽤 있다고 언급됨 -> 조인 성공률을 정량적으로 확인할 필요.
#   5) 카운팅 편향 - 소수 발언자가 발언을 독점하면 절대 빈도가 왜곡될 수 있음
#      -> 절대 빈도와 발언자 수 대비 정규화 지표를 함께 산출해야 한다는 논의.
# 아래 유틸리티들은 이 문제들을 완화하기 위한 전처리/진단 로직이다.


def normalize_text(col: pl.Expr) -> pl.Expr:
    r"""
    회의에서 지적된 '중간에 공백이 낀 단어' 문제를 완화하기 위한 정규화.
    NFKC 정규화만으로는 "u n" -> "un"처럼 임의 위치에 낀 공백을 못 잡으므로,
    아래 단계를 순차 적용한다.

    주의: 이 함수는 원본 Speech가 아니라 '키워드 탐지 전용' 정규화 컬럼에만
    적용한다. 원문(Speech)은 훼손하지 않고 그대로 보존해야 인용/검수가 가능하다.
    """
    # 주의: polars(rust regex 엔진)는 look-around를 지원하지 않으므로,
    # "알파벳 한 글자씩 공백으로 쪼개진 경우 붙이기"(예: "u n" -> "un")는
    # 캡처 그룹 치환으로 구현한다. 짧은 대문자 약어(2~5글자, 예: "U N" ->
    # "UN")에 한해 적용해, 일반 영어 문장의 단어 사이 공백은 건드리지 않는다.
    result = (
        col
        # 1) 줄바꿈/탭 등 제어문자를 공백으로 통일
        .str.replace_all(r"[\r\n\t]+", " ")
        # 2) 흔한 PDF->텍스트 변환 잔재: 페이지 번호, 반복 하이픈/밑줄 등
        .str.replace_all(r"-{3,}|_{3,}", " ")
    )
    # 3) 알파벳 낱글자가 공백으로 쪼개진 2~5글자 약어를 붙인다.
    #    예: "u n" -> "un", "U N E S C O" -> "UNESCO"
    #    (look-around 없이 반복 가능한 만큼 순차적으로 축약)
    for _ in range(4):
        result = result.str.replace_all(
            r"\b([A-Za-z])\s+([A-Za-z]\b)", r"${1}${2}"
        )
    # 4) 다중 공백을 단일 공백으로 축소
    result = result.str.replace_all(r"\s{2,}", " ").str.strip_chars()
    return result


def normalize_for_keyword_matching(col: pl.Expr) -> pl.Expr:
    r"""
    한글 단어 중간에 낀 공백(예: "이스 니" -> "이스니") 문제는 언어적으로
    일반화된 규칙으로 고치기 어렵다 (정상적인 띄어쓰기와 구분이 안 됨).
    따라서 여기서는 '키워드 탐지'라는 좁은 목적에 한해, 사전에 정의된
    키워드 후보 및 유의어에 대해서만 공백-삽입 변형을 흡수하는 방식을 쓴다.
    (KEYWORD_MAP 쪽에서 정규식으로 각 글자 사이에 \s* 를 넣어 처리 -> 아래
    build_keyword_regex 참고)
    """
    normalized = normalize_text(col)
    # NFKC: 전각 영문/숫자, 호환 문자 등을 표준 형태로 통일 (예: "ＵＮ" -> "UN")
    return normalized.map_elements(
        lambda s: unicodedata.normalize("NFKC", s) if s is not None else s,
        return_dtype=pl.Utf8,
    )


def fix_legacy_spelling(col: pl.Expr) -> pl.Expr:
    """
    회의에서 언급된 '옛 표기법' 잔존 문제(예: "있읍니다" 등 두음/받침 표기 차이)
    완화. 완전한 맞춤법 정규화는 아니며, 알려진 대표 패턴만 보수적으로 치환한다.
    """
    legacy_map = {
        r"있읍니다": "있습니다",
        r"없읍니다": "없습니다",
        r"하였읍니다": "하였습니다",
        r"됩니다마는": "됩니다만",
        r"習니다": "습니다",  # 한자/한글 혼용 잔재 방지용 (드문 케이스)
    }
    for pattern, repl in legacy_map.items():
        col = col.str.replace_all(pattern, repl)
    return col


def build_keyword_regex(keyword: str) -> str:
    r"""
    한글 키워드 글자 사이에 선택적 공백(\s*)을 허용하는 정규식을 만들어,
    "노 동"처럼 중간에 공백이 낀 표기도 카운팅에서 놓치지 않도록 한다.
    예: "노동" -> "노\\s*동"
    """
    escaped_chars = [re.escape(ch) for ch in keyword]
    return r"\s*".join(escaped_chars)

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
# 직책/호칭 패턴 대폭 확장 (내부 공백 허용 및 역사적 직책 추가)
# [가-힣\s]{0,10}을 통해 "내무 분과 위원장"처럼 띄어쓰기가 포함된 직책 완벽 흡수
titles_pattern = (
    r"(?:[가-힣\s]{0,10}위원회\s*(?:위원장|부위원장|사무총장|간사)"
    r"|[가-힣\s]{0,10}(?:위원장|장관|차관|부장|국장|처장|청장)"
    r"|임시\s*의장|부의장|의장"
    r"|사무\s*총장|사무\s*차장|국무\s*총리|국무\s*위원|정부\s*위원"
    r"|대통령|부통령|대리"
    r"|위원|의원|증인|참고인|발언자)"
)

# 2. 이름 패턴 정의 (한자 병기나 괄호 부연 설명 흡수)
# 예: 홍길동(洪吉童), 이스니(무소속) 등
name_extract_pattern = r"([가-힣]{2,4}(?:\([^)]*\))?)"
name_match_pattern = r"[가-힣]{2,4}(?:\([^)]*\))?"

# 직책 뒤에 괄호 소속이나 쉼표/콜론이 붙고 이름이 나오는 이상 패턴
NAME_AFTER_TITLE_RE = (
    r"^" + titles_pattern + r"(?:\([^)]*\))?[\s,·:]+" + name_extract_pattern
)

# 회의 논의: "OO위원장 홍길동"처럼 [직책]+[이름] 구조인데도 이름이 헤더에서
# 안 빠지고 본문 속에 그대로 남는 "이상 패턴"이 종종 있었다고 언급됨.
# 원인 후보:
#   (a) 직책과 이름 사이에 공백이 아니라 쉼표/가운뎃점 등 다른 구분자가 온 경우
#   (b) 이름이 2~4자 범위를 벗어나는 경우 (외자 성명, 한자 병기 등)
#   (c) 직책 뒤에 괄호로 소속이 붙어 이름 직전 토큰이 안 맞는 경우
#       (예: "보건사회위원장(자유당) 홍길동")
# 아래 두 헬퍼가 이런 케이스에서도 본문에 남은 이름을 찾아 보정한다.


# 주의: polars(rust regex)는 look-around를 지원하지 않으므로, 뒤에 조사가
# 바로 붙어도(예: "홍길동이") 이름 4자까지만 욕심 없이 뽑도록 상한을 둔다.
NAME_AFTER_TITLE_RE = (
    r"^" + titles_pattern + r"(?:\([^)]*\))?[\s,·]+([가-힣]{2,4})"
)


def dehyphenate_header(chunk_expr: pl.Expr) -> pl.Expr:
    """
    회의에서 실제로 확인된 문제: "이스니"라는 이름이 "이스 니"처럼 한글
    글자 사이에 공백이 낀 채 들어와서, 발언자 헤더(맨 앞의 [직책]/[이름]
    부분)를 규칙 기반으로 못 뽑아내는 경우가 있다.

    이 보정에는 근본적인 모호성이 있다: "[직책] [짧은 한글 조각]"이 왔을 때
    그 조각이 "공백 낀 이름"인지, 아니면 "증인 어떤 사람이..."처럼 직책
    발언 뒤에 새 문장이 시작된 것인지는 규칙만으로 완전히 구분할 수 없다.
    따라서 이 함수는 흔한 케이스(직책 바로 뒤 1~2글자 조각 1~2개)만
    보수적으로 흡수하고, 그렇게 보정된 모든 행은 Speaker_Dehyphen_Flag로
    표시해 연구자가 별도로 검수할 수 있게 한다 (완전 자동화 대신 사람이
    최종 확인하는 절충안).

    이 보정은 Speaker/Title_Role 추출 전용이며, 최종 Speech 본문에는
    영향을 주지 않는다 (본문 추출은 원본 Speech_Chunk 기준으로 수행).
    """
    result = chunk_expr
    # Case 1: [직책] + 공백-분리된 이름 조각 (최대 2조각, 예: "위원장 이스 니")
    for n in (2, 1):
        name_groups = [r"([가-힣]{1,2})" for _ in range(n)]
        pattern = (
            r"^(" + titles_pattern + r")(?:\([^)]*\))?[\s,·]+"
            + r"\s+".join(name_groups)
        )
        name_repl = "".join(f"${{{i+2}}}" for i in range(n))
        result = result.str.replace(pattern, r"${1} " + name_repl)

    # Case 2: 직책 없이 문장 맨 앞이 곧바로 [이름] + [직책] 순서로 오는 경우
    # (예: "이스 니 의원 ..."). 이름 조각들 바로 다음에 직책이 뒤따를 때만
    # 동작하도록 직책을 뒤쪽 앵커로 강제한다. 조각 수는 최대 2개로 제한.
    result = result.str.replace(
        r"^([가-힣]{1,2})\s+([가-힣]{1,2})(\s*(?:" + titles_pattern + r"))",
        r"${1}${2}${3}",
    )
    return result


def was_dehyphenated(original_expr: pl.Expr, fixed_expr: pl.Expr) -> pl.Expr:
    """dehyphenate_header가 실제로 무언가를 바꿨는지(=이름 공백 보정이
    적용됐는지) 표시. 이 플래그가 true인 행은 회의에서 지적된 모호성이
    적용된 것이므로 우선 검수 대상이다."""
    return original_expr != fixed_expr


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
        .with_columns(
            dehyphenate_header(pl.col("Speech_Chunk")).alias("Speech_Chunk_Head_Fixed")
        )
        .with_columns(
            was_dehyphenated(
                pl.col("Speech_Chunk"), pl.col("Speech_Chunk_Head_Fixed")
            ).alias("Speaker_Dehyphen_Flag")
        )
    )

    HC = pl.col("Speech_Chunk_Head_Fixed")

    parsed_lazy = parsed_lazy.with_columns([
        # 1. 직책(Title_Role) 추출
        pl.coalesce([
            HC.str.extract(r"^(" + titles_pattern + r")\s+" + name_match_pattern, 1),
            HC.str.extract(r"^" + name_match_pattern + r"\s*(" + titles_pattern + r")", 1),
            HC.str.extract(r"^(" + titles_pattern + r")", 1),
        ]).fill_null(pl.lit("")).alias("Title_Role"),

        # 2. 이름(Speaker) 추출
        pl.coalesce([
            HC.str.extract(r"^" + titles_pattern + r"\s+" + name_extract_pattern, 1),
            HC.str.extract(r"^" + name_extract_pattern + r"\s*" + titles_pattern, 1),
            HC.str.extract(NAME_AFTER_TITLE_RE, 1),
            HC.str.extract(r"^" + name_extract_pattern + r"(?:\s+|$|:)", 1),
        ]).fill_null(pl.lit("")).alias("Speaker"),
    ])

    # 3. 본문(Speech) 추출 
    # 발언자명 뒤에 붙는 불필요한 콜론(:)이나 잔여 공백까지 말끔히 잘라냅니다.
    parsed_lazy = parsed_lazy.with_columns([
        HC.str.replace(
              r"^(?:"
              + titles_pattern + r"(?:\([^)]*\))?[\s,·:]+" + name_match_pattern
              + r"|" + titles_pattern + r"\s+" + name_match_pattern
              + r"|" + name_match_pattern + r"\s*" + titles_pattern
              + r"|" + titles_pattern
              + r"|" + name_match_pattern + r")[\s:]*",
              "",
          )
          .alias("Speech"),
    ]).drop(["Speech_Chunk", "Speech_Chunk_Head_Fixed"])

    parsed_lazy = parsed_lazy.with_columns(
        (
            (pl.col("Title_Role") != "") & (pl.col("Speaker") == "")
        ).alias("Speaker_Match_Flag")
    )

    return parsed_lazy


# ==============================================================================
# 3. 데이터 실행 및 정당 메타데이터 결합 (Join)
# ==============================================================================
speech_lazy = parse_assembly_speeches("record_assembly_sorted_rhee_keywords.csv")

# 회의 논의: "발언자별 CSV를 보면 매칭이 잘 안 된 경우가 꽤 나온다"는 지적.
# 조인 키(Speaker)에 앞뒤 공백/개행이 남아 있으면 meta_df와 완전히 동일한
# 문자열이 아니게 되어 조인이 실패한다. 회의에서 언급된 "단어 중간 공백"
# 문제도 이름 자체(예: "홍 길동")에 적용되면 조인 실패로 직결되므로, 조인
# 직전에 양쪽 Speaker 컬럼을 동일한 규칙으로 한 번 더 정규화해 맞춰준다.
speech_lazy = speech_lazy.with_columns(
    pl.col("Speaker").str.strip_chars().str.replace_all(r"\s+", "").alias("Speaker")
)
meta_df = meta_df.with_columns(
    pl.col("Speaker").str.strip_chars().str.replace_all(r"\s+", "").alias("Speaker")
)

# Left Join 실행 및 매칭되지 않은 Party는 "미상/직책자"로 채움
final_lazy = speech_lazy.join(
    meta_df.lazy(),
    on=["Speaker", "Assembly_Num"],
    how="left",
).with_columns(
    pl.col("Party").fill_null(pl.lit("미상/직책자")),
)

# --- 조인 진단: 이름이 있는데도(직책자 제외) 메타데이터 매칭에 실패한 비율 ---
# 회의에서 지적된 "매칭이 잘 안 된 경우가 꽤 나온다"를 수치로 확인하기 위한
# 진단 리포트. 최종 산출물에는 포함하지 않고 콘솔에만 출력한다.
_join_diag = (
    final_lazy
    .filter(pl.col("Speaker") != "")
    .select([
        pl.len().alias("total_named_rows"),
        (pl.col("Party") == "미상/직책자").sum().alias("unmatched_rows"),
        pl.col("Speaker_Match_Flag").sum().alias("title_without_name_rows"),
        pl.col("Speaker_Dehyphen_Flag").sum().alias("dehyphenated_rows"),
    ])
    .collect()
)

# ==============================================================================
# 4. 키워드 빈도 집계 (정당별 & 의원/직책별)
# ==============================================================================
# 키워드 컬럼명을 영어로 통일: 기술/소득/노동/이민/이주/외국인
# 회의 논의: 두 가지 담론(DISCOURSE)으로 묶어서 보면 좋겠다는 의견.
#   - 발전 담론(development): 기술/소득/노동 -> "잘 먹고 잘 사는" 담론
#   - 경계 담론(boundary): 이민/이주/외국인 -> "우리 범위를 어디까지로 볼 것인가"
#     담론. 특히 이 쪽은 데이터 규모가 작을 것으로 예상되므로(회의 중 언급)
#     키워드를 유의어까지 확장해서 포착력을 높여야 한다는 논의가 있었음.
# 아래에서는 키워드마다 유의어 리스트를 둘 수 있도록 구조를 변경하고,
# 각 유의어에 공백-삽입 변형까지 흡수하는 정규식(build_keyword_regex)을 적용한다.

KEYWORD_GROUPS = {
    # 영어 컬럼명: (담론 분류, [한글 키워드 및 유의어 리스트])
    "Technology": ("development", ["기술"]),
    "Income": ("development", ["소득"]),
    "Labor": ("development", ["노동"]),
    "Immigration": ("boundary", ["이민"]),
    "Migration": ("boundary", ["이주"]),
    "Foreigner": ("boundary", ["외국인", "외인", "타국인"]),
}
keywords_en = list(KEYWORD_GROUPS.keys())
discourse_map = {en: group for en, (group, _) in KEYWORD_GROUPS.items()}

# --- 4-1. 키워드 탐지 전용 정규화 컬럼 생성 ---
# 원문(Speech)은 그대로 보존하고, 탐지에만 쓰는 별도 컬럼(Speech_Norm)에서
# 공백/옛 표기 이슈를 완화한 뒤 그 컬럼을 기준으로 카운팅한다.
final_lazy = final_lazy.with_columns(
    fix_legacy_spelling(normalize_for_keyword_matching(pl.col("Speech"))).alias(
        "Speech_Norm"
    )
)

# --- 4-2. 키워드(및 유의어) 빈도 카운팅 ---
# 한 컬럼(예: Foreigner)에 여러 유의어가 매핑된 경우, 유의어별 카운트를 모두
# 더해 해당 컬럼 값으로 쓴다. 각 키워드는 글자 사이 공백을 허용하는 정규식으로
# 탐지하여 "노 동"처럼 쪼개진 표기도 놓치지 않는다.
kw_exprs = []
for en, (_, kr_list) in KEYWORD_GROUPS.items():
    per_synonym = [
        pl.col("Speech_Norm").str.count_matches(build_keyword_regex(kr))
        for kr in kr_list
    ]
    kw_exprs.append(pl.sum_horizontal(per_synonym).alias(en))

# LazyFrame에 키워드 표현식 연산 추가
final_with_kw = final_lazy.with_columns(kw_exprs).with_columns(
    pl.sum_horizontal(keywords_en).alias("Keyword_Total")
).with_columns([
    pl.sum_horizontal([en for en in keywords_en if discourse_map[en] == "development"])
      .alias("Discourse_Development"),
    pl.sum_horizontal([en for en in keywords_en if discourse_map[en] == "boundary"])
      .alias("Discourse_Boundary"),
]).drop("Speech_Norm")

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
        "Discourse_Development", "Discourse_Boundary",
        "Speaker_Match_Flag", "Speaker_Dehyphen_Flag",
    ])
    .sort(["Date_Parsed", "Title"])
    .drop("Date_Parsed")
)

speaker_speech_detail.write_csv("output/speaker_speech_detail.csv")

# 회의에서 요청된 별도 검수용 산출물: 파싱 신뢰도가 낮은 발언들만 모아서
# 연구자가 수작업으로 훑어볼 수 있게 한다.
#   - Speaker_Match_Flag: 직책은 파싱됐지만 이름을 못 찾은 경우
#   - Speaker_Dehyphen_Flag: 공백-낀 이름 보정 규칙이 실제로 적용된 경우
#     (해당 규칙에는 "직책 뒤 새 문장 시작"과 "공백 낀 이름"을 완전히
#     구분할 수 없는 근본적 모호성이 있으므로 반드시 육안 검수가 필요함)
unmatched_speaker_rows = speaker_speech_detail.filter(
    pl.col("Speaker_Match_Flag") | pl.col("Speaker_Dehyphen_Flag")
)
unmatched_speaker_rows.write_csv("output/unmatched_speaker_review.csv")

# ==============================================================================
# 6. 결과물 2: 의원/직책별 키워드 집계
# ==============================================================================
# 회의 논의: "절대 빈도는 담론의 강도를 보는 데 좋지만, 소수 발언자가 발언을
# 독점하면 편향될 수 있다"는 지적. 이를 감안해 절대 빈도(Keyword_Total 등)와
# 함께 발언 1건당 평균(정규화 지표)도 같이 산출해 두 관점을 모두 제공한다.
speaker_keyword_summary = (
    final_collected
    .group_by(["Assembly_Num", "Speaker", "Title_Role", "Party"])
    .agg(
        [pl.col(en).sum() for en in keywords_en]
        + [
            pl.col("Discourse_Development").sum().alias("Discourse_Development"),
            pl.col("Discourse_Boundary").sum().alias("Discourse_Boundary"),
            pl.len().alias("Speech_Count"),
        ]
    )
    .with_columns(
        # 발언 1건당 평균 키워드 언급 수 (발언 독점으로 인한 절대량 편향 보정용)
        (
            (pl.col("Discourse_Development") + pl.col("Discourse_Boundary"))
            / pl.col("Speech_Count")
        ).alias("Keyword_Per_Speech")
    )
    .sort(by="Labor", descending=True)
)

speaker_keyword_summary.write_csv("output/speaker_keyword_summary.csv")

# ==============================================================================
# 7. 결과물 3: 정당별 키워드 집계
# ==============================================================================
# 회의 논의: "회기/발언자 기준으로도 카운팅하면 정당별 편향 제거에 좋다"는
# 지적을 반영해, 정당별 절대 빈도뿐 아니라 (a) 해당 정당 소속 서로 다른
# 발언자 수(Distinct_Speakers), (b) 발언자당 평균 키워드 수도 함께 산출한다.
# Distinct_Speakers가 작은데 Keyword_Total이 큰 경우 소수 발언자의 발언
# 독점 가능성을 바로 확인할 수 있다.
party_keyword_summary = (
    final_collected
    .group_by(["Assembly_Num", "Party"])
    .agg(
        [pl.col(en).sum() for en in keywords_en]
        + [
            pl.col("Discourse_Development").sum().alias("Discourse_Development"),
            pl.col("Discourse_Boundary").sum().alias("Discourse_Boundary"),
            pl.len().alias("Speech_Count"),
            pl.col("Speaker").n_unique().alias("Distinct_Speakers"),
        ]
    )
    .with_columns(
        (
            (pl.col("Discourse_Development") + pl.col("Discourse_Boundary"))
            / pl.col("Distinct_Speakers")
        ).alias("Keyword_Per_Speaker")
    )
    .sort(by="Labor", descending=True)
)

party_keyword_summary.write_csv("output/party_keyword_summary.csv")

# ==============================================================================
# 8. 결과 콘솔 출력 (+ 데이터 클렌징 진단 요약)
# ==============================================================================
print("=== 데이터 클렌징 / 조인 진단 ===")
_total = _join_diag["total_named_rows"][0]
_unmatched = _join_diag["unmatched_rows"][0]
_title_no_name = _join_diag["title_without_name_rows"][0]
_dehyphenated = _join_diag["dehyphenated_rows"][0]
_unmatched_rate = (_unmatched / _total * 100) if _total else 0.0
print(f"이름이 파싱된 발언 chunk 수: {_total}")
print(f"메타데이터 조인 실패(미상/직책자) 건수: {_unmatched} ({_unmatched_rate:.2f}%)")
print(f"직책은 파싱됐으나 이름을 못 찾은 건수: {_title_no_name}")
print(f"이름 공백(예: '이스 니') 보정이 적용된 건수: {_dehyphenated} "
      f"(직책 뒤 새 문장 시작과 완전히 구분되지 않으므로 검수 필요)")
print(f"-> 위 두 건 모두 output/unmatched_speaker_review.csv 에서 확인 가능")

print("\n=== 의원별 발언 정리 (상위 5건) ===")
print(speaker_speech_detail.head(5))

print("\n=== 정당별 키워드 집계 ===")
print(party_keyword_summary)

print("\n=== 의원/직책별 키워드 집계 (상위 10건) ===")
print(speaker_keyword_summary.head(10))