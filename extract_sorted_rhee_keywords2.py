import polars as pl

# 1. 키워드 정의
keywords = ['기술', '소득', '노동', '이민', '이주', '외국인']


import polars as pl

def clean_historical_assembly_text(col_name: str) -> pl.Expr:
    """
    1940~1950년대 제헌국회 속기록 데이터의 특수한 띄어쓰기 및 조사/어미 분리 현상을
    완벽하게 정제하는 Polars 전용 표현식입니다.
    """
    expr = pl.col(col_name)

    # 1. NBSP, Zero-Width Space, 줄바꿈 등 특수 공백 제거
    expr = expr.str.replace_all(r"[\xa0\u200b\r\n\t]+", " ")

    # 2. 발언자 기호(◯) 및 구분자 주변 정제
    expr = expr.str.replace_all(r"◯\s*", "◯ ")
    expr = expr.str.replace_all(r"\s*․\s*", "․")

    # 3. 자주 쪼개지는 주요 고유 어휘/단어 일괄 결합
    # 예: 의 원 -> 의원, 하 나 님 -> 하나님, 대 한 민 국 -> 대한민국
    vocab_fixes = {
        r"\b의\s+원\b": "의원",
        r"\b임\s+시\s+의\s+장\b": "임시의장",
        r"\b부\s+의\s+장\b": "부의장",
        r"\b하\s+나\s+님\b": "하나님",
        r"\b국\s+회\b": "국회",
        r"\b선\s+거\s+위\s+원\s+회\b": "선거위원회",
        r"\b투\s+표\b": "투표",
        r"\b결\s+선\s+투\s+표\b": "결선투표",
    }
    for pattern, repl in vocab_fixes.items():
        expr = expr.str.replace_all(pattern, repl)

    # 4. 한국어 조사, 연결어미, 종결어미 앞의 불필요한 공백 제거 (핵심 패턴)
    # 한 글자 어미/조사가 연속으로 분리되어 있는 경우를 대비해 다단계(Multi-pass) 적용
    josa_eomi_pattern = (
        r"\s+(가|이|은|는|을|를|의|에|과|와|도|로|으로|고|며|면|하|하고|하여|해서|했|함|하며|"
        r"합|입|되|되며|되여|되어서|되어서는|되었|습니다|니다|의하야|가지고|하야|되었읍니다|"
        r"하옵니다|시옵소서|하오나|하오니)"
    )
    
    # 3회 반복 적용하여 "되 였 습니다" -> "되었 습니다" -> "되었습니다" 순차 결합
    for _ in range(3):
        expr = expr.str.replace_all(josa_eomi_pattern, r"$1")

    # 5. 연속된 공백 하나로 축소 및 양끝 공백 제거
    expr = expr.str.replace_all(r"\s{2,}", " ")
    expr = expr.str.strip_chars()

    return expr


# 2. 한국어 텍스트 정제 표현식 (Polars 전용)
def clean_korean_text(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        # NBSP(\xa0), 줄바꿈, 탭 등을 일반 공백으로 치환
        .str.replace_all(r"[\xa0\u200b\r\n\t]+", " ")
        # 조사 및 대표 어미 앞의 잘못된 띄어쓰기 제거
        .str.replace_all(
            r"\s+(을|를|이|가|은|는|에|의|로|으로|고|와|과|도|하|하고|하여|해서|했|습니다|합니다|입니다|었습니다|였습니다|되|되며|되여)", 
            r"$1"
        )
        # 연속된 공백 하나로 합치기
        .str.replace_all(r"\s{2,}", " ")
        # 양끝 공백 제거
        .str.strip_chars()
    )

# 3. 키워드 집계 표현식
kw_expressions = [
    pl.col('Content').str.count_matches(kw).fill_null(0).alias(kw)
    for kw in keywords
]

# 4. LazyFrame 파이프라인
lazy_df = pl.scan_csv('record_assembly.csv')

result = (
    lazy_df
    # 날짜 추출
    .with_columns(
        pl.col('Title').str.extract(r'\((\d{4}\.\d{2}\.\d{2}\.)\)', 1).alias('Date')
    )
    # 이승만 정부 시기 필터링
    .filter(pl.col('Date') <= '1960.04.27.')
    
    # ----------------------------------------------------
    # ★ [추가] Content 데이터 클렌징 적용
    # ----------------------------------------------------
    .with_columns(
        clean_historical_assembly_text('Content').alias('Content')
    )
    
    # 키워드 빈도수 산출 (클렌징된 Content 기준)
    .with_columns(kw_expressions)
    # 날짜 정렬
    .sort('Date')
)

# 5. 실행 및 CSV 저장
result.collect().write_csv('record_assembly_sorted_rhee_keywords2.csv')