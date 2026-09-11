import polars as pl

# 1. 집계 대상 키워드 리스트 정의
keywords = ['기술', '소득', '노동', '이민', '이주', '외국인']

# 2. 각 키워드별 등장 횟수를 산출하는 Polars 표현식 리스트 생성
# (Content가 Null인 경우 0으로 채움)
kw_expressions = [
    pl.col('Content').str.count_matches(kw).fill_null(0).alias(kw)
    for kw in keywords
]

# 3. LazyFrame 파이프라인 구성
lazy_df = pl.scan_csv('record_assembly.csv')

result = (
    lazy_df
    # 날짜 추출 및 Date 컬럼 생성 (기존 drop 제거)
    .with_columns(
        pl.col('Title').str.extract(r'\((\d{4}\.\d{2}\.\d{2}\.)\)', 1).alias('Date')
    )
    # 이승만 정부 시기 필터링
    .filter(pl.col('Date') <= '1960.04.27.')
    # 키워드 빈도수 컬럼들 동적 추가
    .with_columns(kw_expressions)
    # 날짜 기준 정렬
    .sort('Date')
)

# 4. 병렬 스트리밍 실행 및 CSV 출력
result.collect().write_csv('record_assembly_sorted_rhee_keywords.csv')