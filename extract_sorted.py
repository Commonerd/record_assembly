import polars as pl

# LazyFrame으로 스캔 시 식별자 파싱 및 필터링 최적화
lazy_df = pl.scan_csv('record_assembly.csv')

result = (
    lazy_dfpip 
    .with_columns(
        pl.col('Title').str.extract(r'\((\d{4}\.\d{2}\.\d{2}\.)\)', 1).alias('Date')
    )
    .filter(pl.col('Date') <= '2026.08.03.')
    .sort('Date')
)

# 병렬 실행 후 CSV 저장
result.collect().write_csv('record_assembly_sorted.csv')