# 세팅
mkdir record_assembly 
cd record_assembly
python3 -m venv .venv
source .venv/bin/activate
pip install aiohttp aiofiles PyMuPDF polars
python3 -m pip install pandas numpy matplotlib networkx plotly
python3 -m pip install pandas numpy matplotlib networkx plotly

# 실행
python3 download.py  1 60000

# 추출
python3 extract.py

# 날짜정렬
python3 extract_sorted.py 

# 날짜정렬 + 이승만시기 + 키워드수
python3 extract_sorted_lee_keywords.py 



