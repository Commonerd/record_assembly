import os
import re
import time
import random
import asyncio
import aiohttp
import aiofiles
import datetime
import urllib.parse
import csv
import io
from asyncio import Semaphore, Lock

# ============================================================
# 설정 (필요에 따라 조정)
# ============================================================
SAVE_DIR = "hwp_downloads"
LOG_FILE = "download_log.txt"
CSV_FILENAME = "download_metadata.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://record.assembly.go.kr/assembly/viewer/minutes/download/hwp.do?id={}"

CONCURRENT_LIMIT = 5 
MIN_INTERVAL = 0.5
MAX_INTERVAL = 1.5
MAX_RETRIES = 3
BLOCK_MIN_SEC = 300
BLOCK_MAX_SEC = 620

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

# ============================================================
# 전역 동기화 객체
# ============================================================
semaphore = Semaphore(CONCURRENT_LIMIT)
block_until = 0.0
csv_lock = Lock()  # CSV 파일 동시 쓰기 방지

def log_message(msg):
    now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_text = f"{now} {msg}"
    print(log_text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_text + "\n")

def get_original_filename(response, default_id):
    cd = response.headers.get("Content-Disposition")
    if not cd:
        return f"minutes_{default_id}.hwp"
    fname_star = re.findall(r"filename\*=UTF-8''([^;]+)", cd, flags=re.IGNORECASE)
    if fname_star:
        return urllib.parse.unquote(fname_star[0])
    fname = re.findall(r'filename="?([^";]+)"?', cd, flags=re.IGNORECASE)
    if fname:
        return urllib.parse.unquote(fname[0])
    return f"minutes_{default_id}.hwp"

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

# CSV 유틸리티
def _make_csv_line(row: list) -> str:
    """리스트를 CSV 문자열 한 줄로 변환 (UTF-8)"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(row)
    return output.getvalue()

async def append_csv_row(row: list):
    """비동기적으로 CSV 파일에 행 추가 (Lock 사용)"""
    line = _make_csv_line(row)
    async with csv_lock:
        csv_path = os.path.join(SAVE_DIR, CSV_FILENAME)
        async with aiofiles.open(csv_path, "a", encoding="utf-8") as f:
            await f.write(line)

async def download_one(session, pdf_id, total, success, skip, fail):
    global block_until

    url = BASE_URL.format(pdf_id)
    retries = 0

    while retries <= MAX_RETRIES:
        now = time.time()
        if now < block_until:
            wait = block_until - now
            log_message(f"전역 차단 대기 중... 약 {wait:.0f}초 남음")
            await asyncio.sleep(wait)
            now = time.time()

        try:
            async with semaphore:
                headers = get_headers()
                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    status = resp.status

                    if status == 200:
                        # 원본 파일명 추출
                        original_name = get_original_filename(resp, pdf_id)
                        clean_name = re.sub(r'[\/:*?"<>|]', '_', original_name)
                        if not clean_name.lower().endswith(".hwp"):
                            clean_name += ".hwp"

                        name, ext = os.path.splitext(clean_name)
                        file_path = os.path.join(SAVE_DIR, clean_name)

                        # 같은 이름 파일이 있으면 ID 붙여서 새 이름
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            clean_name = f"{name}_{pdf_id}{ext}"
                            file_path = os.path.join(SAVE_DIR, clean_name)
                            log_message(f"[{pdf_id}/{total}] 파일명 충돌 → 새 이름으로 저장: {clean_name}")

                        # 파일 저장
                        async with aiofiles.open(file_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                await f.write(chunk)

                        # 파일 크기 확인
                        file_size = os.path.getsize(file_path)

                        # 로그 출력
                        log_message(f"[{pdf_id}/{total}] ✅ 성공: {clean_name} ({file_size:,}B)")

                        # CSV 메타정보 기록
                        download_time = datetime.datetime.now().isoformat()
                        csv_row = [
                            download_time,
                            pdf_id,
                            original_name,
                            clean_name,
                            file_size,
                            f"{file_size:,}B"
                        ]
                        await append_csv_row(csv_row)

                        success[0] += 1
                        return

                    elif status in (429, 403):
                        retries += 1
                        if retries == 1:
                            block_duration = random.uniform(BLOCK_MIN_SEC, BLOCK_MAX_SEC)
                            block_until = time.time() + block_duration
                            log_message(f"⚠️ 서버 차단 감지 (코드 {status}). {block_duration:.0f}초 전체 작업 중지...")
                        else:
                            wait = min(60 * retries, 300)
                            log_message(f"[{pdf_id}/{total}] {status} 오류. {wait}초 후 개별 재시도 ({retries}/{MAX_RETRIES})...")
                            await asyncio.sleep(wait)

                    else:
                        log_message(f"[{pdf_id}/{total}] 실패 (상태 코드: {status})")
                        fail[0] += 1
                        return

        except Exception as e:
            retries += 1
            wait = random.uniform(15, 45)
            log_message(f"[{pdf_id}/{total}] 예외 발생: {e}. {wait:.1f}초 후 재시도 ({retries}/{MAX_RETRIES})")
            await asyncio.sleep(wait)

    fail[0] += 1
    log_message(f"[{pdf_id}/{total}] 최종 실패 (재시도 초과)")

async def main(start_id, end_id):
    start_time = time.time()
    total = end_id - start_id + 1

    log_message("=" * 60)
    log_message(f"비동기 분산 다운로드 시작")
    log_message(f"ID 범위: {start_id} ~ {end_id} (총 {total}개)")
    log_message(f"동시 연결 제한: {CONCURRENT_LIMIT}, 요청 간격: {MIN_INTERVAL}~{MAX_INTERVAL}초")
    log_message("=" * 60)

    success = [0]
    skip = [0]
    fail = [0]

    # CSV 헤더 생성 (파일이 없거나 비어 있을 때만)
    csv_path = os.path.join(SAVE_DIR, CSV_FILENAME)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        header = ["download_time", "pdf_id", "original_filename", "saved_filename", "file_size_bytes", "file_size_human"]
        async with aiofiles.open(csv_path, "w", encoding="utf-8") as f:
            await f.write(_make_csv_line(header))

    connector = aiohttp.TCPConnector(
        limit=CONCURRENT_LIMIT * 2,
        force_close=False,
        enable_cleanup_closed=True
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for pdf_id in range(start_id, end_id + 1):
            task = asyncio.create_task(
                download_one(session, pdf_id, total, success, skip, fail)
            )
            tasks.append(task)
            interval = random.uniform(MIN_INTERVAL, MAX_INTERVAL)
            await asyncio.sleep(interval)

        await asyncio.gather(*tasks)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    log_message("=" * 60)
    log_message("✅ 작업 완료")
    log_message(f"소요 시간: {elapsed}")
    log_message(f"성공: {success[0]} | 건너뜀: {skip[0]} | 실패: {fail[0]}")
    log_message("=" * 60 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    else:
        print("사용법: python download.py <시작 ID> <끝 ID>")
        print("예시: python download.py 1001 29006")
        sys.exit(1)

    asyncio.run(main(start, end))