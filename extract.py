import asyncio
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor
import io
import os
import sys

import aiofiles
import fitz  # PyMuPDF (C++ 기반 초고속 PDF 파서)

# 대용량 CSV 필드 크기 제한 해제
csv.field_size_limit(sys.maxsize)

# ============================================================
# 설정
# ============================================================
PDF_DIR = "pdf_downloads"  # PDF 폴더명
OUTPUT_CSV = "record_assembly.csv"  # 결과 CSV 파일명
LOG_FILE = "record_assembly_log.txt"  # 로그 파일명
BATCH_SIZE = 64  # 코어 활용을 위한 최적 배치 크기
IDLE_TIMEOUT_SEC = 10  # 새 파일 미감지시 자동 종료 대기시간 (초)
POLL_INTERVAL_SEC = 2  # 폴더 감시 주기 (초)


def log_message(msg):
  """콘솔 및 로그 파일 기록"""
  now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
  log_text = f"{now} {msg}"
  print(log_text)
  with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(log_text + "\n")


def extract_pdf_data_fast(file_path):
  """PyMuPDF(fitz) 기반 초고속 C-Level PDF 텍스트 추출"""
  file_name = os.path.basename(file_path)
  title, _ = os.path.splitext(file_name)

  try:
    doc = fitz.open(file_path)
    page_texts = [page.get_text() for page in doc]
    doc.close()

    raw_text = "\n".join(page_texts)

    # C-level split()을 사용해 연속된 모든 공백/개행을 공백 1개로 압축 (수 배 빠름)
    cleaned_text = " ".join(raw_text.split())

    extracted_text = cleaned_text
    status = "성공"
  except Exception as e:
    extracted_text = f"[ERROR] 텍스트 추출 실패: {str(e)}"
    status = f"실패 ({str(e)})"

  return [title, extracted_text, file_name, status]


def _make_csv_line(row: list) -> str:
  """RFC 4180 규격 적용 CSV 행 변환"""
  output = io.StringIO()
  writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
  writer.writerow(row)
  return output.getvalue()


def sort_final_csv(csv_path):
  """종료 후 CSV 전체를 Title 기준 오름차순 최종 정렬"""
  if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    return

  try:
    log_message("🧹 전체 CSV 파일 Title 기준 최종 정렬 중...")
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
      reader = csv.reader(f)
      header = next(reader, None)
      for row in reader:
        if row:
          rows.append(row)

    # Title(0번째 컬럼) 기준으로 오름차순 정렬
    rows.sort(key=lambda x: x[0])

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
      writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
      if header:
        writer.writerow(header)
      writer.writerows(rows)

    log_message("✨ 최종 오름차순 정렬 완료!")
  except Exception as e:
    log_message(f"⚠️ 정렬 중 오류 발생: {e}")


async def main():
  start_time = datetime.datetime.now()
  log_message("=" * 60)
  log_message("🚀 PyMuPDF 기반 초고속 PDF 감지/추출 & 정렬 작업 시작")

  if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR, exist_ok=True)
    log_message(f"📁 '{PDF_DIR}' 폴더가 없어 새로 생성했습니다.")

  log_message(f"대상 폴더: {PDF_DIR}")
  log_message(f"저장 파일: {os.path.abspath(OUTPUT_CSV)}")
  log_message(f"자동 종료 대기시간: {IDLE_TIMEOUT_SEC}초")
  log_message("=" * 60)

  # CSV 헤더 작성 (없는 경우에만)
  if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
    header = ["Title", "Content"]
    async with aiofiles.open(OUTPUT_CSV, "w", encoding="utf-8-sig") as f:
      await f.write(_make_csv_line(header))

  processed_files = set()
  success_count = 0
  fail_count = 0
  idle_time = 0.0

  loop = asyncio.get_running_loop()
  max_workers = os.cpu_count() or 4
  log_message(f"💻 감지된 CPU 코어 수: {max_workers}개 (멀티프로세싱 가동)")

  with ProcessPoolExecutor(max_workers=max_workers) as pool:
    while True:
      # ⭐ [정렬 핵심 1] 파일 목록을 불러올 때 sorted()로 이름순 정렬
      current_all_files = sorted(
          [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
      )

      # 미처리 파일 추출 (이미 이름순으로 정렬된 상태 유지)
      unprocessed = [
          os.path.join(PDF_DIR, f)
          for f in current_all_files
          if f not in processed_files
      ]

      if unprocessed:
        idle_time = 0.0
        batch_to_process = unprocessed[:BATCH_SIZE]

        log_message(
            f"🔍 새 파일 {len(unprocessed)}개 감지 ->"
            f" {len(batch_to_process)}개 정렬 순서대로 처리 시작 (누적 완료:"
            f" {len(processed_files)}개)"
        )

        futures = [
            loop.run_in_executor(pool, extract_pdf_data_fast, pdf_path)
            for pdf_path in batch_to_process
        ]

        # ⭐ [정렬 핵심 2] asyncio.gather는 제출된 순서(이름순)를 100% 보장함
        results = await asyncio.gather(*futures)

        csv_rows = []
        for item in results:
          title, content, file_name, status = item
          csv_rows.append([title, content])
          processed_files.add(file_name)

          if status == "성공":
            success_count += 1
          else:
            fail_count += 1
            log_message(f"⚠️ [{file_name}] 추출 문제: {status}")

        # CSV 파일에 순서대로 기록
        csv_lines = "".join([_make_csv_line(row) for row in csv_rows])
        async with aiofiles.open(
            OUTPUT_CSV, "a", encoding="utf-8-sig"
        ) as f:
          await f.write(csv_lines)

      else:
        await asyncio.sleep(POLL_INTERVAL_SEC)
        idle_time += POLL_INTERVAL_SEC

        if idle_time >= IDLE_TIMEOUT_SEC:
          log_message(
              f"⏳ {IDLE_TIMEOUT_SEC}초 동안 새로운 PDF 파일이 추가되지 않아"
              " 작업을 종료합니다."
          )
          break

  # ⭐ [정렬 핵심 3] 실시간 추출 종료 후 전체 파일 깔끔하게 최종 오름차순 정리
  sort_final_csv(OUTPUT_CSV)

  elapsed = datetime.datetime.now() - start_time
  log_message("=" * 60)
  log_message("✅ 실시간 추출 및 정렬 완료")
  log_message(f"총 소요 시간: {elapsed}")
  log_message(
      f"총 처리 파일: {len(processed_files)}개 (성공: {success_count} / 실패:"
      f" {fail_count})"
  )
  log_message(f"결과 저장 위치: {OUTPUT_CSV}")
  log_message("=" * 60 + "\n")


if __name__ == "__main__":
  asyncio.run(main())