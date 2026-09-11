import asyncio
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor
import io
import os
import re
import struct
import sys
import zlib

import aiofiles
import olefile


# ============================================================
# 설정
# ============================================================

HWP_DIR = "hwp_downloads2"

# ① 파일 단위 요약 CSV: Title, Date, Content
OUTPUT_CSV_SUMMARY = "record_assembly_summary.csv"

# ② 발언자 단위로 행을 분리한 CSV: Title, Date, Seq, Type, Speaker, Content
OUTPUT_CSV_SPEECHES = "record_assembly_speeches.csv"

LOG_FILE = "record_assembly_log.txt"

BATCH_SIZE = 64
IDLE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2

HWPTAG_PARA_TEXT = 67

# 발언자 표시 문자 (원문에서 발언자 이름 앞에 붙는 기호, U+25EF WHITE CIRCLE)
SPEAKER_MARK = "\u25ef"

# 안건/의사일정 번호 패턴 (예: "1. 개회", "10. 하곡수집중지에관한건")
AGENDA_RE = re.compile(r"^\d+[\.\)]\s*\S")

# 파일명에서 날짜(YYYY_MM_DD)를 추출하기 위한 패턴
FILENAME_DATE_RE = re.compile(r"(\d{4})_(\d{1,2})_(\d{1,2})")


# ============================================================
# 로그
# ============================================================

def log_message(msg):
    now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{now} {msg}"

    print(line)

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(line + "\n")


# ============================================================
# HWP PARA_TEXT 제어문자 정의
# ============================================================

# 8 WCHAR = 16 bytes를 차지하는 확장/인라인 제어문자
CONTROL_16BYTE = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9,
    11, 12,
    14, 15, 16, 17, 18,
    19, 20,
    21, 22, 23
}

# 2 bytes만 차지하는 문자형 제어문자
CONTROL_2BYTE = {
    0,
    10,       # LF
    13,       # CR / paragraph end
    24, 25, 26, 27,
    28, 29, 30, 31
}


# ============================================================
# PARA_TEXT 추출
# ============================================================

def parse_hwp_para_text(rec_bytes):
    """
    HWP5 PARA_TEXT를 정확하게 해석한다.

    중요한 점:
    - 일반 문자는 UTF-16LE 2바이트
    - 일부 제어문자는 16바이트 전체를 차지
    - 0x18~0x1F 등은 2바이트만 차지
    - 0x0A / 0x0D는 실제 개행으로 취급
    """

    chars = []

    i = 0
    length = len(rec_bytes)

    while i + 1 < length:

        # ----------------------------------------------------
        # WCHAR 하나 읽기
        # ----------------------------------------------------

        code = int.from_bytes(
            rec_bytes[i:i + 2],
            "little"
        )

        # ----------------------------------------------------
        # 일반 문자
        # ----------------------------------------------------

        if code >= 32:
            chars.append(
                chr(code)
            )

            i += 2
            continue

        # ----------------------------------------------------
        # 0x0A / 0x0D
        # ----------------------------------------------------

        if code in (10, 13):

            # 문단 내부 개행
            chars.append("\n")

            i += 2
            continue

        # ----------------------------------------------------
        # 16바이트 제어문자
        # ----------------------------------------------------

        if code in CONTROL_16BYTE:

            # HWP5 확장 제어문자:
            # 총 8 WCHAR = 16 bytes
            #
            # 범위를 벗어나면 안전하게 종료
            if i + 16 <= length:

                # TAB은 실제 텍스트 의미가 있으므로 보존
                if code == 9:
                    chars.append("\t")

                i += 16

            else:
                break

            continue

        # ----------------------------------------------------
        # 2바이트 제어문자
        # ----------------------------------------------------

        if code in CONTROL_2BYTE:

            # grouped space / fixed width space
            if code in (30, 31):
                chars.append(" ")

            # hyphen
            elif code == 24:
                chars.append("-")

            # 나머지는 의미 없는 제어문자
            i += 2
            continue

        # ----------------------------------------------------
        # 예상하지 못한 제어문자
        # ----------------------------------------------------

        i += 2

    return "".join(chars)


# ============================================================
# 문단 정리
# ============================================================

def clean_paragraph(text):
    """
    문자 자체를 삭제하지 않고 최소한의 정리만 수행한다.
    """

    if not text:
        return ""

    # NUL 제거
    text = text.replace("\x00", "")

    # CRLF/CR 정리
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 탭을 일반 공백으로
    text = text.replace("\t", " ")

    # ----------------------------------------
    # 줄 단위 정리
    # ----------------------------------------
    lines = []

    for line in text.split("\n"):

        # 문단 내부의 연속 공백만 정리
        line = " ".join(
            line.split()
        )

        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# HWP Section 파싱
# ============================================================

def parse_hwp_section_text(data):

    offset = 0
    paragraphs = []

    data_length = len(data)

    while offset + 4 <= data_length:

        # ----------------------------------------------------
        # Record header
        # ----------------------------------------------------

        header = struct.unpack(
            "<I",
            data[offset:offset + 4]
        )[0]

        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF

        offset += 4

        # ----------------------------------------------------
        # Extended size
        # ----------------------------------------------------

        if size == 0xFFF:

            if offset + 4 > data_length:
                break

            size = struct.unpack(
                "<I",
                data[offset:offset + 4]
            )[0]

            offset += 4

        # ----------------------------------------------------
        # 잘못된 record
        # ----------------------------------------------------

        if offset + size > data_length:
            break

        # ----------------------------------------------------
        # PARA_TEXT
        # ----------------------------------------------------

        if tag_id == HWPTAG_PARA_TEXT:

            record_bytes = data[
                offset:offset + size
            ]

            text = parse_hwp_para_text(
                record_bytes
            )

            text = clean_paragraph(
                text
            )

            if text:
                paragraphs.append(text)

        offset += size

    return "\n".join(paragraphs)


# ============================================================
# 파일명 → Title / Date 분리
# ============================================================

def parse_title_and_date(file_name):
    """
    파일명 예:
      제1대국회_제1회_임시회__제1차_국회본회의_전체회의___1948_05_31__.hwp

    -> Title: "제1대국회 제1회 임시회 제1차 국회본회의 전체회의"
    -> Date : "1948-05-31"
    """

    base, _ = os.path.splitext(file_name)

    date_str = ""

    m = FILENAME_DATE_RE.search(base)

    if m:

        year, month, day = m.groups()

        date_str = (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        )

        # 날짜 부분을 잘라내고 제목만 남긴다
        title_part = base[:m.start()]

    else:

        title_part = base

    # 밑줄(_)들을 공백 하나로 정리
    title = re.sub(
        r"_+",
        " ",
        title_part
    ).strip()

    return title, date_str


# ============================================================
# 발언자별 행 분리 (Speech Turn 분리)
# ============================================================

def split_speech_turns(content):
    """
    한 회의록 전체 텍스트(content)를 아래 세 종류의 "행"으로 나눈다.

    - 발언 (Type="발언"): "◯직함/이름 ... 발언내용" 형태의 줄로 시작하는 구간.
      다음 마커(◯, 괄호, 안건번호)가 나올 때까지 이어지는 문단들은
      같은 발언자의 발언으로 합쳐진다.
    - 안건 (Type="안건"): "1. 개회"처럼 번호로 시작하는 의사일정/안건 줄.
    - 기타 (Type="기타"): "(박수)"처럼 괄호로 감싸인 행동 지시문이나,
      그 외 발언자 표시가 없는 머리말/부가 정보 줄.

    반환값: [{"seq": 1, "type": "발언", "speaker": "의장 이승만", "content": "..."}, ...]
    """

    turns = []
    seq = 0
    current = None  # 현재 채워지고 있는 "발언" 버퍼

    def flush():
        nonlocal current

        if current is None:
            return

        text = " ".join(current["parts"]).strip()

        turns.append(
            {
                "seq": current["seq"],
                "type": "발언",
                "speaker": current["speaker"],
                "content": text,
            }
        )

        current = None

    for raw_line in content.split("\n"):

        line = raw_line.strip()

        if not line:
            continue

        # ------------------------------------------------
        # 발언자 마커로 시작하는 줄
        # ------------------------------------------------

        if line.startswith(SPEAKER_MARK):

            flush()
            seq += 1

            body = line[len(SPEAKER_MARK):]
            tokens = body.split(" ", 2)

            if len(tokens) >= 3:
                speaker = f"{tokens[0]} {tokens[1]}"
                rest = tokens[2]

            elif len(tokens) == 2:
                speaker = f"{tokens[0]} {tokens[1]}"
                rest = ""

            else:
                speaker = tokens[0]
                rest = ""

            current = {
                "seq": seq,
                "speaker": speaker,
                "parts": [rest] if rest else [],
            }

            continue

        # ------------------------------------------------
        # 괄호로 감싸인 행동 지시문 ( 예: (박수), (애국가 봉창) )
        # ------------------------------------------------

        if line.startswith("(") and line.endswith(")"):

            flush()
            seq += 1

            turns.append(
                {
                    "seq": seq,
                    "type": "기타",
                    "speaker": "",
                    "content": line[1:-1].strip(),
                }
            )

            continue

        # ------------------------------------------------
        # 안건/의사일정 번호로 시작하는 줄
        # ------------------------------------------------

        if AGENDA_RE.match(line):

            flush()
            seq += 1

            turns.append(
                {
                    "seq": seq,
                    "type": "안건",
                    "speaker": "",
                    "content": line,
                }
            )

            continue

        # ------------------------------------------------
        # 발언 도중의 줄 -> 직전 발언에 이어붙임
        # ------------------------------------------------

        if current is not None:

            current["parts"].append(line)

        # ------------------------------------------------
        # 발언자가 아직 없는 머리말/부가 정보 줄
        # ------------------------------------------------

        else:

            seq += 1

            turns.append(
                {
                    "seq": seq,
                    "type": "기타",
                    "speaker": "",
                    "content": line,
                }
            )

    flush()

    return turns


# ============================================================
# HWP 파일 추출
# ============================================================

def extract_hwp_text_fast(file_path):

    file_name = os.path.basename(
        file_path
    )

    title, date_str = parse_title_and_date(
        file_name
    )

    ole = None

    try:

        # ----------------------------------------------------
        # HWP 확인
        # ----------------------------------------------------

        if not olefile.isOleFile(file_path):

            return {
                "title": title,
                "date": date_str,
                "content": "[ERROR] 올바른 HWP 파일이 아닙니다.",
                "turns": [],
                "file_name": file_name,
                "status": "실패",
            }

        ole = olefile.OleFileIO(
            file_path
        )

        dirs = ole.listdir()

        # ----------------------------------------------------
        # 압축 여부
        # ----------------------------------------------------

        is_compressed = False

        if ["FileHeader"] in dirs:

            header = ole.openstream(
                "FileHeader"
            ).read()

            if len(header) > 36:
                is_compressed = bool(
                    header[36] & 1
                )

        # ----------------------------------------------------
        # Section 찾기
        # ----------------------------------------------------

        sections = [
            d
            for d in dirs
            if (
                len(d) == 2
                and d[0] == "BodyText"
                and d[1].startswith("Section")
            )
        ]

        sections.sort(
            key=lambda x: int(
                x[1].replace(
                    "Section",
                    ""
                )
            )
        )

        section_texts = []

        # ----------------------------------------------------
        # Section 처리
        # ----------------------------------------------------

        for section in sections:

            data = ole.openstream(
                section
            ).read()

            if is_compressed:

                try:

                    data = zlib.decompress(
                        data,
                        -15
                    )

                except zlib.error as e:

                    log_message(
                        f"⚠️ 압축 해제 실패: "
                        f"{file_name} / {section}: {e}"
                    )

                    continue

            text = parse_hwp_section_text(
                data
            )

            if text:
                section_texts.append(
                    text
                )

        # ----------------------------------------------------
        # Section 간에는 개행
        # ----------------------------------------------------

        extracted_text = "\n".join(
            section_texts
        ).strip()

        if not extracted_text:

            extracted_text = (
                "[빈 문서 또는 파싱 실패]"
            )

            turns = []

        else:

            turns = split_speech_turns(
                extracted_text
            )

        return {
            "title": title,
            "date": date_str,
            "content": extracted_text,
            "turns": turns,
            "file_name": file_name,
            "status": "성공",
        }

    except Exception as e:

        return {
            "title": title,
            "date": date_str,
            "content": f"[ERROR] HWP 텍스트 추출 실패: {e}",
            "turns": [],
            "file_name": file_name,
            "status": f"실패 ({e})",
        }

    finally:

        if ole is not None:

            try:
                ole.close()
            except Exception:
                pass


# ============================================================
# CSV 생성
# ============================================================

def make_csv_line(row):

    output = io.StringIO(
        newline=""
    )

    writer = csv.writer(
        output,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n"
    )

    writer.writerow(row)

    return output.getvalue()


# ============================================================
# CSV 최종 정렬
# ============================================================

def sort_final_csv(csv_path, key_func=None):
    """
    key_func(row) -> 정렬 키.
    지정하지 않으면 기존과 동일하게 0번째 컬럼(Title) 기준 정렬.
    """

    if key_func is None:
        key_func = lambda row: row[0]

    if (
        not os.path.exists(csv_path)
        or os.path.getsize(csv_path) == 0
    ):
        return

    try:

        log_message(
            f"🧹 최종 정렬 시작... ({os.path.basename(csv_path)})"
        )

        rows = []

        with open(
            csv_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.reader(f)

            header = next(
                reader,
                None
            )

            for row in reader:

                if row:
                    rows.append(row)

        rows.sort(
            key=key_func
        )

        with open(
            csv_path,
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            writer = csv.writer(
                f,
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n"
            )

            if header:
                writer.writerow(header)

            writer.writerows(rows)

        log_message(
            "✨ 최종 정렬 완료"
        )

    except Exception as e:

        log_message(
            f"⚠️ 정렬 오류: {e}"
        )


# ============================================================
# 메인
# ============================================================

async def main():

    start_time = datetime.datetime.now()

    log_message("=" * 60)

    log_message(
        "🚀 HWP → CSV 정확 추출 시작"
    )

    # --------------------------------------------------------
    # 폴더
    # --------------------------------------------------------

    if not os.path.exists(
        HWP_DIR
    ):

        os.makedirs(
            HWP_DIR,
            exist_ok=True
        )

        log_message(
            f"📁 '{HWP_DIR}' 생성"
        )

    log_message(
        f"대상 폴더: {HWP_DIR}"
    )

    log_message(
        f"출력 CSV ①(요약): "
        f"{os.path.abspath(OUTPUT_CSV_SUMMARY)}"
    )

    log_message(
        f"출력 CSV ②(발언분리): "
        f"{os.path.abspath(OUTPUT_CSV_SPEECHES)}"
    )

    # --------------------------------------------------------
    # 기존 CSV 삭제
    # --------------------------------------------------------

    for path in (OUTPUT_CSV_SUMMARY, OUTPUT_CSV_SPEECHES):

        if os.path.exists(path):

            try:
                os.remove(path)

            except Exception as e:

                log_message(
                    f"⚠️ 기존 CSV 삭제 실패 ({path}): {e}"
                )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    async with aiofiles.open(
        OUTPUT_CSV_SUMMARY,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        await f.write(
            make_csv_line(
                ["Title", "Date", "Content"]
            )
        )

    async with aiofiles.open(
        OUTPUT_CSV_SPEECHES,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        await f.write(
            make_csv_line(
                ["Title", "Date", "Seq", "Type", "Speaker", "Content"]
            )
        )

    processed_files = set()

    success_count = 0
    fail_count = 0

    idle_time = 0.0

    loop = asyncio.get_running_loop()

    max_workers = os.cpu_count() or 4

    log_message(
        f"💻 CPU 코어: {max_workers}"
    )

    # --------------------------------------------------------
    # multiprocessing
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as pool:

        while True:

            current_files = sorted(
                [
                    f
                    for f in os.listdir(
                        HWP_DIR
                    )
                    if f.lower().endswith(
                        ".hwp"
                    )
                ]
            )

            unprocessed = [
                os.path.join(
                    HWP_DIR,
                    f
                )
                for f in current_files
                if f not in processed_files
            ]

            # ------------------------------------------------
            # 새 파일 있음
            # ------------------------------------------------

            if unprocessed:

                idle_time = 0.0

                batch = unprocessed[
                    :BATCH_SIZE
                ]

                log_message(
                    f"🔍 새 파일 {len(unprocessed)}개 → "
                    f"{len(batch)}개 처리"
                )

                futures = [
                    loop.run_in_executor(
                        pool,
                        extract_hwp_text_fast,
                        path
                    )
                    for path in batch
                ]

                results = await asyncio.gather(
                    *futures
                )

                summary_lines = []
                speech_lines = []

                for r in results:

                    title = r["title"]
                    date_str = r["date"]
                    content = r["content"]
                    turns = r["turns"]
                    file_name = r["file_name"]
                    status = r["status"]

                    # ① 요약 CSV: 파일 하나 = 한 행
                    summary_lines.append(
                        make_csv_line(
                            [title, date_str, content]
                        )
                    )

                    # ② 발언분리 CSV: 발언/안건/기타 단위로 여러 행
                    for t in turns:

                        speech_lines.append(
                            make_csv_line(
                                [
                                    title,
                                    date_str,
                                    t["seq"],
                                    t["type"],
                                    t["speaker"],
                                    t["content"],
                                ]
                            )
                        )

                    processed_files.add(
                        file_name
                    )

                    if status == "성공":

                        success_count += 1

                    else:

                        fail_count += 1

                        log_message(
                            f"⚠️ {file_name}: "
                            f"{status}"
                        )

                async with aiofiles.open(
                    OUTPUT_CSV_SUMMARY,
                    "a",
                    encoding="utf-8",
                    newline=""
                ) as f:

                    await f.write(
                        "".join(summary_lines)
                    )

                async with aiofiles.open(
                    OUTPUT_CSV_SPEECHES,
                    "a",
                    encoding="utf-8",
                    newline=""
                ) as f:

                    await f.write(
                        "".join(speech_lines)
                    )

            # ------------------------------------------------
            # 새 파일 없음
            # ------------------------------------------------

            else:

                await asyncio.sleep(
                    POLL_INTERVAL_SEC
                )

                idle_time += (
                    POLL_INTERVAL_SEC
                )

                if idle_time >= IDLE_TIMEOUT_SEC:

                    log_message(
                        f"⏳ {IDLE_TIMEOUT_SEC}초 동안 "
                        "새 HWP가 없어 종료"
                    )

                    break

    # --------------------------------------------------------
    # 정렬 (날짜순 → 같은 날짜 내에서는 원래 순서/순번 유지)
    # --------------------------------------------------------

    # ① 요약 CSV: Title, Date, Content  → Date 기준
    sort_final_csv(
        OUTPUT_CSV_SUMMARY,
        key_func=lambda row: row[1],  # Date
    )

    # ② 발언분리 CSV: Title, Date, Seq, Type, Speaker, Content
    #    → Date 기준, 같은 날짜 안에서는 Seq(발언 순서) 기준
    sort_final_csv(
        OUTPUT_CSV_SPEECHES,
        key_func=lambda row: (
            row[1],
            int(row[2]) if row[2].isdigit() else 0,
        ),
    )

    elapsed = (
        datetime.datetime.now()
        - start_time
    )

    log_message("=" * 60)

    log_message(
        "✅ 추출 완료"
    )

    log_message(
        f"총 소요 시간: {elapsed}"
    )

    log_message(
        f"총 처리: {len(processed_files)}개 "
        f"(성공 {success_count} / "
        f"실패 {fail_count})"
    )

    log_message(
        f"결과 ①: {OUTPUT_CSV_SUMMARY}"
    )

    log_message(
        f"결과 ②: {OUTPUT_CSV_SPEECHES}"
    )

    log_message("=" * 60)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())