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

# Python csv 모듈의 기본 필드 길이 제한(131,072자)으로 인해
# 장문의 국회 회의록 Content를 읽을 때 정렬 단계에서 오류가 나지 않도록 확장한다.
# 1 GiB를 상한으로 사용하며, 이는 회의록 한 필드에 충분히 큰 값이다.
csv.field_size_limit(1024 * 1024 * 1024)

import aiofiles
import olefile


# ============================================================
# 설정
# ============================================================

HWP_DIR = "hwp_downloads"
INPUT_DIR = "input"

# ./input 아래의 의원-당적 CSV를 국회 차수별로 자동 탐색한다.
MEMBER_CSV_SUFFIX = "의원-당적.csv"


# ① 파일 단위 요약 CSV: Title, Date, Content
OUTPUT_CSV_SUMMARY = "hwp_record_assembly_summary.csv"

# ② 발언자 단위로 행을 분리한 CSV: Title, Date, Seq, Type, Speaker, 정당, 지역, 성별, 당선횟수, 당선방법, Content
OUTPUT_CSV_SPEECHES = "hwp_record_assembly_speeches.csv"

# ③ 이승만 시기 발언분리 CSV: Date <= 1960-04-27
OUTPUT_CSV_SPEECHES_RHEE = "hwp_record_assembly_speeches_이승만시기.csv"

# ④ 추출 실패/부분성공 파일 목록
OUTPUT_CSV_FAILED = "hwp_record_assembly_failed.csv"

LOG_FILE = "hwp_record_assembly_log.txt"

BATCH_SIZE = 32
IDLE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2

# 다운로드 중인 HWP를 조기에 읽지 않도록 파일 크기/수정시각이
# 이 시간 동안 변하지 않은 파일만 추출한다.
FILE_STABLE_SEC = 4

HWPTAG_PARA_TEXT = 67

# 발언자 표시 문자 (원문에서 발언자 이름 앞에 붙는 기호, U+25EF WHITE CIRCLE)
SPEAKER_MARK = "\u25ef"

# 안건/의사일정 번호 패턴 (예: "1. 개회", "10. 하곡수집중지에관한건")
AGENDA_RE = re.compile(r"^\d+[\.\)]\s*\S")

# 파일명에서 날짜를 추출하기 위한 패턴
# 실제 국회회의록 파일명 예: "... (1948.06.01.).hwp"
# 과거 형식 예: "...___1948_05_31__.hwp"도 함께 지원한다.
FILENAME_DATE_RE = re.compile(
    r"\((\d{4})[._-](\d{1,2})[._-](\d{1,2})\.?\)"
)

FILENAME_DATE_UNDERSCORE_RE = re.compile(
    r"_+(\d{4})[._-](\d{1,2})[._-](\d{1,2})_+"
)

FILENAME_DATE_GENERIC_RE = re.compile(
    r"(?<!\d)(\d{4})[._-](\d{1,2})[._-](\d{1,2})(?!\d)"
)


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
    text = re.sub(r"[\ud800-\udfff]", "", text)

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
# 의원-당적 CSV / 발언자 매핑
# ============================================================


# multiprocessing 작업자에서 공유할 의원 매핑
_WORKER_MEMBER_MAPPINGS = {}


def init_worker(member_mappings):
    """각 worker 프로세스가 시작될 때 의원 매핑을 한 번만 전달한다."""
    global _WORKER_MEMBER_MAPPINGS
    _WORKER_MEMBER_MAPPINGS = member_mappings


def normalize_name(value):
    """이름 비교용 정규화: BOM/제로폭문자/모든 공백 제거."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return re.sub(r"\s+", "", text).strip()


def parse_assembly_number(text):
    """국회 차수를 정수로 반환한다. 제헌국회는 1대국회로 취급한다."""
    if not text:
        return None

    if "제헌국회" in text:
        return 1

    m = re.search(r"제\s*(\d+)\s*대\s*국회", text)
    if not m:
        m = re.search(r"(?<!\d)(\d+)\s*대\s*국회", text)

    if m:
        return int(m.group(1))

    return None


def read_member_csv(path):
    """
    의원-당적 CSV를 읽는다.

    원칙:
    - '의원명'을 기준으로 발언자 이름을 매칭한다.
    - 원본 의원 CSV의 실제 주요 필드는 다음과 같이 취급한다:
        3열 의원명
        4열 소속위원회
        5열 당(또는 정당)
        6열 지역
        7열 성별
        8열 당선횟수
        9열 당선방법
    - 가능하면 헤더명으로 읽어서 중간 빈 열/추가 열에도 안전하게 대응한다.
    - 헤더를 찾지 못하는 비정상 CSV에 대해서만 3~9열(1-based) 위치를
      fallback으로 사용한다.
    """
    encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    rows = None
    last_error = None

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError as e:
            last_error = e

    if rows is None:
        raise last_error or UnicodeError("CSV를 읽을 수 없습니다.")

    # --------------------------------------------------------
    # 헤더 탐색: 열 위치가 아니라 실제 헤더명으로 매핑
    # --------------------------------------------------------
    required_headers = {
        "의원명": "name",
        "지역": "region",
        "성별": "gender",
        "당선횟수": "election_count",
        "당선방법": "election_method",
    }

    # 원본 CSV에서는 '당'을 사용하는 경우가 있고, 기존 파일에서는
    # '정당'을 사용하는 경우가 있을 수 있으므로 둘 다 허용한다.
    party_header_candidates = ("당", "정당")

    header_index = None
    header_row_index = None

    for row_index, row in enumerate(rows):
        normalized_headers = [normalize_name(x) for x in row]
        found = {
            key: normalized_headers.index(key)
            for key in required_headers
            if key in normalized_headers
        }

        party_index = None
        for party_header in party_header_candidates:
            if party_header in normalized_headers:
                party_index = normalized_headers.index(party_header)
                break

        # 의원명 + 당(또는 정당) + 지역 + 성별 + 당선횟수 + 당선방법이
        # 모두 있는 행을 실제 헤더로 사용한다.
        if len(found) == len(required_headers) and party_index is not None:
            found["party"] = party_index
            header_index = found
            header_row_index = row_index
            break

    mapping = {}

    # --------------------------------------------------------
    # 5개 헤더 모두 찾은 경우: 헤더 기반 추출
    # --------------------------------------------------------
    if header_index is not None:
        for row in rows[header_row_index + 1:]:
            max_index = max(header_index.values())
            if len(row) <= max_index:
                continue

            name = normalize_name(row[header_index["의원명"]])
            if not name:
                continue

            mapping[name] = {
                "party": str(row[header_index["party"]]).strip(),
                "region": str(row[header_index["지역"]]).strip(),
                "gender": str(row[header_index["성별"]]).strip(),
                "election_count": str(row[header_index["당선횟수"]]).strip(),
                "election_method": str(row[header_index["당선방법"]]).strip(),
            }

        return mapping

    # --------------------------------------------------------
    # fallback: 헤더가 없는 CSV만 3~7열(1-based) 사용
    # --------------------------------------------------------
    for row in rows:
        if len(row) < 9:
            continue

        name = normalize_name(row[2])
        if not name:
            continue

        mapping[name] = {
            # 3열 의원명 / 4열 소속위원회 / 5열 당 / 6열 지역
            # 7열 성별 / 8열 당선횟수 / 9열 당선방법
            "party": str(row[4]).strip(),
            "region": str(row[5]).strip(),
            "gender": str(row[6]).strip(),
            "election_count": str(row[7]).strip(),
            "election_method": str(row[8]).strip(),
        }

    return mapping


def load_member_mappings(input_dir=INPUT_DIR):
    """./input 아래 모든 의원-당적 CSV를 국회 차수별로 로드한다."""
    mappings = {}
    input_path = os.path.abspath(input_dir)

    if not os.path.isdir(input_path):
        log_message(f"ℹ️ 의원-당적 입력 폴더가 없습니다: {input_path}")
        return mappings

    files = sorted(
        f for f in os.listdir(input_path)
        if f.lower().endswith(MEMBER_CSV_SUFFIX.lower())
    )

    for file_name in files:
        assembly_no = parse_assembly_number(file_name)
        if assembly_no is None:
            log_message(f"⚠️ 국회 차수를 판별하지 못해 건너뜁니다: {file_name}")
            continue

        path = os.path.join(input_path, file_name)
        try:
            mapping = read_member_csv(path)
            mappings[assembly_no] = mapping
            log_message(
                f"👥 {file_name}: 국회 {assembly_no}대 / 의원 {len(mapping):,}명 매핑 로드"
            )
        except Exception as e:
            log_message(f"⚠️ 의원 CSV 읽기 실패: {file_name}: {e}")

    return mappings


def map_speaker_metadata(speaker, assembly_no, mappings):
    """
    발언자 문자열에서 해당 국회의 의원명을 찾는다.
    직책+이름 / 이름+직책 순서를 모두 처리하며, 가장 긴 이름부터 비교한다.
    """
    blank = {
        "party": "",
        "region": "",
        "gender": "",
        "election_count": "",
        "election_method": "",
    }

    if not speaker or assembly_no is None:
        return blank

    mapping = mappings.get(assembly_no)
    if not mapping:
        return blank

    speaker_norm = normalize_name(speaker)
    if not speaker_norm:
        return blank

    for member_name in sorted(mapping.keys(), key=len, reverse=True):
        if member_name and member_name in speaker_norm:
            return mapping[member_name]

    return blank


# ============================================================
# HWP Section 파싱
# ============================================================

def parse_hwp_section_text(data):
    """
    HWP Section 내부의 record를 파싱한다.

    반환:
      (text, error)

    error가 None이 아니면 Section 자체를 끝까지 정상 파싱하지 못했다는 뜻이다.
    부분적으로 추출된 text는 참고용으로 반환하지만, 상위 함수에서는
    해당 문서를 "성공"으로 처리하지 않는다.
    """

    offset = 0
    paragraphs = []
    data_length = len(data)

    if data_length == 0:
        return "", None

    while offset + 4 <= data_length:

        header = struct.unpack(
            "<I",
            data[offset:offset + 4]
        )[0]

        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        offset += 4

        if size == 0xFFF:

            if offset + 4 > data_length:
                return "\n".join(paragraphs), (
                    "record extended-size header가 잘렸습니다"
                )

            size = struct.unpack(
                "<I",
                data[offset:offset + 4]
            )[0]
            offset += 4

        if offset + size > data_length:
            return "\n".join(paragraphs), (
                f"record payload가 Section 범위를 벗어났습니다 "
                f"(offset={offset}, size={size}, length={data_length})"
            )

        if tag_id == HWPTAG_PARA_TEXT:

            record_bytes = data[offset:offset + size]

            text = parse_hwp_para_text(record_bytes)
            text = clean_paragraph(text)

            if text:
                paragraphs.append(text)

        offset += size

    # record header(4바이트)를 읽을 수 없는 0~3바이트 잔여분은 허용한다.
    # 실제 record header가 시작됐는데 payload가 잘린 경우는 위에서 오류 처리된다.
    return "\n".join(paragraphs), None


# ============================================================
# 파일명 → Title / Date 분리
# ============================================================

def parse_title_and_date(file_name):
    """
    파일명에서 Title과 Date를 분리한다.

    지원 예:
      제1대국회_제1회_임시회__제1차_국회본회의_전체회의___1948_05_31__.hwp
      제1대국회 제1회 임시회 제2차 국회본회의(전체회의) (1948.06.01.).hwp

    반환:
      title    -> 날짜 부분을 제외한 제목
      date_str -> YYYY-MM-DD
    """

    base, _ = os.path.splitext(file_name)

    date_str = ""
    title_part = base

    # 1순위: (1948.06.01.) 형태
    m = FILENAME_DATE_RE.search(base)

    # 2순위: ___1948_05_31__ 형태
    if m:
        year, month, day = m.groups()
        title_part = base[:m.start()]
    else:
        m = FILENAME_DATE_UNDERSCORE_RE.search(base)

        if m:
            year, month, day = m.groups()
            title_part = base[:m.start()]
        else:
            # 3순위: 일반적인 YYYY.MM.DD / YYYY-MM-DD / YYYY_MM_DD
            # 파일명 뒤쪽에서 마지막 날짜를 찾는다.
            matches = list(FILENAME_DATE_GENERIC_RE.finditer(base))

            if matches:
                m = matches[-1]
                year, month, day = m.groups()
                title_part = base[:m.start()]

            else:
                year = month = day = None

    if year is not None:
        try:
            date_obj = datetime.date(
                int(year),
                int(month),
                int(day),
            )
            date_str = date_obj.isoformat()
        except ValueError:
            # 잘못된 날짜라면 날짜를 억지로 만들지 않고 빈 값으로 둔다.
            date_str = ""
            title_part = base

    # 날짜 앞쪽에 붙은 밑줄/공백 제거
    title_part = re.sub(r"[_\\s]+$", "", title_part).strip()

    # 밑줄(_)들을 공백 하나로 정리
    title = re.sub(r"_+", " ", title_part).strip()

    # 제목 앞뒤의 불필요한 공백 정리
    title = re.sub(r"\\s+", " ", title).strip()

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
    """
    HWP 파일 하나를 추출한다.

    status:
      성공     -> 모든 Section을 정상적으로 읽고 파싱함
      부분성공 -> 일부 Section은 읽었지만 하나 이상 오류가 발생함
      실패     -> 문서 자체를 읽지 못했거나 유효한 Section이 없음

    정확성을 위해 "부분성공"과 "실패" 문서는 메인 CSV에 기록하지 않는다.
    """

    file_name = os.path.basename(file_path)

    title, date_str = parse_title_and_date(file_name)
    assembly_no = parse_assembly_number(file_name)

    member_mappings = _WORKER_MEMBER_MAPPINGS

    ole = None

    try:

        if not olefile.isOleFile(file_path):
            return {
                "title": title,
                "date": date_str,
                "content": "",
                "turns": [],
                "file_name": file_name,
                "status": "실패",
                "error": "올바른 OLE 기반 HWP 파일이 아닙니다.",
                "failed_sections": [],
            }

        ole = olefile.OleFileIO(file_path)
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
                is_compressed = bool(header[36] & 1)

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
                x[1].replace("Section", "")
            )
        )

        if not sections:
            return {
                "title": title,
                "date": date_str,
                "content": "",
                "turns": [],
                "file_name": file_name,
                "status": "실패",
                "error": "BodyText/Section을 찾지 못했습니다.",
                "failed_sections": [],
            }

        section_texts = []
        failed_sections = []

        # ----------------------------------------------------
        # Section 처리
        # ----------------------------------------------------

        for section in sections:

            try:
                data = ole.openstream(section).read()

            except Exception as e:
                failed_sections.append(
                    f"{section}: OLE 스트림 읽기 실패 - {e}"
                )
                continue

            if is_compressed:

                decompressed = None
                decompress_error = None

                # 정상 HWP는 raw DEFLATE(-15)를 사용한다.
                # 일부 변형 파일을 위해 zlib wrapper도 한 번 시도한다.
                for wbits in (-15, zlib.MAX_WBITS):
                    try:
                        decompressed = zlib.decompress(
                            data,
                            wbits
                        )
                        break
                    except zlib.error as e:
                        decompress_error = e

                if decompressed is None:
                    failed_sections.append(
                        f"{section}: 압축 해제 실패 - {decompress_error}"
                    )
                    continue

                data = decompressed

            text, parse_error = parse_hwp_section_text(data)

            if parse_error is not None:
                failed_sections.append(
                    f"{section}: HWP record 파싱 실패 - {parse_error}"
                )
                # 부분 텍스트라도 디버깅용으로 확보하지만
                # 문서 전체는 성공으로 인정하지 않는다.

            if text:
                section_texts.append(text)

        extracted_text = "\n".join(section_texts).strip()

        # ----------------------------------------------------
        # 상태 결정
        # ----------------------------------------------------

        if failed_sections:
            if extracted_text:
                status = "부분성공"
                error = " | ".join(failed_sections)
            else:
                status = "실패"
                error = " | ".join(failed_sections)

            return {
                "title": title,
                "date": date_str,
                "content": extracted_text,
                "turns": [],
                "file_name": file_name,
                "status": status,
                "error": error,
                "failed_sections": failed_sections,
            }

        if not extracted_text:
            return {
                "title": title,
                "date": date_str,
                "content": "",
                "turns": [],
                "file_name": file_name,
                "status": "실패",
                "error": "빈 문서 또는 텍스트를 추출하지 못했습니다.",
                "failed_sections": [],
            }

        turns = split_speech_turns(extracted_text)

        for turn in turns:
            metadata = map_speaker_metadata(
                turn.get("speaker", ""),
                assembly_no,
                member_mappings,
            )
            turn["party"] = metadata["party"]
            turn["region"] = metadata["region"]
            turn["gender"] = metadata["gender"]
            turn["election_count"] = metadata["election_count"]
            turn["election_method"] = metadata["election_method"]

        return {
            "title": title,
            "date": date_str,
            "content": extracted_text,
            "turns": turns,
            "file_name": file_name,
            "status": "성공",
            "error": "",
            "failed_sections": [],
        }

    except Exception as e:
        return {
            "title": title,
            "date": date_str,
            "content": "",
            "turns": [],
            "file_name": file_name,
            "status": "실패",
            "error": str(e),
            "failed_sections": [],
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
    """CSV 전체를 메모리에서 읽어 정렬 후 다시 저장한다."""

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
            header = next(reader, None)

            for row in reader:
                if row:
                    rows.append(row)

        rows.sort(key=key_func)

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

        log_message("✨ 최종 정렬 완료")

    except Exception as e:
        log_message(
            f"⚠️ 정렬 오류: {e}"
        )


def valid_date_sort_key(date_str):
    """
    YYYY-MM-DD 문자열을 날짜순으로 정렬하기 위한 키.
    날짜가 없는 행은 마지막으로 보낸다.
    """

    if not date_str:
        return (1, "")

    try:
        datetime.date.fromisoformat(date_str)
        return (0, date_str)
    except ValueError:
        return (1, date_str)


def speech_sort_key(row):
    """
    발언 CSV 정렬: Date → Title → Seq.

    같은 날짜에 여러 회의가 있기 때문에 Title을 Seq보다 먼저 정렬해야
    같은 회의 내부에서 Seq 1, 2, 3... 순서가 유지된다.
    """
    date_value = row[1] if len(row) > 1 else ""
    title_value = row[0] if row else ""
    seq_value = 0

    if len(row) > 2:
        try:
            seq_value = int(row[2])
        except (ValueError, TypeError):
            seq_value = 0

    return (
        *valid_date_sort_key(date_value),
        title_value,
        seq_value,
    )

def summary_sort_key(row):
    """요약 CSV 정렬: Date → Title."""

    date_value = row[1] if len(row) > 1 else ""
    title_value = row[0] if row else ""

    return (
        *valid_date_sort_key(date_value),
        title_value,
    )


# ============================================================
# 기간 필터링 CSV 생성
# ============================================================

def make_date_filtered_csv(source_csv, output_csv, cutoff_date, label):
    """
    source_csv의 Date 열이 cutoff_date 이하인 행만 별도 CSV로 저장한다.

    - cutoff_date는 datetime.date 객체를 받는다.
    - 날짜가 없거나 잘못된 행은 제외한다.
    - source_csv의 정렬 순서를 그대로 유지한다.
    """

    if (
        not os.path.exists(source_csv)
        or os.path.getsize(source_csv) == 0
    ):
        log_message(
            f"⚠️ 기간 필터 원본 CSV가 없습니다: {source_csv}"
        )
        return 0

    count = 0

    try:
        with open(
            source_csv,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as src, open(
            output_csv,
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as dst:

            reader = csv.reader(src)
            writer = csv.writer(
                dst,
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n"
            )

            header = next(reader, None)

            if not header:
                return 0

            writer.writerow(header)

            try:
                date_index = header.index("Date")
            except ValueError:
                log_message(
                    f"⚠️ Date 열을 찾지 못했습니다: {source_csv}"
                )
                return 0

            for row in reader:

                if len(row) <= date_index:
                    continue

                date_text = row[date_index].strip()

                if not date_text:
                    continue

                try:
                    row_date = datetime.date.fromisoformat(date_text)
                except ValueError:
                    continue

                if row_date <= cutoff_date:
                    writer.writerow(row)
                    count += 1

        log_message(
            f"📌 {label}: {count:,}개 행 생성 "
            f"(Date <= {cutoff_date.isoformat()})"
        )

        return count

    except Exception as e:
        log_message(
            f"⚠️ 기간 필터 CSV 생성 실패: {e}"
        )
        return 0


# ============================================================
# 메인
# ============================================================

async def main():

    start_time = datetime.datetime.now()

    log_message("=" * 60)
    log_message("🚀 HWP → CSV 정확 추출 시작")

    # --------------------------------------------------------
    # 폴더
    # --------------------------------------------------------

    if not os.path.exists(HWP_DIR):
        os.makedirs(HWP_DIR, exist_ok=True)
        log_message(f"📁 '{HWP_DIR}' 생성")

    log_message(f"대상 폴더: {HWP_DIR}")
    log_message(
        f"출력 CSV ①(요약): {os.path.abspath(OUTPUT_CSV_SUMMARY)}"
    )
    log_message(
        f"출력 CSV ②(발언분리): {os.path.abspath(OUTPUT_CSV_SPEECHES)}"
    )
    log_message(
        f"출력 CSV ③(이승만시기): {os.path.abspath(OUTPUT_CSV_SPEECHES_RHEE)}"
    )
    log_message(
        f"출력 CSV ④(실패목록): {os.path.abspath(OUTPUT_CSV_FAILED)}"
    )

    # --------------------------------------------------------
    # 기존 CSV 삭제
    # --------------------------------------------------------

    for path in (
        OUTPUT_CSV_SUMMARY,
        OUTPUT_CSV_SPEECHES,
        OUTPUT_CSV_SPEECHES_RHEE,
        OUTPUT_CSV_FAILED,
    ):

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
            make_csv_line(["Title", "Date", "Content"])
        )

    async with aiofiles.open(
        OUTPUT_CSV_SPEECHES,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        await f.write(
            make_csv_line(
                [
                    "Title",
                    "Date",
                    "Seq",
                    "Type",
                    "Speaker",
                    "정당",
                    "지역",
                    "성별",
                    "당선횟수",
                    "당선방법",
                    "Content",
                ]
            )
        )

    async with aiofiles.open(
        OUTPUT_CSV_FAILED,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        await f.write(
            make_csv_line(
                ["FileName", "Title", "Date", "Status", "Error"]
            )
        )

    processed_files = set()

    # 파일 안정성 추적:
    # 파일 크기/mtime이 일정 시간 유지된 뒤에만 추출한다.
    file_stability = {}

    success_count = 0
    partial_count = 0
    fail_count = 0

    idle_time = 0.0

    loop = asyncio.get_running_loop()
    max_workers = 4

    log_message(f"💻 CPU 코어: {max_workers}")
    log_message(
        f"⏱️ 파일 안정화 대기: {FILE_STABLE_SEC}초"
    )

    # --------------------------------------------------------
    # 의원-당적 CSV 로드
    # --------------------------------------------------------

    member_mappings = load_member_mappings(INPUT_DIR)
    log_message(
        f"👥 국회별 의원 매핑: {len(member_mappings)}개 차수 로드"
    )

    # --------------------------------------------------------
    # multiprocessing
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(member_mappings,),
    ) as pool:

        while True:

            current_files = sorted(
                [
                    f
                    for f in os.listdir(HWP_DIR)
                    if f.lower().endswith(".hwp")
                ]
            )

            now_monotonic = asyncio.get_running_loop().time()

            unprocessed_names = [
                f
                for f in current_files
                if f not in processed_files
            ]

            stable_paths = []

            for file_name in unprocessed_names:

                path = os.path.join(HWP_DIR, file_name)

                try:
                    stat = os.stat(path)
                except OSError:
                    # 파일이 이동/삭제되었거나 아직 생성 중
                    continue

                signature = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )

                state = file_stability.get(file_name)

                if state is None or state["signature"] != signature:
                    file_stability[file_name] = {
                        "signature": signature,
                        "stable_since": now_monotonic,
                    }
                    continue

                stable_for = (
                    now_monotonic
                    - state["stable_since"]
                )

                if stable_for >= FILE_STABLE_SEC:
                    stable_paths.append(path)

            # ------------------------------------------------
            # 안정된 새 파일 있음
            # ------------------------------------------------

            if stable_paths:

                idle_time = 0.0

                batch = stable_paths[:BATCH_SIZE]

                log_message(
                    f"🔍 처리 대기 {len(unprocessed_names)}개 / "
                    f"안정화 완료 {len(stable_paths)}개 → "
                    f"이번 배치 {len(batch)}개"
                )

                futures = [
                    loop.run_in_executor(
                        pool,
                        extract_hwp_text_fast,
                        path,
                    )
                    for path in batch
                ]

                results = await asyncio.gather(*futures)

                summary_lines = []
                speech_lines = []
                failed_lines = []

                for r in results:

                    title = r["title"]
                    date_str = r["date"]
                    content = r["content"]
                    turns = r["turns"]
                    file_name = r["file_name"]
                    status = r["status"]
                    error = r.get("error", "")

                    # 성공한 문서만 메인 CSV에 기록한다.
                    if status == "성공":

                        summary_lines.append(
                            make_csv_line(
                                [title, date_str, content]
                            )
                        )

                        for t in turns:
                            speech_lines.append(
                                make_csv_line(
                                    [
                                        title,
                                        date_str,
                                        t["seq"],
                                        t["type"],
                                        t["speaker"],
                                        t.get("party", ""),
                                        t.get("region", ""),
                                        t.get("gender", ""),
                                        t.get("election_count", ""),
                                        t.get("election_method", ""),
                                        t["content"],
                                    ]
                                )
                            )

                        success_count += 1

                    elif status == "부분성공":

                        partial_count += 1

                        failed_lines.append(
                            make_csv_line(
                                [
                                    file_name,
                                    title,
                                    date_str,
                                    status,
                                    error,
                                ]
                            )
                        )

                        log_message(
                            f"⚠️ 부분성공(메인 CSV 제외): "
                            f"{file_name}: {error}"
                        )

                    else:

                        fail_count += 1

                        failed_lines.append(
                            make_csv_line(
                                [
                                    file_name,
                                    title,
                                    date_str,
                                    status,
                                    error,
                                ]
                            )
                        )

                        log_message(
                            f"⚠️ 실패(메인 CSV 제외): "
                            f"{file_name}: {error}"
                        )

                    processed_files.add(file_name)
                    file_stability.pop(file_name, None)

                if summary_lines:
                    async with aiofiles.open(
                        OUTPUT_CSV_SUMMARY,
                        "a",
                        encoding="utf-8",
                        newline=""
                    ) as f:
                        await f.write("".join(summary_lines))

                if speech_lines:
                    async with aiofiles.open(
                        OUTPUT_CSV_SPEECHES,
                        "a",
                        encoding="utf-8",
                        newline=""
                    ) as f:
                        await f.write("".join(speech_lines))

                if failed_lines:
                    async with aiofiles.open(
                        OUTPUT_CSV_FAILED,
                        "a",
                        encoding="utf-8",
                        newline=""
                    ) as f:
                        await f.write("".join(failed_lines))

            # ------------------------------------------------
            # 아직 처리하지 않은 파일이 있지만 안정화 대기 중
            # ------------------------------------------------

            elif unprocessed_names:

                # 다운로드 중인 파일이 존재할 수 있으므로
                # idle timeout을 누적하지 않는다.
                await asyncio.sleep(POLL_INTERVAL_SEC)

            # ------------------------------------------------
            # 정말 새 파일이 없음
            # ------------------------------------------------

            else:

                await asyncio.sleep(POLL_INTERVAL_SEC)
                idle_time += POLL_INTERVAL_SEC

                if idle_time >= IDLE_TIMEOUT_SEC:
                    log_message(
                        f"⏳ {IDLE_TIMEOUT_SEC}초 동안 "
                        "새 HWP가 없어 종료"
                    )
                    break

    # --------------------------------------------------------
    # 최종 정렬
    # --------------------------------------------------------

    # ① 요약 CSV: Date → Title
    sort_final_csv(
        OUTPUT_CSV_SUMMARY,
        key_func=summary_sort_key,
    )

    # ② 발언 CSV: Date → Title → Seq
    sort_final_csv(
        OUTPUT_CSV_SPEECHES,
        key_func=speech_sort_key,
    )

    # ③ 이승만 시기: 1960-04-27까지(당일 포함)
    # 원본 발언 CSV의 정렬 순서를 그대로 유지한다.
    make_date_filtered_csv(
        OUTPUT_CSV_SPEECHES,
        OUTPUT_CSV_SPEECHES_RHEE,
        datetime.date(1960, 4, 27),
        "이승만 시기 발언분리 CSV",
    )

    # ④ 실패 목록: Date → FileName
    sort_final_csv(
        OUTPUT_CSV_FAILED,
        key_func=lambda row: (
            *valid_date_sort_key(row[2] if len(row) > 2 else ""),
            row[0] if row else "",
        ),
    )

    elapsed = (
        datetime.datetime.now()
        - start_time
    )

    log_message("=" * 60)
    log_message("✅ 추출 완료")
    log_message(f"총 소요 시간: {elapsed}")
    log_message(
        f"총 처리: {len(processed_files)}개 "
        f"(성공 {success_count} / "
        f"부분성공 {partial_count} / "
        f"실패 {fail_count})"
    )
    log_message(f"결과 ①: {OUTPUT_CSV_SUMMARY}")
    log_message(f"결과 ②: {OUTPUT_CSV_SPEECHES}")
    log_message(f"결과 ③: {OUTPUT_CSV_SPEECHES_RHEE}")
    log_message(f"결과 ④: {OUTPUT_CSV_FAILED}")
    log_message("=" * 60)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())