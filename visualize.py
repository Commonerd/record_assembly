#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이승만 시기 국회회의록 고급 담론·네트워크 분석

핵심 설계
---------
1. 기존 발전/경계 담론 분석 유지
2. 상대빈도(relative frequency) 추가
3. 키워드 × 시간 heatmap / 주요 키워드 시계열 추가
4. Bingenheimer식 'diachronic / micro-meso-macro zoomability'를 국회회의록에 맞게 변형
5. '발언흐름 네트워크'를 별도 구축
   - 같은 회의에서 연속한 두 발언을 연결
   - 이것을 직접 상호작용으로 간주하지 않음
   - future annotation에서 질문-응답/호명/반박 등의 관계가 확보되면 교체 가능
6. 네트워크 중심성 확장
   - 차수중심성(Degree)
   - 가중 차수/연결강도(Weighted Degree / Strength)
   - 매개중심성(Betweenness)
   - 근접중심성(Closeness)
   - 고유벡터중심성(Eigenvector)
   - 빈도 가중치는 연결강도로 사용하고, 최단거리 기반 지표에서는 distance=1/weight로 변환
   - '영향력'이라고 부르지 않고 관찰된 네트워크상의 구조적 중심성으로만 해석
7. Ego network 추가
8. Timeline network 추가
9. Keyword co-occurrence network 추가
10. 지식그래프는 '의원→속성→담론' 단순 트리가 아니라 event/speech 중심 구조로 재설계
11. 미래 어노테이션 필드가 없으면 해당 분석을 자동 SKIP + logs/skipped.txt 기록
12. CSV field-size 제한 완화
13. matplotlib 의존성을 제거하고 Plotly HTML을 기본으로 사용

입력 기본값
-----------
hwp_record_assembly_speeches_이승만시기.csv

출력
----
visualization_rhee_advanced/
  01_월별_담론량.html
  02_월별_담론상대빈도.html
  03_6개계열_시간Heatmap.html
  04_키워드_시간Heatmap.html
  05_주요키워드_시계열.html
  06_키워드_시간구성비Heatmap.html
  07_키워드_popularity.html
  08~12_메타데이터별_담론.html
  13_발언자_프로파일.html
  14_의원별_담론프로파일.html
  15_의원_담론_시간.html
  16_발언자_담론_시간.html
  17_담론간_관계네트워크.html
  18_키워드_공출현네트워크.html
  19_발언흐름_네트워크.html
  20_중심성_분석.html  # Degree / Weighted Degree / Betweenness / Closeness / Eigenvector
  21_에고네트워크.html
  22_시간축_네트워크.html
  23_지식그래프.html
  annotations/ (해당 열이 존재할 때만 생성)
  network_*.csv
  validation_*.csv
  logs/run.log
  대시보드.html
"""

from __future__ import annotations

import csv
import html as html_lib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------

INPUT_CSV = Path("hwp_record_assembly_speeches_이승만시기.csv")
OUTPUT_DIR = Path("visualization_rhee_advanced")
LOG_DIR = OUTPUT_DIR / "logs"
ANNOTATION_DIR = OUTPUT_DIR / "annotations"

START_DATE = pd.Timestamp("1948-06-01")
END_DATE = pd.Timestamp("1960-04-27")

TOP_GROUPS = 25
TOP_SPEAKERS = 30
TOP_NETWORK_NODES = 35
TOP_NETWORK_KEYWORDS = 25
TOP_KEYWORDS = 25
TOP_COMMUNITY_NODES = 60

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# CSV field-limit 오류 방지
try:
    csv.field_size_limit(2**31 - 1)
except OverflowError:
    csv.field_size_limit(1024 * 1024 * 1024)


# ------------------------------------------------------------
# 로깅
# ------------------------------------------------------------

LOG_PATH = LOG_DIR / "run.log"
LOG_LINES: list[str] = []


def log(message: str):
    print(message)
    LOG_LINES.append(message)


def skip(message: str):
    log(f"[SKIP] {message}")


def write_log():
    LOG_PATH.write_text("\n".join(LOG_LINES), encoding="utf-8")


# ------------------------------------------------------------
# 담론 구조
# ------------------------------------------------------------

FAMILY_TO_DISCOURSE = {
    "기술": "발전",
    "소득": "발전",
    "노동": "발전",
    "이민": "경계",
    "이주": "경계",
    "외국인": "경계",
}

FAMILIES = list(FAMILY_TO_DISCOURSE)
EXCLUSIVE_CLASSES = ["발전", "경계", "발전+경계", "미분류"]
METADATA_DIMENSIONS = ["정당", "지역", "성별", "당선횟수", "당선방법"]

KEYWORD_TO_FAMILY = {
    # 기술
    **{k: "기술" for k in [
        "산업", "공업", "생산", "기계", "기계화", "동력", "전력", "발전소",
        "개발", "건설", "부흥", "재건", "시설",
    ]},
    # 소득
    **{k: "소득" for k in [
        "물가", "생활비", "임금", "급여", "봉급", "물자", "생계", "경제",
        "재정", "예산", "국민소득", "세입",
    ]},
    # 노동
    **{k: "노동" for k in [
        "노무자", "근로자", "근로", "노동조합", "노조", "실업", "취업",
        "고용", "근로기준", "노동력", "노동자",
    ]},
    # 이민
    **{k: "이민" for k in [
        "교포", "동포", "이주민", "이민자", "해외동포", "재외국민", "귀환동포",
        "재일교포", "재중교포",
    ]},
    # 이주
    **{k: "이주" for k in [
        "이동", "인구이동", "유입", "유출", "정착", "피난민", "난민",
        "월남민", "실향민",
    ]},
    # 외국인
    **{k: "외국인" for k in [
        "외인", "외래", "화교", "왜인", "미군", "미인", "미국인", "양인",
        "이방인", "외지인",
    ]},
}


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------


def clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def split_semicolon(v) -> list[str]:
    s = clean(v)
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def mode_or_unknown(values) -> str:
    vals = [clean(x) for x in values if clean(x)]
    return Counter(vals).most_common(1)[0][0] if vals else "미상"


def safe_filename(text: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", str(text))


def capped_node_size(value, base=7.0, scale=1.35, cap=15.0):
    try:
        v = max(0.0, float(value))
    except (TypeError, ValueError):
        v = 0.0
    return min(cap, base + math.sqrt(v) * scale)


def top_label_nodes(nodes, score_map, n=7):
    ranked = sorted(nodes, key=lambda x: score_map.get(x, 0), reverse=True)
    return set(ranked[:min(n, len(ranked))])


def _node_label_annotations(pos, nodes, label_nodes, font_size=10, yshift=8,
                           bgcolor="rgba(255,255,255,0.92)", bordercolor="rgba(183,191,200,0.80)", label_map=None):
    """네트워크 노드명을 흰색 라벨 박스와 굵은 글씨로 표시한다."""
    anns = []
    for n in nodes:
        if n not in label_nodes or n not in pos:
            continue
        label = html_lib.escape(str((label_map or {}).get(n, n)))
        anns.append(dict(
            x=float(pos[n][0]), y=float(pos[n][1]),
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="center", yanchor="bottom",
            yshift=yshift,
            align="center",
            bgcolor=bgcolor,
            bordercolor=bordercolor,
            borderwidth=1,
            borderpad=2,
            font=dict(size=font_size, color="#253443", family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        ))
    return anns


def _speaker_label(graph, node):
    if graph is not None and node in graph:
        return clean(graph.nodes[node].get("speaker_label", node)) or str(node)
    return str(node)


def _append_layout_annotations(fig, extra_annotations):
    current = list(fig.layout.annotations) if fig.layout.annotations else []
    fig.update_layout(annotations=current + list(extra_annotations))


def figure_write(fig, path: Path, title: str | None = None):
    if go is None:
        skip(f"Plotly가 없어 {path.name} 생성 생략")
        return
    fig.update_layout(template="plotly_white")
    if title and not fig.layout.title.text:
        fig.update_layout(title=title)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def json_dump(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {INPUT_CSV.resolve()}")
    if go is None:
        raise RuntimeError("plotly가 설치되어 있지 않습니다. .venv에서 plotly를 설치하세요.")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    log(f"[LOAD] 원본 행 수: {len(df):,}")

    required = [
        "Title", "Date", "Speaker", "Content",
        "정당", "지역", "성별", "당선횟수", "당선방법",
        "발전담론", "발전계열", "발전키워드",
        "경계담론", "경계계열", "경계키워드", "담론분류",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].between(START_DATE, END_DATE, inclusive="both")].copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Year"] = df["Date"].dt.year.astype("Int64")
    df["MeetingID"] = (
        df["Date"].dt.strftime("%Y-%m-%d") + "||" + df["Title"].astype(str)
    )

    if "Seq" in df.columns:
        df["SeqNum"] = pd.to_numeric(df["Seq"], errors="coerce")
    else:
        df["SeqNum"] = np.nan
        skip("Seq 열이 없어 발언 순서 기반 네트워크는 비활성화됩니다.")

    df["SpeechID"] = (
        df["MeetingID"] + "||" +
        df["SeqNum"].fillna(df.groupby("MeetingID").cumcount() + 1).astype(str)
    )
    df["SpeakerKey"] = df["Speaker"].map(lambda x: clean(x).replace(" ", ""))

    df["ExclusiveClass"] = df.apply(
        lambda r: (
            "발전+경계" if clean(r["발전담론"]) == "예" and clean(r["경계담론"]) == "예"
            else "발전" if clean(r["발전담론"]) == "예"
            else "경계" if clean(r["경계담론"]) == "예"
            else "미분류"
        ),
        axis=1,
    )

    # 누락 검증
    missing_meta = {}
    for c in ["Speaker", *METADATA_DIMENSIONS]:
        missing_meta[c] = int((df[c].map(clean) == "").sum())
    pd.DataFrame([missing_meta]).to_csv(
        OUTPUT_DIR / "validation_메타데이터_결측.csv", encoding="utf-8-sig", index=False
    )

    log(f"[LOAD] 분석 행 수: {len(df):,}")
    log(f"[LOAD] 발언자 수: {df.loc[df['SpeakerKey'] != '', 'SpeakerKey'].nunique():,}")
    log(f"[LOAD] 회의 수: {df['MeetingID'].nunique():,}")
    return df


# ------------------------------------------------------------
# 기초 통계
# ------------------------------------------------------------


def monthly_stats(df):
    months = sorted(df["Month"].dropna().unique())
    out = pd.DataFrame(index=months)
    out["전체"] = df.groupby("Month").size()
    for cls in EXCLUSIVE_CLASSES:
        out[cls] = df[df["ExclusiveClass"] == cls].groupby("Month").size()
    out = out.fillna(0)
    out["발전_전체대비비중"] = np.where(out["전체"] > 0, out["발전"] / out["전체"] * 100, 0)
    out["경계_전체대비비중"] = np.where(out["전체"] > 0, out["경계"] / out["전체"] * 100, 0)
    out["발전+경계_전체대비비중"] = np.where(out["전체"] > 0, out["발전+경계"] / out["전체"] * 100, 0)
    out = out.reset_index(names="Month")
    return out


def family_monthly_stats(df):
    months = sorted(df["Month"].unique())
    counts = pd.DataFrame(0, index=FAMILIES, columns=months, dtype=float)
    for family in FAMILIES:
        col = "발전계열" if FAMILY_TO_DISCOURSE[family] == "발전" else "경계계열"
        mask = df[col].apply(lambda x: family in split_semicolon(x))
        tmp = df.loc[mask].groupby("Month").size()
        for m, v in tmp.items():
            counts.loc[family, m] = v
    total = df.groupby("Month").size()
    ratio = counts.copy()
    for m in months:
        ratio[m] = ratio[m] / max(total.get(m, 0), 1) * 1000
    return counts, ratio


def keyword_monthly_stats(df):
    months = sorted(df["Month"].unique())
    counter = defaultdict(lambda: Counter())
    for _, row in df.iterrows():
        kws = set(split_semicolon(row["발전키워드"]) + split_semicolon(row["경계키워드"]))
        for kw in kws:
            counter[kw][row["Month"]] += 1
    total_counts = Counter({k: sum(v.values()) for k, v in counter.items()})
    top_keywords = [k for k, _ in total_counts.most_common(TOP_NETWORK_KEYWORDS)]
    mat = pd.DataFrame(0, index=top_keywords, columns=months, dtype=float)
    for kw in top_keywords:
        for m, v in counter[kw].items():
            mat.loc[kw, m] = v
    return mat.sort_index()


def keyword_temporal_normalized_stats(keyword_month):
    """각 키워드의 전체 관측량을 100으로 두고 월별 구성비를 계산."""
    if keyword_month.empty:
        return keyword_month.copy()
    row_totals = keyword_month.sum(axis=1).replace(0, np.nan)
    return keyword_month.div(row_totals, axis=0).fillna(0) * 100


def keyword_stats(df):
    dev = Counter()
    boundary = Counter()
    for v in df["발전키워드"]:
        dev.update(set(split_semicolon(v)))
    for v in df["경계키워드"]:
        boundary.update(set(split_semicolon(v)))
    return dev, boundary


def group_stats(df, dimension):
    temp = df.copy()
    temp[dimension] = temp[dimension].map(clean).replace("", "미상")
    t = pd.crosstab(temp[dimension], temp["ExclusiveClass"])
    for cls in EXCLUSIVE_CLASSES:
        if cls not in t:
            t[cls] = 0
    t = t[EXCLUSIVE_CLASSES].copy()
    t["전체"] = t.sum(axis=1)
    t["관련발언"] = t["발전"] + t["경계"] + t["발전+경계"]
    t["관련비중"] = np.where(t["전체"] > 0, t["관련발언"] / t["전체"] * 100, 0)
    t["발전비중"] = np.where(t["전체"] > 0, t["발전"] / t["전체"] * 100, 0)
    t["경계비중"] = np.where(t["전체"] > 0, t["경계"] / t["전체"] * 100, 0)
    return t.sort_values(["관련비중", "전체"], ascending=False)


def speaker_stats(df):
    t = df[df["SpeakerKey"] != ""].copy()
    if t.empty:
        return pd.DataFrame()
    agg = t.groupby("SpeakerKey").agg(
        Speaker=("Speaker", lambda s: mode_or_unknown(s)),
        전체발언=("SpeechID", "size"),
        발전=("ExclusiveClass", lambda s: (s == "발전").sum()),
        경계=("ExclusiveClass", lambda s: (s == "경계").sum()),
        발전경계=("ExclusiveClass", lambda s: (s == "발전+경계").sum()),
    )
    agg["관련발언"] = agg["발전"] + agg["경계"] + agg["발전경계"]
    agg["관련비중"] = np.where(agg["전체발언"] > 0, agg["관련발언"] / agg["전체발언"] * 100, 0)
    for c in METADATA_DIMENSIONS:
        meta = t.groupby("SpeakerKey")[c].agg(mode_or_unknown)
        agg[c] = meta
    return agg.sort_values(["관련발언", "전체발언"], ascending=False)


# ------------------------------------------------------------
# 대시보드용 동적 인사이트
# ------------------------------------------------------------


def _pct(v):
    return f"{float(v):.1f}%"


def _short_date(v):
    try:
        return pd.Timestamp(v).strftime("%Y-%m")
    except Exception:
        return str(v)


def build_dashboard_insights(df, monthly, family_ratio, keyword_month, keyword_normalized, group_tables, speakers, cross_graph, keyword_graph, flow_graph, centrality_df, knowledge_graph):
    """대시보드 각 카드에 표시할 한 줄짜리 데이터 기반 요점을 만든다."""
    insights = {}

    if not monthly.empty:
        rel_cols = [c for c in ["발전", "경계", "발전+경계"] if c in monthly.columns]
        if rel_cols:
            tmp = monthly.assign(관련=monthly[rel_cols].sum(axis=1)).sort_values("관련", ascending=False).iloc[0]
            dominant = max(rel_cols, key=lambda c: tmp[c])
            insights["01_월별_담론량.html"] = f"{_short_date(tmp['Month'])}: {dominant} 발언 {int(tmp[dominant])}건으로 최대."
        if "발전_전체대비비중" in monthly.columns and "경계_전체대비비중" in monthly.columns:
            tmp = monthly.assign(관련비중=monthly["발전_전체대비비중"] + monthly["경계_전체대비비중"]).sort_values("관련비중", ascending=False).iloc[0]
            insights["02_월별_담론상대빈도.html"] = f"{_short_date(tmp['Month'])}: 관련 담론 비중 {_pct(tmp['관련비중'])}로 최대."

    if not family_ratio.empty:
        stacked = family_ratio.stack().sort_values(ascending=False)
        if not stacked.empty:
            (fam, month), val = stacked.index[0], float(stacked.iloc[0])
            insights["03_6개계열_시간Heatmap.html"] = f"{_short_date(month)}: {fam} 계열 밀도 {val:.2f}/1,000발언으로 최대."

    if not keyword_month.empty:
        top_cell = keyword_month.stack().sort_values(ascending=False)
        if not top_cell.empty:
            (kw, month), val = top_cell.index[0], float(top_cell.iloc[0])
            insights["04_키워드_시간Heatmap.html"] = f"{_short_date(month)}: ‘{kw}’ 관측 {int(val)}건으로 최대."
        totals = keyword_month.sum(axis=1).sort_values(ascending=False)
        if not totals.empty:
            kw = totals.index[0]
            peak_month = keyword_month.loc[kw].idxmax()
            insights["05_주요키워드_시계열.html"] = f"최다 키워드 ‘{kw}’, 최고점 {_short_date(peak_month)}."
            top3 = ", ".join([f"{k}({int(v)})" for k, v in totals.head(3).items()])
            insights["07_키워드_popularity.html"] = f"상위 키워드: {top3}."

    if not keyword_normalized.empty:
        concentration = keyword_normalized.max(axis=1).sort_values(ascending=False)
        if not concentration.empty:
            kw = concentration.index[0]
            month = keyword_normalized.loc[kw].idxmax()
            val = float(concentration.loc[kw])
            insights["06_키워드_시간구성비Heatmap.html"] = f"‘{kw}’의 {val:.1f}%가 {_short_date(month)}에 집중."

    dim_files = {
        "정당": "08_정당별_담론.html", "지역": "09_지역별_담론.html", "성별": "10_성별별_담론.html",
        "당선횟수": "11_당선횟수별_담론.html", "당선방법": "12_당선방법별_담론.html"
    }
    for dim, fn in dim_files.items():
        tab = group_tables.get(dim)
        if tab is not None and not tab.empty:
            cand = tab[tab["전체"] >= max(3, int(len(df) * 0.03))]
            if cand.empty:
                cand = tab
            row = cand.sort_values(["관련비중", "전체"], ascending=False).iloc[0]
            insights[fn] = f"{dim}: ‘{row.name}’ 관련 담론 비중 {_pct(row['관련비중'])}로 최대."

    if not speakers.empty:
        row = speakers.sort_values(["관련발언", "관련비중"], ascending=False).iloc[0]
        insights["13_발언자_프로파일.html"] = f"최다 관련 발언자: {row['Speaker']}({int(row['관련발언'])}건)."
        st = df[df["SpeakerKey"] == row.name]
        fam_counts = {}
        for fam in FAMILIES:
            col = "발전계열" if FAMILY_TO_DISCOURSE[fam] == "발전" else "경계계열"
            fam_counts[fam] = int(st[col].apply(lambda x: fam in split_semicolon(x)).sum())
        if fam_counts:
            fam = max(fam_counts, key=fam_counts.get)
            insights["14_의원별_담론프로파일.html"] = f"주요 의원 프로파일: ‘{fam}’ 계열이 최다."
        insights["15_의원_담론_시간.html"] = f"{row['Speaker']}: 관련 담론이 {st['Month'].nunique()}개 연월에서 관측."
        insights["16_발언자_전체발언_시간.html"] = f"관련 담론 최다 발언자: {row['Speaker']}."

    if cross_graph is not None and getattr(cross_graph, "number_of_edges", lambda: 0)() > 0:
        e = max(cross_graph.edges(data=True), key=lambda x: x[2].get("weight", 0))
        insights["17_담론간_관계네트워크.html"] = f"최강 담론 연결: ‘{e[0]}–{e[1]}’({int(e[2].get('weight',0))}회)."
    if keyword_graph is not None and getattr(keyword_graph, "number_of_edges", lambda: 0)() > 0:
        e = max(keyword_graph.edges(data=True), key=lambda x: x[2].get("weight", 0))
        insights["18_키워드_공출현네트워크.html"] = f"최강 키워드 연결: ‘{e[0]}–{e[1]}’({int(e[2].get('weight',0))}회)."
    if flow_graph is not None and getattr(flow_graph, "number_of_edges", lambda: 0)() > 0:
        e = max(flow_graph.edges(data=True), key=lambda x: x[2].get("weight", 0))
        insights["19_발언흐름_네트워크.html"] = f"최대 발언 인접성: ‘{e[0]}–{e[1]}’({int(e[2].get('weight',0))}회)."
    if centrality_df is not None and not centrality_df.empty:
        metric_top = []
        for col, label in [
            ("DegreeCentrality", "차수"),
            ("WeightedDegree", "가중차수"),
            ("WeightedBetweennessCentrality", "매개"),
            ("WeightedClosenessCentrality", "근접"),
            ("WeightedEigenvectorCentrality", "고유벡터"),
        ]:
            if col in centrality_df.columns:
                rr = centrality_df.sort_values(col, ascending=False).iloc[0]
                metric_top.append(f"{label}={rr.get('Speaker', rr['SpeakerKey'])}")
        insights["20_중심성_분석.html"] = " · ".join(metric_top) + "."
        center = centrality_df.sort_values("WeightedBetweennessCentrality", ascending=False).iloc[0]
        ego_parts=[]
        for col,label in [("DegreeCentrality","차수"),("BetweennessCentrality","매개"),("ClosenessCentrality","근접"),("EigenvectorCentrality","고유벡터")]:
            if col in centrality_df.columns:
                rr=centrality_df.sort_values(col,ascending=False).iloc[0]
                ego_parts.append(f"{label}={rr.get('Speaker', rr['SpeakerKey'])}")
        insights["21_에고네트워크.html"] = " · ".join(ego_parts) + "."
    if flow_graph is not None and getattr(flow_graph, "number_of_edges", lambda: 0)() > 0:
        ys = []
        for y, part in df.groupby("Year"):
            part = part.sort_values(["MeetingID", "SeqNum", "SpeechID"])
            n = 0
            for _, g in part.groupby("MeetingID"):
                ss = [x for x in g["SpeakerKey"] if x]
                n += sum(a != b for a, b in zip(ss, ss[1:]))
            ys.append((int(y), n))
        if ys:
            y, n = max(ys, key=lambda x: x[1])
            insights["22_시간축_네트워크.html"] = f"시간 연결 밀도 최대: {y}년."
    if knowledge_graph is not None:
        insights["23_지식그래프.html"] = f"발언 중심 그래프: {knowledge_graph.number_of_nodes():,}노드, {knowledge_graph.number_of_edges():,}관계."
    party_summary_path = OUTPUT_DIR / "통계_정당중심.csv"
    if party_summary_path.exists():
        try:
            pdf = pd.read_csv(party_summary_path)
            if not pdf.empty and "관련비중" in pdf.columns:
                rr = pdf.sort_values(["관련비중", "관련발언"], ascending=False).iloc[0]
                insights["25_정당중심.html"] = f"{rr['정당']}: 관련 담론 비중 {_pct(rr['관련비중'])}로 최대."
        except Exception:
            pass
    community_summary_path = OUTPUT_DIR / "network_community_summary.csv"
    if community_summary_path.exists():
        try:
            csum = pd.read_csv(community_summary_path)
            if not csum.empty:
                rr = csum.iloc[0]
                insights["24_커뮤니티_탐지.html"] = f"{int(rr['Communities'])}개 구조적 커뮤니티 · 최대 {int(rr['LargestSize'])}명."
        except Exception:
            pass

    # 섹션별 상단 요점(대시보드 SUMMARY)
    try:
        total_related = int((df["ExclusiveClass"].isin(["발전", "경계", "발전+경계"])).sum())
        rel_share = total_related / len(df) * 100 if len(df) else 0
        peak_rel = monthly.assign(관련=monthly[[c for c in ["발전","경계","발전+경계"] if c in monthly.columns]].sum(axis=1)).sort_values("관련", ascending=False).iloc[0] if not monthly.empty else None
        insights["SUMMARY_01"] = f"관련 담론 {total_related:,}건({_pct(rel_share)}) · {(_short_date(peak_rel['Month']) + '에 집중') if peak_rel is not None else '시계열 확인'}"
    except Exception:
        insights["SUMMARY_01"] = "발언량과 담론 비중의 시간 변화를 확인."
    try:
        kw_totals = keyword_month.sum(axis=1).sort_values(ascending=False) if not keyword_month.empty else pd.Series(dtype=float)
        kw = kw_totals.index[0] if len(kw_totals) else "주요 키워드"
        insights["SUMMARY_02"] = f"‘{kw}’가 가장 많이 관측 · 시간별 키워드 구성이 변화."
    except Exception:
        insights["SUMMARY_02"] = "핵심 어휘의 빈도와 시간적 변화를 확인."
    try:
        best = None
        for dim, tab in group_tables.items():
            if tab is not None and not tab.empty and "관련비중" in tab.columns:
                cand = tab[tab["전체"] >= max(3, int(len(df)*0.03))]
                if cand.empty: cand = tab
                r = cand.sort_values("관련비중", ascending=False).iloc[0]
                item=(float(r["관련비중"]), dim, str(r.name))
                if best is None or item[0] > best[0]: best=item
        insights["SUMMARY_03"] = f"{best[1]} 기준 ‘{best[2]}’의 관련 담론 비중이 가장 높음." if best else "정당·지역·성별·선거경력별 차이를 확인."
    except Exception:
        insights["SUMMARY_03"] = "정치사회적 행위자별 담론 분포를 확인."
    try:
        insights["SUMMARY_04"] = f"최다 관련 발언자: {speakers.sort_values('관련발언',ascending=False).iloc[0]['Speaker']}" if not speakers.empty else "주요 의원의 담론 프로파일과 활동 시기를 확인."
    except Exception:
        insights["SUMMARY_04"] = "주요 의원의 담론 프로파일과 활동 시기를 확인."
    try:
        pdf = pd.read_csv(OUTPUT_DIR/"통계_정당중심.csv")
        r=pdf.sort_values(["관련비중","관련발언"],ascending=False).iloc[0]
        insights["SUMMARY_05"] = f"‘{r['정당']}’의 관련 담론 비중이 정당 중 가장 높음."
        insights["25_정당중심.html"] = f"‘{r['정당']}’의 관련 담론 비중이 가장 높음."
        # 정당 프로파일은 정당 내부 구성에서 가장 큰 담론계열을 요점으로 표시
        try:
            prof = pd.read_csv(OUTPUT_DIR/"통계_정당별_6개담론프로파일.csv", index_col=0)
            vals = prof.stack().sort_values(ascending=False)
            if not vals.empty:
                (party, fam), val = vals.index[0], float(vals.iloc[0])
                insights["26_정당별_담론프로파일.html"] = f"‘{party}’에서 ‘{fam}’ 계열 비중 {val:.1f}%로 최대."
        except Exception:
            pass
        try:
            pt = pd.read_csv(OUTPUT_DIR/"통계_정당_담론_시간.csv")
            if not pt.empty:
                rr = pt.sort_values("Count", ascending=False).iloc[0]
                insights["27_정당_담론_시간.html"] = f"{_short_date(rr['Month'])}: ‘{rr['정당']}’의 ‘{rr['Family']}’ 발언 {int(rr['Count'])}건으로 최대."
        except Exception:
            pass
        try:
            ptime = df[df["정당"].map(clean)!=""].groupby(["Month","정당"]).size().reset_index(name="Count")
            if not ptime.empty:
                rr = ptime.sort_values("Count", ascending=False).iloc[0]
                insights["28_정당_전체발언_시간.html"] = f"{_short_date(rr['Month'])}: ‘{rr['정당']}’ 전체 발언 {int(rr['Count'])}건으로 최대."
        except Exception:
            pass
    except Exception:
        insights["SUMMARY_05"] = "정당별 담론 프로파일과 시간적 활동 변화를 확인."
    try:
        parts=[]
        if cross_graph is not None and cross_graph.number_of_edges():
            e=max(cross_graph.edges(data=True),key=lambda x:x[2].get('weight',0)); parts.append(f"담론 연결 {e[0]}–{e[1]}")
        if centrality_df is not None and not centrality_df.empty:
            c=centrality_df.sort_values("BetweennessCentrality",ascending=False).iloc[0]; parts.append(f"매개 중심 의원 {c.get('Speaker', c['SpeakerKey'])}")
        insights["SUMMARY_06"] = " · ".join(parts) if parts else "담론·키워드·발언흐름의 연결 구조를 확인."
    except Exception:
        insights["SUMMARY_06"] = "담론·키워드·발언흐름의 연결 구조를 확인."

    return insights


# ------------------------------------------------------------
# Plotly 시각화
# ------------------------------------------------------------


def plot_monthly(monthly):
    fig = go.Figure()
    for col, name, dash in [("발전", "발전", None), ("경계", "경계", None), ("발전+경계", "발전+경계", "dash")]:
        line = {} if not dash else {"dash": dash}
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly[col], mode="lines", name=name, line=line))
    fig.update_layout(title="월별 발전·경계 담론 발언량", xaxis_title="연월", yaxis_title="발언 수", hovermode="x unified")
    figure_write(fig, OUTPUT_DIR / "01_월별_담론량.html")

    fig2 = go.Figure()
    for col, name in [("발전_전체대비비중", "발전"), ("경계_전체대비비중", "경계"), ("발전+경계_전체대비비중", "발전+경계")]:
        fig2.add_trace(go.Scatter(x=monthly["Month"], y=monthly[col], mode="lines", name=name))
    fig2.update_layout(title="월별 발전·경계 담론의 상대빈도", xaxis_title="연월", yaxis_title="전체 발언 대비 비중 (%)", hovermode="x unified")
    figure_write(fig2, OUTPUT_DIR / "02_월별_담론상대빈도.html")


def plot_family_heatmap(family_ratio):
    fig = go.Figure(go.Heatmap(
        x=list(family_ratio.columns),
        y=list(family_ratio.index),
        z=family_ratio.values,
        hovertemplate="%{y}<br>%{x}<br>1,000발언당 %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="6개 담론 계열의 시간적 분포", xaxis_title="연월", yaxis_title="담론 계열")
    figure_write(fig, OUTPUT_DIR / "03_6개계열_시간Heatmap.html")


def plot_keyword_heatmap(keyword_month):
    if keyword_month.empty:
        skip("키워드 데이터가 없어 키워드×시간 heatmap 생략")
        return
    fig = go.Figure(go.Heatmap(
        x=list(keyword_month.columns),
        y=list(keyword_month.index),
        z=keyword_month.values,
        hovertemplate="%{y}<br>%{x}<br>발언 수: %{z}<extra></extra>",
    ))
    fig.update_layout(title="주요 키워드 × 시간 Heatmap", xaxis_title="연월", yaxis_title="키워드")
    figure_write(fig, OUTPUT_DIR / "04_키워드_시간Heatmap.html")


def plot_keyword_trends(keyword_month):
    if keyword_month.empty:
        return
    top = keyword_month.sum(axis=1).sort_values(ascending=False).head(12).index
    fig = go.Figure()
    for kw in top:
        fig.add_trace(go.Scatter(x=list(keyword_month.columns), y=keyword_month.loc[kw].values, mode="lines", name=kw))
    fig.update_layout(title="주요 키워드의 시간적 변화", xaxis_title="연월", yaxis_title="해당 키워드가 포함된 발언 수", hovermode="x unified")
    figure_write(fig, OUTPUT_DIR / "05_주요키워드_시계열.html")


def plot_keyword_temporal_normalized(keyword_month, normalized):
    if normalized.empty:
        skip("키워드 시간 정규화 데이터가 없어 키워드별 시간구성비 생략")
        return
    fig = go.Figure(go.Heatmap(
        x=list(normalized.columns),
        y=list(normalized.index),
        z=normalized.values,
        hovertemplate="%{y}<br>%{x}<br>해당 키워드 전체 관측량 대비 %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="키워드별 시간적 구성비(각 키워드=100%)",
        xaxis_title="연월", yaxis_title="키워드"
    )
    figure_write(fig, OUTPUT_DIR / "06_키워드_시간구성비Heatmap.html")


def plot_keyword_popularity(keyword_month):
    if keyword_month.empty:
        skip("키워드 빈도 데이터가 없어 키워드 popularity 생략")
        return
    counts = keyword_month.sum(axis=1).sort_values(ascending=False).head(TOP_KEYWORDS).sort_values()
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index.astype(str), orientation="h"))
    fig.update_layout(title="주요 키워드의 전체 관측 빈도", xaxis_title="키워드가 포함된 발언 수", yaxis_title="키워드", height=760)
    figure_write(fig, OUTPUT_DIR / "07_키워드_popularity.html")


def plot_group(table, dimension, idx):
    plot = table.head(TOP_GROUPS).sort_values("관련비중")
    fig = go.Figure()
    for cls in EXCLUSIVE_CLASSES:
        fig.add_trace(go.Bar(y=plot.index.astype(str), x=plot[cls], name=cls, orientation="h"))
    fig.update_layout(
        title=f"{dimension}별 담론 분포",
        barmode="stack",
        xaxis_title="발언 수",
        yaxis_title=dimension,
        legend=dict(orientation="h"),
        height=760,
    )
    figure_write(fig, OUTPUT_DIR / f"{idx:02d}_{safe_filename(dimension)}별_담론.html")


def plot_speaker_profile(speakers):
    if speakers.empty:
        skip("발언자가 없어 발언자 프로파일 생략")
        return
    s = speakers.head(TOP_SPEAKERS)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["전체발언"], y=s["관련비중"], mode="markers+text",
        text=[x if i < 8 else "" for i, x in enumerate(s["Speaker"])], textposition="top center",
        customdata=np.column_stack([s["정당"], s["지역"], s["성별"], s["당선횟수"], s["당선방법"]]),
        hovertemplate=(
            "%{text}<br>전체 발언=%{x}<br>관련 비중=%{y:.2f}%"
            "<br>정당=%{customdata[0]}<br>지역=%{customdata[1]}"
            "<br>성별=%{customdata[2]}<br>당선횟수=%{customdata[3]}"
            "<br>당선방법=%{customdata[4]}<extra></extra>"
        ),
        marker=dict(size=np.maximum(9, np.sqrt(s["관련발언"].values) * 4), opacity=0.75),
    ))
    fig.update_layout(title="발언자 프로파일: 총발언량 × 발전·경계 담론 비중", xaxis_title="전체 발언 수", yaxis_title="관련 담론 비중 (%)")
    figure_write(fig, OUTPUT_DIR / "13_발언자_프로파일.html")


def plot_speaker_time(df, speakers):
    if speakers.empty:
        return
    top = list(speakers.head(15).index)
    tmp = df[df["SpeakerKey"].isin(top)].copy()
    piv = pd.crosstab(tmp["Month"], tmp["SpeakerKey"])
    fig = go.Figure()
    for sp in top:
        if sp in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[sp], mode="lines", name=speakers.loc[sp, "Speaker"]))
    fig.update_layout(title="주요 발언자의 시간적 발언량", xaxis_title="연월", yaxis_title="발언 수", hovermode="x unified")
    figure_write(fig, OUTPUT_DIR / "16_발언자_전체발언_시간.html")


def plot_speaker_discourse_profile(df, speakers):
    if speakers.empty:
        skip("발언자가 없어 의원별 세부 담론 프로파일 생략")
        return
    top = list(speakers.head(TOP_SPEAKERS).index)
    rows = []
    for sp in top:
        sub = df[df["SpeakerKey"] == sp]
        total = len(sub)
        for fam in FAMILIES:
            col = "발전계열" if FAMILY_TO_DISCOURSE[fam] == "발전" else "경계계열"
            n = int(sub[col].apply(lambda x: fam in split_semicolon(x)).sum())
            rows.append({"SpeakerKey": sp, "Speaker": speakers.loc[sp, "Speaker"], "Family": fam, "Count": n, "Rate": n / total * 100 if total else 0})
    long = pd.DataFrame(rows)
    long.to_csv(OUTPUT_DIR / "통계_의원별_6개담론프로파일.csv", encoding="utf-8-sig", index=False)
    pivot = long.pivot(index="Speaker", columns="Family", values="Rate").fillna(0)
    pivot = pivot.loc[[speakers.loc[sp, "Speaker"] for sp in top if speakers.loc[sp, "Speaker"] in pivot.index]]
    fig = go.Figure(go.Heatmap(
        x=list(pivot.columns), y=list(pivot.index), z=pivot.values,
        hovertemplate="%{y}<br>%{x}<br>관련 발언 비중=%{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="주요 의원별 6개 담론계열 프로파일", xaxis_title="담론계열", yaxis_title="의원")
    figure_write(fig, OUTPUT_DIR / "14_의원별_담론프로파일.html")


def plot_speaker_discourse_time(df, speakers):
    """주요 의원 × 6개 담론계열 × 시간의 2×3 Heatmap."""
    if speakers.empty:
        skip("발언자가 없어 의원×담론×시간 분석 생략")
        return

    top = list(speakers.head(min(12, len(speakers))).index)
    sub = df[df["SpeakerKey"].isin(top)].copy()
    if sub.empty:
        skip("주요 의원의 발언 데이터가 없어 의원×담론×시간 분석 생략")
        return

    months = sorted(sub["Month"].dropna().astype(str).unique())
    if not months:
        skip("유효한 연월이 없어 의원×담론×시간 분석 생략")
        return

    names = [speakers.loc[sp, "Speaker"] for sp in top]
    fig = make_subplots(rows=2, cols=3, subplot_titles=FAMILIES,
                        horizontal_spacing=0.08, vertical_spacing=0.14)
    export_rows = []

    for i, fam in enumerate(FAMILIES):
        colname = "발전계열" if FAMILY_TO_DISCOURSE[fam] == "발전" else "경계계열"
        mask = sub[colname].apply(lambda x: fam in split_semicolon(x))
        mat = pd.crosstab(sub.loc[mask, "SpeakerKey"], sub.loc[mask, "Month"]) \
                .reindex(index=top, columns=months, fill_value=0)
        r, c = divmod(i, 3)
        fig.add_trace(
            go.Heatmap(
                z=mat.values, x=months, y=names,
                showscale=(i == 5),
                colorbar=dict(title="발언 수", len=0.45) if i == 5 else None,
                hovertemplate="의원=%{y}<br>연월=%{x}<br>발언 수=%{z}<extra></extra>",
            ), row=r+1, col=c+1
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8), row=r+1, col=c+1)
        fig.update_yaxes(autorange="reversed", showticklabels=(c == 0), tickfont=dict(size=8), automargin=False, row=r+1, col=c+1)
        for sp in top:
            for month in months:
                export_rows.append({
                    "SpeakerKey": sp,
                    "Speaker": speakers.loc[sp, "Speaker"],
                    "Family": fam,
                    "Month": month,
                    "Count": int(mat.loc[sp, month]),
                })

    fig.update_layout(
        title="의원 × 담론계열 × 시간", height=820,
        margin=dict(l=125, r=45, t=70, b=60),
        annotations=list(fig.layout.annotations) + [dict(
            text="왼쪽 열의 의원명이 6개 히트맵에 공통 적용됩니다.",
            x=0, y=-0.065, xref="paper", yref="paper", showarrow=False,
            align="left", font=dict(size=9),
        )],
    )
    figure_write(fig, OUTPUT_DIR / "15_의원_담론_시간.html")
    pd.DataFrame(export_rows).to_csv(
        OUTPUT_DIR / "통계_의원_담론_시간.csv", encoding="utf-8-sig", index=False
    )

def _party_base_data(df):
    sub = df[(df["정당"].map(clean) != "") & (df["SpeakerKey"].map(clean) != "")].copy()
    if sub.empty:
        return sub, pd.DataFrame()
    party_order = sub.groupby("정당").size().sort_values(ascending=False).head(12)
    parties = list(party_order.index)
    rows = []
    for party in parties:
        st = sub[sub["정당"] == party]
        total = len(st)
        related = int((st["ExclusiveClass"].isin(["발전", "경계", "발전+경계"])).sum())
        row = {"정당": party, "전체발언": total, "관련발언": related,
               "관련비중": (related / total * 100 if total else 0)}
        for fam in FAMILIES:
            colname = "발전계열" if FAMILY_TO_DISCOURSE[fam] == "발전" else "경계계열"
            row[fam] = int(st[colname].apply(lambda x, f=fam: f in split_semicolon(x)).sum())
        rows.append(row)
    return sub, pd.DataFrame(rows).set_index("정당")


def party_center_analysis(df):
    """정당 중심 4종 분석을 생성한다.

    25: 정당-주요 의원 네트워크
    26: 정당별 6개 담론 프로파일
    27: 정당 × 담론 × 시간
    28: 정당 × 전체발언 × 시간
    """
    if go is None or nx is None:
        skip("networkx/plotly가 없어 정당 중심 분석 생략")
        return
    if "정당" not in df.columns or "SpeakerKey" not in df.columns:
        skip("정당 또는 발언자 열이 없어 정당 중심 분석 생략")
        return

    sub, pdf = _party_base_data(df)
    if sub.empty or pdf.empty:
        skip("유효한 정당 자료가 없어 정당 중심 분석 생략")
        return

    parties = list(pdf.index)
    speaker_labels = sub.groupby("SpeakerKey")["Speaker"].agg(mode_or_unknown).to_dict()

    # 25. 정당-주요의원 네트워크
    G = nx.Graph()
    for party in parties:
        pn = f"정당:{party}"
        G.add_node(pn, node_type="party", label=party, weight=float(pdf.loc[party, "전체발언"]))
        top_s = sub[sub["정당"] == party].groupby("SpeakerKey").size().sort_values(ascending=False).head(6)
        for sp, cnt in top_s.items():
            sn = f"의원:{sp}"
            G.add_node(sn, node_type="speaker", label=speaker_labels.get(sp, sp), speaker_key=sp, weight=float(cnt))
            G.add_edge(pn, sn, weight=float(cnt))

    pos = _safe_spring_layout(G, weight="weight", k=1.0, iterations=100)
    ex, ey, et = [], [], []
    for u, v, d in G.edges(data=True):
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        h = f"{G.nodes[u]['label']} ↔ {G.nodes[v]['label']}<br>연결 발언={int(d['weight'])}건<extra></extra>"
        et += [h, h, None]
    edge = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1), hoverinfo="text", text=et, showlegend=False)
    pnodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "party"]
    snodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "speaker"]
    ps = {n: G.nodes[n]["weight"] for n in pnodes}; ss = {n: G.nodes[n]["weight"] for n in snodes}
    ptr = go.Scatter(x=[pos[n][0] for n in pnodes], y=[pos[n][1] for n in pnodes], mode="markers",
                     marker=dict(size=_normalized_node_sizes(ps, pnodes, 13, 30), opacity=.9, line=dict(width=1.3, color="#ffffff")), name="정당",
                     hovertemplate=[f"{G.nodes[n]['label']}<br>전체 발언={int(G.nodes[n]['weight'])}건<extra></extra>" for n in pnodes])
    sl = top_label_nodes(snodes, ss, n=min(10, len(snodes)))
    strace = go.Scatter(x=[pos[n][0] for n in snodes], y=[pos[n][1] for n in snodes], mode="markers",
                        marker=dict(size=_normalized_node_sizes(ss, snodes, 7, 18), opacity=.78, line=dict(width=1.0, color="#ffffff")), name="주요 의원",
                        hovertemplate=[f"{G.nodes[n]['label']}<br>정당={G.nodes[next(iter(G[n]))]['label'] if G[n] else '미상'}<br>발언={int(G.nodes[n]['weight'])}건<extra></extra>" for n in snodes])
    fig = go.Figure([edge, ptr, strace])
    party_label_nodes = set(pnodes) | set(sl)
    party_label_map = {n: G.nodes[n]["label"] for n in G.nodes}
    fig.update_layout(title="정당 중심: 정당–주요 의원", height=680, margin=dict(l=25, r=25, t=60, b=60),
                      annotations=_node_label_annotations(pos, pnodes + snodes, party_label_nodes, font_size=10, yshift=8, label_map=party_label_map),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), legend=dict(orientation="h", y=-.03))
    figure_write(fig, OUTPUT_DIR / "25_정당중심.html")

    # 26. 정당별 담론 프로파일 (정당 내 전체 발언 대비 비중)
    profile = pdf[FAMILIES].div(pdf["전체발언"].replace(0, np.nan), axis=0).fillna(0) * 100
    profile.to_csv(OUTPUT_DIR / "통계_정당별_6개담론프로파일.csv", encoding="utf-8-sig")
    fig2 = go.Figure(go.Heatmap(z=profile.values, x=FAMILIES, y=profile.index.astype(str),
                                hovertemplate="정당=%{y}<br>담론=%{x}<br>전체 발언 대비 %{z:.2f}%<extra></extra>"))
    fig2.update_layout(title="정당별 6개 담론계열 프로파일", height=max(520, 36*len(profile)+120),
                       margin=dict(l=105, r=30, t=60, b=55), xaxis_title="담론계열", yaxis_title="정당")
    figure_write(fig2, OUTPUT_DIR / "26_정당별_담론프로파일.html")

    # 27. 정당 × 담론 × 시간 — 2×3 heatmap
    top_sub = sub[sub["정당"].isin(parties)].copy()
    months = sorted(top_sub["Month"].dropna().astype(str).unique())
    if months:
        fig3 = make_subplots(rows=2, cols=3, subplot_titles=FAMILIES, horizontal_spacing=0.07, vertical_spacing=0.13)
        export_rows = []
        for i, fam in enumerate(FAMILIES):
            colname = "발전계열" if FAMILY_TO_DISCOURSE[fam] == "발전" else "경계계열"
            mask = top_sub[colname].apply(lambda x, f=fam: f in split_semicolon(x))
            mat = pd.crosstab(top_sub.loc[mask, "정당"], top_sub.loc[mask, "Month"]).reindex(index=parties, columns=months, fill_value=0)
            r, c = divmod(i, 3)
            fig3.add_trace(go.Heatmap(z=mat.values, x=months, y=parties, showscale=(i == 5),
                                      colorbar=dict(title="발언 수", len=.45) if i == 5 else None,
                                      hovertemplate="정당=%{y}<br>연월=%{x}<br>발언 수=%{z}<extra></extra>"), row=r+1, col=c+1)
            fig3.update_xaxes(tickangle=45, tickfont=dict(size=8), row=r+1, col=c+1)
            fig3.update_yaxes(showticklabels=(c == 0), tickfont=dict(size=8), autorange="reversed", row=r+1, col=c+1)
            for party in parties:
                for month in months:
                    export_rows.append({"정당": party, "Family": fam, "Month": month, "Count": int(mat.loc[party, month])})
        fig3.update_layout(title="정당 × 담론계열 × 시간", height=max(860, 34*len(parties)+360),
                           margin=dict(l=105, r=35, t=70, b=65))
        figure_write(fig3, OUTPUT_DIR / "27_정당_담론_시간.html")
        pd.DataFrame(export_rows).to_csv(OUTPUT_DIR / "통계_정당_담론_시간.csv", encoding="utf-8-sig", index=False)
    else:
        skip("유효한 연월이 없어 정당×담론×시간 분석 생략")

    # 28. 정당 × 전체발언량 × 시간
    piv = pd.crosstab(top_sub["Month"], top_sub["정당"]).reindex(columns=parties, fill_value=0)
    fig4 = go.Figure()
    for party in parties:
        fig4.add_trace(go.Scatter(x=piv.index, y=piv[party], mode="lines", name=str(party)))
    fig4.update_layout(title="주요 정당의 전체 발언량 × 시간", xaxis_title="연월", yaxis_title="전체 발언 수", hovermode="x unified")
    figure_write(fig4, OUTPUT_DIR / "28_정당_전체발언_시간.html")

    pdf_out = pdf.reset_index()
    pdf_out["관련발언"] = pdf_out["관련발언"].astype(int)
    pdf_out.to_csv(OUTPUT_DIR / "통계_정당중심.csv", encoding="utf-8-sig", index=False)
    log(f"[OK] 정당 중심 분석: {len(parties)}개 정당 · 4종 시각화")
    return pdf_out


def build_discourse_cross_network(df):
    if nx is None:
        skip("networkx가 없어 담론 간 관계 네트워크 생략")
        return None
    G = nx.Graph()
    for fam in FAMILIES:
        G.add_node(fam)
    counter = Counter()
    for _, row in df.iterrows():
        fams = []
        for col in ["발전계열", "경계계열"]:
            fams.extend([x for x in split_semicolon(row[col]) if x in FAMILIES])
        for a, b in itertools.combinations(sorted(set(fams)), 2):
            counter[(a, b)] += 1
    for (a, b), w in counter.items():
        if w > 0:
            G.add_edge(a, b, weight=w)
    if G.number_of_edges() == 0:
        skip("서로 다른 담론계열이 같은 발언에서 충분히 결합하지 않아 담론 간 관계 네트워크 생략")
        return G
    pos = nx.spring_layout(G, seed=42, weight="weight")
    ex, ey = [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none", showlegend=False)
    node_scores = {n: sum(d.get("weight", 0) for _, _, d in G.edges(n, data=True)) for n in G.nodes}
    label_nodes = top_label_nodes(list(G.nodes), node_scores, n=6)
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes], y=[pos[n][1] for n in G.nodes],
        mode="markers",
        textfont=dict(size=10, color="#253443", family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        marker=dict(size=[capped_node_size(node_scores[n], base=8, scale=1.0, cap=14) for n in G.nodes], opacity=0.82),
        hovertemplate=[f"{n}<br>담론 교차연결={node_scores[n]}<extra></extra>" for n in G.nodes],
        showlegend=False,
    )
    edge_ann = [dict(
        x=(pos[u][0]+pos[v][0])/2, y=(pos[u][1]+pos[v][1])/2,
        text=str(d["weight"]), showarrow=False, font=dict(size=11)
    ) for u, v, d in G.edges(data=True)]
    fig = go.Figure([edge_trace, node_trace])
    label_anns = _node_label_annotations(pos, list(G.nodes), label_nodes, font_size=10, yshift=8)
    fig.update_layout(title="6개 담론계열의 발언 내 교차연결", xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), annotations=edge_ann + label_anns)
    figure_write(fig, OUTPUT_DIR / "17_담론간_관계네트워크.html")
    pd.DataFrame([{"FamilyA":a,"FamilyB":b,"Weight":w} for (a,b),w in counter.items()]).sort_values("Weight", ascending=False).to_csv(OUTPUT_DIR / "network_discourse_cross_edges.csv", encoding="utf-8-sig", index=False)
    return G


# ------------------------------------------------------------
# 네트워크: keyword co-occurrence
# ------------------------------------------------------------


def build_keyword_network(df):
    if nx is None:
        skip("networkx가 없어 키워드 공출현 네트워크 생략")
        return None, None

    G = nx.Graph()
    all_kw_counter = Counter()
    edge_counter = Counter()

    for _, row in df.iterrows():
        kws = sorted(set(split_semicolon(row["발전키워드"]) + split_semicolon(row["경계키워드"])))
        kws = [k for k in kws if k in KEYWORD_TO_FAMILY]
        for kw in kws:
            all_kw_counter[kw] += 1
        for a, b in itertools.combinations(kws, 2):
            edge_counter[(a, b)] += 1

    top_nodes = {k for k, _ in all_kw_counter.most_common(TOP_NETWORK_KEYWORDS)}
    for kw in top_nodes:
        G.add_node(kw, family=KEYWORD_TO_FAMILY.get(kw, "기타"), count=all_kw_counter[kw])
    for (a, b), w in edge_counter.items():
        if a in top_nodes and b in top_nodes and w >= 2:
            G.add_edge(a, b, weight=w)

    if len(G.nodes) < 2:
        skip("키워드 공출현 관계가 충분하지 않아 네트워크 생략")
        return G, edge_counter

    pos = nx.spring_layout(G, seed=42, weight="weight")
    edge_x, edge_y = [], []
    for a, b in G.edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", showlegend=False)
    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    node_text = [f"{n}<br>공출현 발언={G.nodes[n]['count']}" for n in G.nodes]
    label_nodes = top_label_nodes(list(G.nodes), {n: G.nodes[n]["count"] for n in G.nodes}, n=7)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=[capped_node_size(G.nodes[n]["count"], base=7, scale=1.25, cap=15) for n in G.nodes], opacity=0.82),
        hovertemplate="%{hovertext}<extra></extra>", hovertext=node_text, showlegend=False,
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(title="키워드 공출현 네트워크", xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), annotations=_node_label_annotations(pos, list(G.nodes), label_nodes, font_size=10, yshift=8), showlegend=False)
    figure_write(fig, OUTPUT_DIR / "18_키워드_공출현네트워크.html")

    pd.DataFrame([
        {"KeywordA": a, "KeywordB": b, "Weight": w}
        for (a, b), w in edge_counter.items() if a in top_nodes and b in top_nodes and w >= 2
    ]).sort_values("Weight", ascending=False).to_csv(
        OUTPUT_DIR / "network_keyword_cooccurrence_edges.csv", encoding="utf-8-sig", index=False
    )
    return G, edge_counter


# ------------------------------------------------------------
# 네트워크: 발언 흐름
# ------------------------------------------------------------


def build_speech_flow_network(df):
    if nx is None:
        skip("networkx가 없어 발언흐름 네트워크 생략")
        return None, pd.DataFrame()
    if "SeqNum" not in df.columns or df["SeqNum"].notna().sum() == 0:
        skip("유효한 Seq가 없어 발언흐름 네트워크 생략")
        return None, pd.DataFrame()

    ordered = df[df["SpeakerKey"] != ""].copy()
    ordered = ordered.sort_values(["MeetingID", "SeqNum", "SpeechID"])

    edge_counter = Counter()
    edge_examples = defaultdict(list)
    for meeting, part in ordered.groupby("MeetingID", sort=False):
        rows = part[["SpeakerKey", "SpeechID", "SeqNum", "Date", "Title"]].to_dict("records")
        for a, b in zip(rows, rows[1:]):
            sa, sb = a["SpeakerKey"], b["SpeakerKey"]
            if not sa or not sb or sa == sb:
                continue
            u, v = sorted([sa, sb])
            edge_counter[(u, v)] += 1
            if len(edge_examples[(u, v)]) < 3:
                edge_examples[(u, v)].append({
                    "MeetingID": meeting,
                    "Date": str(a["Date"].date()),
                    "SeqA": a["SeqNum"],
                    "SeqB": b["SeqNum"],
                    "SpeechA": a["SpeechID"],
                    "SpeechB": b["SpeechID"],
                })

    G = nx.Graph()
    speaker_labels = ordered.groupby("SpeakerKey")["Speaker"].agg(mode_or_unknown).to_dict()
    for sp, n in ordered["SpeakerKey"].value_counts().items():
        G.add_node(sp, speech_count=int(n), speaker_label=speaker_labels.get(sp, sp))
    for (u, v), w in edge_counter.items():
        G.add_edge(u, v, weight=int(w))

    # top-node subgraph for readability
    top_nodes = {n for n, _ in Counter(ordered["SpeakerKey"]).most_common(TOP_NETWORK_NODES)}
    H = G.subgraph(top_nodes).copy()

    if len(H.nodes) < 2:
        skip("발언흐름 네트워크의 주요 발언자가 2명 미만")
        return G, pd.DataFrame()

    pos = nx.spring_layout(H, seed=42, weight="weight")
    edge_x, edge_y = [], []
    hover_edges = []
    for u, v, d in H.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        hover_edges.append(d.get("weight", 1))

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none", showlegend=False)
    label_nodes = top_label_nodes(list(H.nodes), {n: H.nodes[n]["speech_count"] for n in H.nodes}, n=7)
    node_trace = go.Scatter(
        x=[pos[n][0] for n in H.nodes],
        y=[pos[n][1] for n in H.nodes],
        mode="markers",
        textfont=dict(size=10, color="#253443", family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        marker=dict(size=[capped_node_size(H.nodes[n]["speech_count"], base=7, scale=1.25, cap=15) for n in H.nodes], opacity=0.82, line=dict(width=1.1, color="#ffffff")),
        hovertemplate=[f"{_speaker_label(H, n)}<br>발언 수={H.nodes[n]['speech_count']}<br>식별자={n}<extra></extra>" for n in H.nodes],
        showlegend=False,
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="발언흐름 네트워크 (연속 발언 인접성)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        annotations=_node_label_annotations(pos, list(H.nodes), label_nodes, font_size=10, yshift=8) + [dict(
            text="※ 이것은 직접 상호작용 네트워크가 아니라 같은 회의에서 연속한 발언 사이의 구조적 인접성입니다.",
            x=0, y=-0.08, xref="paper", yref="paper", showarrow=False,
        )],
    )
    figure_write(fig, OUTPUT_DIR / "19_발언흐름_네트워크.html")

    rows = []
    for (u, v), w in edge_counter.items():
        rows.append({"SpeakerA": u, "SpeakerB": v, "Weight": w, "Examples": json.dumps(edge_examples[(u, v)], ensure_ascii=False)})
    edge_df = pd.DataFrame(rows).sort_values("Weight", ascending=False)
    edge_df.to_csv(OUTPUT_DIR / "network_speech_flow_edges.csv", encoding="utf-8-sig", index=False)
    # downstream 분석(중심성/ego/timeline)은 전체 flow graph를 사용하고,
    # 시각화만 가독성을 위해 상위 노드 H로 제한한다.
    return G, edge_df


# ------------------------------------------------------------
# Centrality
# ------------------------------------------------------------


def _inverse_weight_graph(G):
    """빈도(weight)를 거리(distance=1/weight)로 변환한 복제 그래프."""
    H = G.copy()
    for u, v, d in H.edges(data=True):
        try:
            w = float(d.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        if w <= 0:
            w = 1.0
        H[u][v]["distance"] = 1.0 / w
    return H


def _safe_spring_layout(G, weight=None, seed=42, k=0.8, iterations=100):
    """SciPy가 없어도 네트워크 시각화가 중단되지 않도록 layout fallback 제공."""
    try:
        return nx.spring_layout(G, seed=seed, weight=weight, iterations=iterations, k=k)
    except ImportError:
        log("[WARN] SciPy 없음 → spring_layout 대신 circular_layout 사용")
        return nx.circular_layout(G)
    except Exception as exc:
        log(f"[WARN] spring_layout 실패({type(exc).__name__}) → circular_layout 사용")
        return nx.circular_layout(G)


def _normalized_node_sizes(score_map, nodes, min_size=8, max_size=30):
    vals = np.array([float(score_map.get(n, 0.0) or 0.0) for n in nodes], dtype=float)
    if len(vals) == 0:
        return []
    lo = float(np.nanmin(vals)); hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [float((min_size + max_size) / 2)] * len(vals)
    z = np.clip((vals - lo) / (hi - lo), 0, 1)
    z = np.sqrt(z)
    return list(min_size + z * (max_size - min_size))


def centrality_analysis(G):
    """발언흐름 네트워크의 중심성을 계산하고, 가중치 ON/OFF를 한 화면에서 비교한다.

    가중치 정의
    ----------
    edge weight = 같은 회의에서 두 의원이 연속 발언자로 등장한 횟수(전체 기간 합산).

    가중 모드
    ---------
    - Weighted Degree / Strength: 인접 발언 횟수의 합
    - Weighted Betweenness / Closeness: shortest-path용 distance = 1 / weight
    - Weighted Eigenvector: weight를 연결 강도로 사용

    주의
    ----
    - Degree는 연결된 상대 의원 수이며 weight를 사용하지 않는다.
    - 중심성 순위는 전체 flow graph 기준이다.
    - 네트워크 미리보기는 가독성을 위해 발언량 상위 TOP_NETWORK_NODES만 표시한다.
    """
    if G is None or nx is None or len(G.nodes) < 2:
        skip("중심성 분석에 사용할 네트워크가 없습니다.")
        return pd.DataFrame()

    n = len(G.nodes)

    # Unweighted
    degree_c = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight=None, normalized=True)
    closeness = nx.closeness_centrality(G)
    try:
        eigen = nx.eigenvector_centrality_numpy(G, weight=None)
    except Exception:
        eigen = nx.eigenvector_centrality(G, weight=None, max_iter=5000)

    # Weighted
    strength = {node: float(G.degree(node, weight="weight")) for node in G.nodes}
    strength_norm = {node: (strength[node] / (n - 1) if n > 1 else 0.0) for node in G.nodes}
    D = _inverse_weight_graph(G)
    weighted_betweenness = nx.betweenness_centrality(D, weight="distance", normalized=True)
    weighted_closeness = nx.closeness_centrality(D, distance="distance")
    try:
        weighted_eigen = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        weighted_eigen = nx.eigenvector_centrality(G, weight="weight", max_iter=5000)

    rows = []
    for node in G.nodes:
        rows.append({
            "SpeakerKey": node,
            "Speaker": _speaker_label(G, node),
            "SpeechCount": int(G.nodes[node].get("speech_count", 0)),
            "DegreeCentrality": float(degree_c.get(node, 0.0)),
            "WeightedDegree": float(strength.get(node, 0.0)),
            "WeightedDegreeNorm": float(strength_norm.get(node, 0.0)),
            "BetweennessCentrality": float(betweenness.get(node, 0.0)),
            "WeightedBetweennessCentrality": float(weighted_betweenness.get(node, 0.0)),
            "ClosenessCentrality": float(closeness.get(node, 0.0)),
            "WeightedClosenessCentrality": float(weighted_closeness.get(node, 0.0)),
            "EigenvectorCentrality": float(eigen.get(node, 0.0)),
            "WeightedEigenvectorCentrality": float(weighted_eigen.get(node, 0.0)),
        })

    dfc = pd.DataFrame(rows).sort_values(
        ["WeightedBetweennessCentrality", "WeightedDegree", "SpeechCount"],
        ascending=False
    ).reset_index(drop=True)

    rank_cols = [
        "DegreeCentrality", "WeightedDegree", "BetweennessCentrality",
        "WeightedBetweennessCentrality", "ClosenessCentrality",
        "WeightedClosenessCentrality", "EigenvectorCentrality",
        "WeightedEigenvectorCentrality",
    ]
    for col in rank_cols:
        dfc[f"Rank_{col}"] = dfc[col].rank(method="min", ascending=False).astype(int)

    pd.DataFrame([
        {"Metric": "EdgeWeight", "Definition": "같은 회의에서 두 의원이 연속 발언자로 등장한 횟수(전체 기간 합산)"},
        {"Metric": "WeightedDegree", "Definition": "연결된 의원들과의 연속 발언 연결 횟수 합(Strength)"},
        {"Metric": "WeightedShortestPath", "Definition": "거리 = 1 / EdgeWeight"},
    ]).to_csv(OUTPUT_DIR / "network_centrality_weight_definition.csv", encoding="utf-8-sig", index=False)
    dfc.to_csv(OUTPUT_DIR / "network_centrality.csv", encoding="utf-8-sig", index=False)

    # --------------------------------------------------------
    # 중심성 시각화: 4대 중심성을 독립 패널로 표시
    # - 각 패널의 OFF/ON은 같은 지표의 비가중/가중 버전
    # - 그래프 노드 크기 = 해당 지표값
    # - 랭킹 = 전체 그래프의 해당 지표값
    # --------------------------------------------------------
    metric_defs = [
        ("DegreeCentrality", "WeightedDegree", "차수중심성", "연결된 다른 의원 수", "가중 차수 · Strength", "연속 발언 연결 횟수의 총합"),
        ("BetweennessCentrality", "WeightedBetweennessCentrality", "매개중심성", "다른 의원 사이 최단경로를 지나는 정도", "가중 매개중심성", "연결 빈도를 거리로 반영한 최단경로 매개 위치"),
        ("ClosenessCentrality", "WeightedClosenessCentrality", "근접중심성", "다른 의원까지 평균 최단거리의 역수", "가중 근접중심성", "연결 빈도를 반영한 평균 거리의 역수"),
        ("EigenvectorCentrality", "WeightedEigenvectorCentrality", "고유벡터중심성", "중심성 높은 의원과 연결될수록 높은 값", "가중 고유벡터중심성", "연결강도와 이웃의 중심성을 함께 반영"),
    ]

    def metric_size(score_map, nodes):
        return _normalized_node_sizes(score_map, nodes, min_size=7, max_size=30)

    metric_blocks = []
    net_div_ids = []
    rank_div_ids = []
    net_annotations_off = []
    net_annotations_on = []

    for i, (off_col, on_col, name, off_desc, on_name, on_desc) in enumerate(metric_defs):
        off_nodes = list(dfc.nlargest(min(TOP_NETWORK_NODES, len(dfc)), off_col)["SpeakerKey"])
        on_nodes = list(dfc.nlargest(min(TOP_NETWORK_NODES, len(dfc)), on_col)["SpeakerKey"])
        off_H = G.subgraph(off_nodes).copy()
        on_H = G.subgraph(on_nodes).copy()
        pos_off = _safe_spring_layout(off_H, weight=None, k=0.9, iterations=80)
        pos_on = _safe_spring_layout(on_H, weight="weight", k=0.9, iterations=80)

        def make_network_traces(H, pos, score_col):
            nodes = list(H.nodes)
            score_map = dict(zip(dfc["SpeakerKey"], dfc[score_col]))
            labels = top_label_nodes(nodes, score_map, n=min(7, len(nodes)))
            ex, ey, et = [], [], []
            for u, v, d in H.edges(data=True):
                ex += [pos[u][0], pos[v][0], None]
                ey += [pos[u][1], pos[v][1], None]
                w = int(d.get("weight", 1))
                et += [f"{_speaker_label(H, u)} ↔ {_speaker_label(H, v)}<br>연속 발언 연결={w}회<extra></extra>"] * 2 + [None]
            edge = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.2, color="rgba(135,112,170,0.22)"),
                              hoverinfo="text", text=et, showlegend=False)
            node = go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                mode="markers",
                marker=dict(size=metric_size(score_map, nodes), opacity=0.86, line=dict(width=1.2, color="#ffffff")),
                hovertemplate=[
                    f"{_speaker_label(H, n)}<br>중심성={float(score_map.get(n,0)):.6f}<br>발언={int(H.nodes[n].get('speech_count',0))}<br>연결수={int(H.degree(n))}<br>Strength={float(strength.get(n,0)):.1f}<br>식별자={n}<extra></extra>"
                    for n in nodes
                ], showlegend=False,
            )
            return [edge, node]

        nt_off = make_network_traces(off_H, pos_off, off_col)
        nt_on = make_network_traces(on_H, pos_on, on_col)
        off_labels = top_label_nodes(list(off_H.nodes), dict(zip(dfc["SpeakerKey"], dfc[off_col])), n=min(7, len(off_H)))
        on_labels = top_label_nodes(list(on_H.nodes), dict(zip(dfc["SpeakerKey"], dfc[on_col])), n=min(7, len(on_H)))
        net_annotations_off.append(_node_label_annotations(pos_off, list(off_H.nodes), off_labels, font_size=11, yshift=9, label_map={n: _speaker_label(off_H, n) for n in off_H.nodes}))
        net_annotations_on.append(_node_label_annotations(pos_on, list(on_H.nodes), on_labels, font_size=11, yshift=9, label_map={n: _speaker_label(on_H, n) for n in on_H.nodes}))

        nf = go.Figure()
        for tr in nt_off:
            nf.add_trace(tr)
        for tr in nt_on:
            tr.visible = False
            nf.add_trace(tr)
        nf.update_layout(title=f"{name} · 가중 OFF", height=470,
                         margin=dict(l=25,r=15,t=55,b=20),
                         xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                         annotations=net_annotations_off[-1])

        off_rank = dfc.sort_values([off_col, "SpeechCount"], ascending=False).head(10).iloc[::-1].copy()
        on_rank = dfc.sort_values([on_col, "SpeechCount"], ascending=False).head(10).iloc[::-1].copy()
        rf = go.Figure()
        rf.add_trace(go.Bar(
            x=off_rank[off_col], y=off_rank["Speaker"], orientation="h",
            customdata=np.column_stack([off_rank[f"Rank_{off_col}"], off_rank["SpeechCount"], off_rank["SpeakerKey"]]),
            hovertemplate="%{y}<br>값=%{x:.6f}<br>순위=%{customdata[0]}<br>발언=%{customdata[1]}<br>식별자=%{customdata[2]}<extra></extra>"
        ))
        rf.add_trace(go.Bar(
            x=on_rank[on_col], y=on_rank["Speaker"], orientation="h",
            customdata=np.column_stack([on_rank[f"Rank_{on_col}"], on_rank["SpeechCount"], on_rank["SpeakerKey"]]),
            hovertemplate="%{y}<br>값=%{x:.6f}<br>순위=%{customdata[0]}<br>발언=%{customdata[1]}<br>식별자=%{customdata[2]}<extra></extra>",
            visible=False,
        ))
        rf.update_layout(title=f"{name} 랭킹 · 가중 OFF", height=470,
                         margin=dict(l=85,r=15,t=55,b=35), xaxis_title="중심성 값", yaxis_title="발언자")

        net_id=f"cent_net_{i}"; rank_id=f"cent_rank_{i}"
        net_div_ids.append(net_id); rank_div_ids.append(rank_id)
        metric_blocks.append(
            f'<section class="metric-panel"><div class="metric-head"><div><h2>{name}</h2><p>OFF: {off_desc} · ON: {on_name} — {on_desc}</p></div><span class="metric-mode" id="mode_{i}">가중 OFF</span></div>'
            f'<div class="metric-grid"><div>{nf.to_html(full_html=False, include_plotlyjs=False, div_id=net_id)}</div><div>{rf.to_html(full_html=False, include_plotlyjs=False, div_id=rank_id)}</div></div></section>'
        )

    defs_html = ''.join(
        f'<div class="def"><b>{name}</b><span>{off_desc}</span></div>'
        for _, _, name, off_desc, _, _ in metric_defs
    )
    html_parts = [
        '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>네트워크 중심성</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        '<style>',
        'body{margin:0;background:#fff;color:#18212b;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}',
        '.wrap{padding:18px 22px 28px;max-width:1700px;margin:0 auto}',
        '.top{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;margin-bottom:12px}',
        'h1{font-size:22px;margin:0 0 4px}.sub{font-size:11px;color:#687482}',
        '.controls{display:flex;gap:6px;align-items:center}.controls button{border:1px solid #cfd6dc;background:#fff;padding:7px 12px;border-radius:8px;font-size:11px;cursor:pointer}.controls button.active{font-weight:800;background:#eef2f5;border-color:#9da8b2}',
        '.defs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:9px}.def{border:1px solid #dde2e7;border-radius:9px;padding:8px;background:#f8fafb;min-height:47px}.def b{display:block;font-size:12px;margin-bottom:2px}.def span{font-size:11px;color:#5e6b77;line-height:1.35}',
        '.note{font-size:11px;color:#687482;padding:8px 10px;background:#f5f7f9;border-radius:8px;margin-bottom:10px;line-height:1.5}',
        '.metric-panel{border:1px solid #dde2e7;border-radius:12px;padding:10px;margin-bottom:12px;background:#fff}.metric-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;padding:2px 4px 4px}.metric-head h2{margin:0;font-size:16px}.metric-head p{margin:2px 0 0;font-size:11px;color:#697681}.metric-mode{font-size:10px;border:1px solid #cfd6dc;border-radius:999px;padding:4px 7px;color:#5f6b75;white-space:nowrap}',
        '.metric-grid{display:grid;grid-template-columns:1.25fr .75fr;gap:10px}',
        '@media(max-width:1050px){.defs{grid-template-columns:repeat(2,1fr)}.metric-grid{grid-template-columns:1fr}.top{flex-direction:column}.controls{align-self:flex-end}}@media(max-width:600px){.defs{grid-template-columns:1fr}.wrap{padding:12px}}',
        '</style></head><body><div class="wrap">',
        '<div class="top"><div><h1>네트워크 중심성</h1><div class="sub">발언흐름 네트워크 · 순위는 전체 네트워크 기준 · 네트워크는 지표별 상위 35명</div></div><div class="controls"><button id="btnOff" class="active" onclick="setWeight(false)">가중 OFF</button><button id="btnOn" onclick="setWeight(true)">가중 ON</button></div></div>',
        f'<div class="defs">{defs_html}</div>',
        '<div class="note"><b>가중치:</b> 같은 회의에서 두 의원이 연속 발언자로 등장한 횟수의 전체 기간 합. 가중 ON: 매개·근접의 거리=1/가중치, 고유벡터에 연결강도 반영.</div>',
        ''.join(metric_blocks),
        f'<script>function setWeight(weighted){{const nets={json.dumps(net_div_ids, ensure_ascii=False)};const ranks={json.dumps(rank_div_ids, ensure_ascii=False)};const annsOff={json.dumps(net_annotations_off, ensure_ascii=False)};const annsOn={json.dumps(net_annotations_on, ensure_ascii=False)};const vn=weighted?[false,false,true,true]:[true,true,false,false];const vr=weighted?[false,true]:[true,false];nets.forEach((id,i)=>{{const e=document.getElementById(id);if(e){{Plotly.restyle(e,{{visible:vn}});Plotly.relayout(e,{{title:(i===0?"차수중심성":i===1?"매개중심성":i===2?"근접중심성":"고유벡터중심성")+" · "+(weighted?"가중 ON":"가중 OFF"),annotations:weighted?annsOn[i]:annsOff[i]}});}}}});ranks.forEach((id,i)=>{{const e=document.getElementById(id);if(e){{Plotly.restyle(e,{{visible:vr}});Plotly.relayout(e,{{title:(i===0?"차수중심성":i===1?"매개중심성":i===2?"근접중심성":"고유벡터중심성")+" 랭킹 · "+(weighted?"가중 ON":"가중 OFF")}});}}}});document.getElementById("btnOff").classList.toggle("active",!weighted);document.getElementById("btnOn").classList.toggle("active",weighted);document.querySelectorAll(".metric-mode").forEach(e=>e.textContent=weighted?"가중 ON":"가중 OFF");}}</script>',
        '</div></body></html>'
    ]
    (OUTPUT_DIR / "20_중심성_분석.html").write_text("\n".join(html_parts), encoding="utf-8")
    return dfc

    pos_unweighted = nx.spring_layout(H, seed=42, weight=None, iterations=100, k=0.9)
    pos_weighted = nx.spring_layout(H, seed=42, weight="weight", iterations=100, k=0.9)
    label_nodes = top_label_nodes(list(H.nodes), {node: strength.get(node, 0.0) for node in H.nodes}, n=min(8, len(H.nodes)))

    def network_traces(pos, weighted_mode):
        ex, ey = [], []
        edge_hover = []
        for u, v, data in H.edges(data=True):
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
            edge_hover.extend([f"{u} ↔ {v}<br>연속 발언 연결={int(data.get('weight', 1))}회<extra></extra>",
                               f"{u} ↔ {v}<br>연속 발언 연결={int(data.get('weight', 1))}회<extra></extra>",
                               None])
        edge_trace = go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=1.8 if weighted_mode else 1.0),
            hoverinfo="text", text=edge_hover, showlegend=False,
        )

        node_sizes = [
            capped_node_size(
                strength.get(node, 0.0) if weighted_mode else H.nodes[node].get("speech_count", 0),
                base=6.0, scale=0.75 if weighted_mode else 1.0, cap=12
            )
            for node in H.nodes
        ]
        node_trace = go.Scatter(
            x=[pos[node][0] for node in H.nodes],
            y=[pos[node][1] for node in H.nodes],
            text=[node if node in label_nodes else "" for node in H.nodes],
            mode="markers+text", textposition="top center", textfont=dict(size=8),
            marker=dict(size=node_sizes, opacity=0.82),
            hovertemplate=[
                f"{node}<br>발언={int(H.nodes[node].get('speech_count', 0))}<br>연결수={H.degree(node)}<br>Strength={strength.get(node, 0):.1f}<extra></extra>"
                for node in H.nodes
            ],
            showlegend=False,
        )
        return [edge_trace, node_trace]

    traces_unweighted = network_traces(pos_unweighted, False)
    traces_weighted = network_traces(pos_weighted, True)
    n_net = len(traces_unweighted)

    def rank_bar(value_col, topn=10):
        top = dfc.sort_values(value_col, ascending=False).head(topn).iloc[::-1]
        custom = np.column_stack([top[f"Rank_{value_col}"].to_numpy(), top["SpeechCount"].to_numpy()])
        return go.Bar(
            x=top[value_col], y=top["SpeakerKey"], orientation="h", customdata=custom,
            hovertemplate="%{y}<br>값=%{x:.6f}<br>순위=%{customdata[0]}<br>발언=%{customdata[1]}<extra></extra>",
            showlegend=False,
        )

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type":"xy", "colspan":2}, None], [{"type":"xy"},{"type":"xy"}], [{"type":"xy"},{"type":"xy"}]],
        subplot_titles=(
            "발언흐름 네트워크 · 가중치 OFF / ON",
            "차수 · 가중 차수(Strength)",
            "매개중심성",
            "근접중심성",
            "고유벡터중심성",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )
    for tr in traces_unweighted:
        fig.add_trace(tr, row=1, col=1)
    for tr in traces_weighted:
        tr.visible = False
        fig.add_trace(tr, row=1, col=1)

    panel_specs = [
        (2, 1, "DegreeCentrality", "WeightedDegree"),
        (2, 2, "BetweennessCentrality", "WeightedBetweennessCentrality"),
        (3, 1, "ClosenessCentrality", "WeightedClosenessCentrality"),
        (3, 2, "EigenvectorCentrality", "WeightedEigenvectorCentrality"),
    ]
    for row, col, off_col, on_col in panel_specs:
        off = rank_bar(off_col)
        on = rank_bar(on_col)
        on.visible = False
        fig.add_trace(off, row=row, col=col)
        fig.add_trace(on, row=row, col=col)

    total = len(fig.data)
    vis_off = [False] * total
    vis_on = [False] * total
    for i in range(n_net):
        vis_off[i] = True
    for i in range(n_net, 2*n_net):
        vis_on[i] = True
    idx = 2*n_net
    for _ in panel_specs:
        vis_off[idx] = True
        vis_on[idx+1] = True
        idx += 2

    fig.update_layout(
        template="plotly_white", height=1320,
        margin=dict(l=65, r=30, t=105, b=105),
        updatemenus=[dict(
            type="buttons", direction="left", showactive=True,
            buttons=[
                {"label":"가중치 OFF", "method":"update", "args":[{"visible":vis_off}]},
                {"label":"가중치 ON", "method":"update", "args":[{"visible":vis_on}]},
            ], x=0.0, y=1.07,
        )],
        annotations=[
            dict(text="가중치 = 같은 회의에서 두 의원이 연속 발언자로 등장한 횟수. 가중 ON: 매개·근접 거리=1/가중치, 고유벡터=연결강도 반영.", x=0, y=1.035, xref="paper", yref="paper", showarrow=False, align="left", font=dict(size=10)),
            dict(text="순위는 전체 발언흐름 네트워크 기준 · 네트워크 표시는 발언량 상위 35명 · 동률은 공동순위.", x=0, y=-0.045, xref="paper", yref="paper", showarrow=False, align="left", font=dict(size=9)),
        ],
        showlegend=False,
    )
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1, scaleanchor="x", scaleratio=1)
    for row, col in [(2,1),(2,2),(3,1),(3,2)]:
        fig.update_xaxes(title_text="중심성 값", row=row, col=col)
        fig.update_yaxes(title_text="발언자", row=row, col=col)

    defs = [
        ("차수", "연결된 다른 의원 수"),
        ("가중 차수 · Strength", "연속 발언 연결 횟수의 총합"),
        ("매개", "다른 의원 사이 최단경로를 지나는 정도"),
        ("근접", "다른 의원까지 평균 최단거리의 역수"),
        ("고유벡터", "중심성 높은 의원과 연결될수록 높은 값"),
    ]
    def_html = "".join(f'<div class="def"><b>{a}</b><span>{b}</span></div>' for a,b in defs)
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id="centrality_main")
    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>네트워크 중심성</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{margin:0;background:#fff;color:#18212b;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}}
.wrap{{padding:18px 22px 24px;max-width:1500px;margin:0 auto}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{font-size:11px;color:#687482;margin-bottom:12px}}
.defs{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:0 0 10px}}
.def{{border:1px solid #dde2e7;border-radius:9px;padding:8px;background:#f8fafb;min-height:48px}}
.def b{{display:block;font-size:11px;margin-bottom:2px}} .def span{{font-size:10px;color:#5e6b77;line-height:1.35}}
.note{{font-size:10px;color:#687482;padding:7px 9px;background:#f5f7f9;border-radius:8px;margin-bottom:6px;line-height:1.45}}
@media(max-width:1000px){{.defs{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:560px){{.defs{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>네트워크 중심성</h1><div class="sub">발언흐름 네트워크 · 가중치 ON/OFF 비교</div>
<div class="defs">{def_html}</div>
<div class="note"><b>가중치 기준:</b> 동일 회의에서 두 의원이 연속 발언자로 등장한 횟수의 전체 기간 합산. 가중치 자체는 실제 정치적 영향력이나 상호작용의 강도를 직접 의미하지 않습니다.</div>
{inner}
</div></body></html>"""
    (OUTPUT_DIR / "20_중심성_분석.html").write_text(html, encoding="utf-8")
    return dfc


# ------------------------------------------------------------
# Ego network
# ------------------------------------------------------------


def ego_network_analysis(G, centrality_df):
    """4개 중심성의 1위 의원 에고 네트워크를 2×2 패널로 만들고 각 패널에서 가중 OFF/ON을 전환한다."""
    if G is None or nx is None or centrality_df.empty:
        skip("에고 네트워크 생성에 필요한 데이터가 없습니다.")
        return

    specs = [
        ("DegreeCentrality", "WeightedDegree", "차수중심성"),
        ("BetweennessCentrality", "WeightedBetweennessCentrality", "매개중심성"),
        ("ClosenessCentrality", "WeightedClosenessCentrality", "근접중심성"),
        ("EigenvectorCentrality", "WeightedEigenvectorCentrality", "고유벡터중심성"),
    ]
    panels=[]
    for i,(off_col,on_col,label) in enumerate(specs):
        if off_col not in centrality_df.columns or on_col not in centrality_df.columns:
            continue
        off_center=centrality_df.sort_values([off_col,"SpeechCount"],ascending=False).iloc[0]["SpeakerKey"]
        on_center=centrality_df.sort_values([on_col,"SpeechCount"],ascending=False).iloc[0]["SpeakerKey"]
        if off_center not in G or on_center not in G:
            continue

        def ego_fig(center, weighted, mode_label):
            ego=nx.ego_graph(G,center,radius=1)
            neighbors=sorted(((n,G[center][n].get("weight",1)) for n in ego.neighbors(center)),key=lambda x:x[1],reverse=True)
            keep={center}|{n for n,_ in neighbors[:12]}
            ego=G.subgraph(keep).copy()
            pos=_safe_spring_layout(ego,weight="weight" if weighted else None,seed=42,k=1.0,iterations=110)
            nodes=list(ego.nodes)
            ex,ey,et=[],[],[]
            for u,v,d in ego.edges(data=True):
                ex += [pos[u][0],pos[v][0],None]; ey += [pos[u][1],pos[v][1],None]
                h=f"{_speaker_label(ego,u)} ↔ {_speaker_label(ego,v)}<br>연속 발언={int(d.get('weight',1))}회<extra></extra>"
                et += [h,h,None]
            edge=go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1.5 if weighted else 1.1,color="rgba(135,112,170,0.29)"),hoverinfo="text",text=et,showlegend=False)
            sizes=[]
            for n in nodes:
                if n==center:
                    sizes.append(30)
                else:
                    sizes.append(float(min(19,8+math.sqrt(max(float(G[center][n].get("weight",1)),1))*1.75)))
            node=go.Scatter(
                x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],mode="markers",
                marker=dict(size=sizes,opacity=.88,line=dict(width=1.3,color="#ffffff")),
                hovertemplate=[f"{_speaker_label(ego,n)}<br>센터={_speaker_label(ego,center)}<br>연속 발언={int(G[center][n].get('weight',0)) if n!=center else '-'}<br>식별자={n}<extra></extra>" for n in nodes],
                showlegend=False)
            label_nodes={center}|{n for n,_ in neighbors[:5]}
            anns=_node_label_annotations(pos,nodes,label_nodes,font_size=11,yshift=9,label_map={n:_speaker_label(ego,n) for n in nodes})
            fig=go.Figure([edge,node])
            fig.update_layout(title=dict(text=f"{mode_label} · {'가중 ON' if weighted else '가중 OFF'} · {_speaker_label(G,center)}",x=.5,xanchor="center",y=.97,yanchor="top",font=dict(size=15)),height=430,margin=dict(l=10,r=10,t=60,b=12),xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x"),annotations=anns)
            return fig

        off_html=ego_fig(off_center,False,label).to_html(full_html=False,include_plotlyjs=False,div_id=f"ego_off_{i}")
        on_html=ego_fig(on_center,True,label).to_html(full_html=False,include_plotlyjs=False,div_id=f"ego_on_{i}")
        panels.append((i,off_html,on_html))

    if not panels:
        skip("중심성별 1위 의원을 찾지 못해 Ego Network 생략")
        return

    parts=[]
    parts.append('''<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>
body{margin:0;background:#fff;color:#18212b;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
.wrap{width:100%;box-sizing:border-box;padding:16px 20px 24px;margin:0 auto;max-width:1800px}
.note{font-size:12px;color:#53616d;margin:0 0 11px;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,430px);gap:14px;align-items:stretch;width:100%}
.panel{position:relative;border:1px solid #d9e0e5;border-radius:12px;padding:2px;overflow:hidden;min-width:0;min-height:0;background:#fff}
.toggle{position:absolute;z-index:20;right:12px;top:10px;display:flex;gap:4px;background:rgba(255,255,255,.96);border:1px solid #cbd4db;border-radius:9px;padding:3px;box-shadow:0 2px 8px rgba(30,45,60,.08)}
.toggle button{border:0;background:transparent;color:#55636f;padding:5px 9px;border-radius:6px;font-size:11px;font-weight:700;cursor:pointer}.toggle button.active{background:#263f55;color:#fff}
.plot-off,.plot-on{width:100%;height:100%}.plot-on{display:none}.panel .plotly-graph-div{width:100%!important;height:100%!important}
@media(max-width:1000px){.grid{grid-template-columns:1fr;grid-template-rows:none}.panel{min-height:430px}}
</style></head><body><div class="wrap"><div class="note"><b>에고 네트워크</b> · 4개 중심성별 1위 의원을 중심으로 1차 연결망을 비교합니다. 각 패널에서 <b>가중 OFF / 가중 ON</b>을 전환할 수 있습니다.</div><div class="grid">''')
    for i,off_html,on_html in panels:
        parts.append(f'<div class="panel"><div class="toggle"><button class="active" onclick="setMode({i},false,this)">가중 OFF</button><button onclick="setMode({i},true,this)">가중 ON</button></div><div class="plot-off">{off_html}</div><div class="plot-on">{on_html}</div></div>')
    parts.append('''</div></div><script>
function setMode(i,on,btn){var panel=btn.closest('.panel');panel.querySelector('.plot-off').style.display=on?'none':'block';panel.querySelector('.plot-on').style.display=on?'block':'none';panel.querySelectorAll('.toggle button').forEach(function(b){b.classList.remove('active')});btn.classList.add('active');setTimeout(function(){try{var target=panel.querySelector(on?'.plot-on .plotly-graph-div':'.plot-off .plotly-graph-div');if(target&&window.Plotly)Plotly.Plots.resize(target);}catch(e){}},30);}
</script></body></html>''')
    (OUTPUT_DIR/"21_에고네트워크.html").write_text("".join(parts),encoding="utf-8")

# ------------------------------------------------------------
# Community detection
# ------------------------------------------------------------

def community_detection_analysis(G):
    # 발언흐름 네트워크의 구조적 커뮤니티를 탐색적으로 검출.
    if G is None or nx is None or len(G.nodes) < 2:
        skip("커뮤니티 탐지에 사용할 네트워크가 없습니다.")
        return None
    communities = None
    method = None
    try:
        fn = getattr(nx.community, "louvain_communities", None)
        if callable(fn):
            communities = fn(G, weight="weight", seed=42, resolution=1.0)
            method = "Louvain(weighted)"
    except Exception as exc:
        log(f"[WARN] Louvain 실패: {type(exc).__name__}")
    if communities is None:
        try:
            communities = list(nx.community.asyn_lpa_communities(G, weight="weight", seed=42))
            method = "Asynchronous LPA(weighted)"
        except Exception as exc:
            log(f"[WARN] LPA 실패: {type(exc).__name__}")
    if communities is None:
        try:
            communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
            method = "Greedy Modularity(weighted)"
        except Exception as exc:
            skip(f"커뮤니티 탐지 실패: {type(exc).__name__}: {exc}")
            return None

    # 크기순 번호: C1 최대, C2 2위, C3 3위
    ordered = sorted(communities, key=lambda m: (-len(m), sorted(map(str, m))[0] if m else ""))
    cmap = {}
    sizes = {}
    for cid, members in enumerate(ordered, start=1):
        sizes[cid] = len(members)
        for node in members:
            cmap[node] = cid

    cdf = pd.DataFrame([
        {"SpeakerKey": node, "Community": int(cmap[node]), "CommunityName": f"커뮤니티 {cmap[node]}",
         "CommunitySize": int(sizes[cmap[node]]), "SpeechCount": int(G.nodes[node].get("speech_count", 0)),
         "Strength": float(G.degree(node, weight="weight"))}
        for node in G.nodes
    ]).sort_values(["Community", "Strength", "SpeechCount"], ascending=[True, False, False])
    cdf.to_csv(OUTPUT_DIR / "network_communities.csv", encoding="utf-8-sig", index=False)

    base_nodes = [n for n, _ in sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:min(TOP_COMMUNITY_NODES, len(G))]]
    must_show = set(base_nodes)
    top3_summary = []
    for cid in range(1, min(3, len(ordered)) + 1):
        members = sorted(list(ordered[cid - 1]), key=lambda n: (G.degree(n, weight="weight"), G.nodes[n].get("speech_count", 0)), reverse=True)[:3]
        must_show.update(members)
        top3_summary.append({"Community": cid, "CommunityName": f"커뮤니티 {cid}", "Size": sizes[cid], "Members": members})

    required = {m for x in top3_summary for m in x["Members"]}
    display_nodes = list(must_show)
    if len(display_nodes) > TOP_COMMUNITY_NODES + 9:
        remain = [n for n in base_nodes if n not in required]
        display_nodes = list(required) + remain[:max(0, TOP_COMMUNITY_NODES + 9 - len(required))]
    H = G.subgraph(display_nodes).copy()
    if len(H.nodes) < 2:
        skip("커뮤니티 시각화용 주요 노드가 2명 미만입니다.")
        return cdf

    pos = _safe_spring_layout(H, weight="weight", k=1.1, iterations=120)
    nodes = list(H.nodes)
    scores = {n: H.degree(n, weight="weight") for n in nodes}
    label_nodes = set()
    for x in top3_summary:
        label_nodes.update([n for n in x["Members"] if n in H])
    label_nodes.update(top_label_nodes(nodes, scores, n=min(8, len(nodes))))

    ex, ey, et = [], [], []
    for u, v, d in H.edges(data=True):
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
        h = f"{u} ↔ {v}<br>연속 발언 연결={int(d.get('weight',1))}회<extra></extra>"
        et += [h, h, None]
    edge = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.15), hoverinfo="text", text=et, showlegend=False)

    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#64748b", "#a16207", "#be185d", "#4f46e5"]
    traces = [edge]
    for cid in sorted({cmap[n] for n in nodes}):
        ns = [n for n in nodes if cmap[n] == cid]
        col = palette[(cid - 1) % len(palette)]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers",
            marker=dict(size=_normalized_node_sizes(scores, ns, 8, 24), opacity=0.84, color=col, line=dict(width=1.1, color="#ffffff")),
            name=f"커뮤니티 {cid} ({sizes[cid]}명)",
            hovertemplate=[f"{n}<br>커뮤니티 {cid}<br>발언={int(H.nodes[n].get('speech_count',0))}<br>연결강도={float(H.degree(n,weight='weight')):.1f}<extra></extra>" for n in ns]
        ))
    fig = go.Figure(traces)
    _append_layout_annotations(fig, _node_label_annotations(pos, nodes, label_nodes, font_size=10, yshift=8))
    fig.update_layout(title=f"커뮤니티 탐지 · {method} · {len(ordered)}개", height=690,
                      margin=dict(l=35, r=35, t=60, b=125), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.03, yanchor="top", x=0, xanchor="left", font=dict(size=9)))

    cards=[]
    for x in top3_summary:
        cid=x["Community"]; col=palette[(cid-1)%len(palette)]; members=" · ".join(x["Members"]) if x["Members"] else "주요 구성원 없음"
        cards.append(f'<div class="community-card"><div class="community-name"><span class="swatch" style="background:{col}"></span><b>커뮤니티 {cid}</b><span class="community-size">{x["Size"]:,}명</span></div><div class="community-members">주요 구성원: {members}</div></div>')
    pd.DataFrame([{"Community":x["Community"],"CommunityName":x["CommunityName"],"Size":x["Size"],"Member1":x["Members"][0] if len(x["Members"])>0 else "","Member2":x["Members"][1] if len(x["Members"])>1 else "","Member3":x["Members"][2] if len(x["Members"])>2 else ""} for x in top3_summary]).to_csv(OUTPUT_DIR/"network_community_top3.csv",encoding="utf-8-sig",index=False)
    inner=fig.to_html(full_html=False,include_plotlyjs=False,div_id="community_main")
    html=f'''<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>커뮤니티 탐지</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>body{{margin:0;background:#fff;color:#18212b;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}}.wrap{{max-width:1600px;margin:0 auto;padding:16px 20px 22px}}.community-note{{font-size:10px;color:#687482;margin:0 0 8px}}.community-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:8px}}.community-card{{border:1px solid #dde2e7;border-radius:10px;padding:9px 11px;background:#f8fafb}}.community-name{{display:flex;align-items:center;gap:7px;font-size:11px}}.swatch{{width:10px;height:10px;border-radius:50%;display:inline-block}}.community-size{{margin-left:auto;color:#687482;font-size:9px}}.community-members{{margin-top:5px;color:#475461;font-size:10px;line-height:1.4}}.result-note{{margin-top:9px;font-size:9px;color:#687482}}</style></head><body><div class="wrap"><div class="community-note"><b>요점</b> {len(ordered)}개 구조적 커뮤니티 · 상위 3개 그룹과 주요 구성원 3명을 아래에 표시합니다.</div>{inner}<div class="community-grid">{"".join(cards)}</div><div class="result-note">※ 관측된 발언 연결구조의 군집이며 정치적 세력·이념·파벌을 직접 의미하지 않습니다.</div></div></body></html>'''
    (OUTPUT_DIR/"24_커뮤니티_탐지.html").write_text(html,encoding="utf-8")
    pd.DataFrame([{"Method":method,"Communities":len(ordered),"LargestSize":max(sizes.values()) if sizes else 0}]).to_csv(OUTPUT_DIR/"network_community_summary.csv",encoding="utf-8-sig",index=False)
    log(f"[OK] 커뮤니티 탐지: {method}, {len(ordered)}개 커뮤니티")
    return cdf


# ------------------------------------------------------------
# Timeline network
# ------------------------------------------------------------


def timeline_network(df, flow_graph):
    if flow_graph is None or nx is None:
        skip("시간축 네트워크에 사용할 flow graph가 없습니다.")
        return
    top=[n for n,_ in sorted(flow_graph.degree,key=lambda x:x[1],reverse=True)[:TOP_NETWORK_NODES]]
    G=flow_graph.subgraph(top).copy()
    if len(G.nodes)<2:
        skip("시간축 네트워크용 주요 노드가 부족합니다.")
        return
    pos=nx.spring_layout(G,seed=42,weight="weight",k=1.15,iterations=140)
    years=sorted(int(y) for y in df["Year"].dropna().unique())
    frame_data=[]; all_speakers=list(G.nodes); label_nodes=set(top[:9])
    for year in years:
        part=df[(df["Year"]==year)&(df["SpeakerKey"].isin(all_speakers))].copy().sort_values(["MeetingID","SeqNum"])
        edges=Counter()
        for _,meeting in part.groupby("MeetingID",sort=False):
            sp=[x for x in meeting["SpeakerKey"] if x]
            for a,b in zip(sp,sp[1:]):
                if a==b: continue
                u,v=sorted([a,b]); edges[(u,v)]+=1
        ex,ey,et=[],[],[]
        for (u,v),w in edges.items():
            if u not in pos or v not in pos: continue
            x0,y0=pos[u]; x1,y1=pos[v]; ex += [x0,x1,None]; ey += [y0,y1,None]
            h=f"{_speaker_label(G,u)} ↔ {_speaker_label(G,v)}<br>연속 발언 연결={int(w)}회<extra></extra>"; et += [h,h,None]
        active=part["SpeakerKey"].value_counts(); nodes=[n for n in all_speakers if n in pos]
        nx_trace=go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],mode="markers",textfont=dict(size=11,color="#253443",family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),marker=dict(size=[capped_node_size(active.get(n,0),base=8,scale=1.45,cap=18) for n in nodes],opacity=.86,line=dict(width=1.2,color="#ffffff")),hovertemplate=[f"{_speaker_label(G,n)}<br>{year}년 발언={active.get(n,0)}<br>식별자={n}<extra></extra>" for n in nodes],showlegend=False)
        edge_trace=go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1.35,color="rgba(135,112,170,0.30)"),hoverinfo="text",text=et,showlegend=False)
        frame_data.append(go.Frame(data=[edge_trace,nx_trace],name=str(year)))
    if not frame_data:
        skip("연도별 frame을 만들 수 없습니다."); return
    first=frame_data[0]; fig=go.Figure(data=first.data,frames=frame_data)
    anns=[dict(text="동일한 node 좌표를 고정하여 연도별 구조 변화를 비교",x=.5,y=-.085,xref="paper",yref="paper",showarrow=False,xanchor="center",align="center",font=dict(size=11,color="#52606c"))]
    anns += _node_label_annotations(pos,list(G.nodes),label_nodes,font_size=11,yshift=9,label_map={n:_speaker_label(G,n) for n in G.nodes})
    fig.update_layout(title=dict(text="시간축 네트워크: 연도별 발언흐름 구조",x=.5,xanchor="center",y=.965,yanchor="top",font=dict(size=20)),xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x"),margin=dict(l=18,r=18,t=78,b=150),height=900,updatemenus=[dict(type="buttons",showactive=True,buttons=[dict(label="▶ 재생",method="animate",args=[None,{"frame":{"duration":1500,"redraw":True},"transition":{"duration":350},"fromcurrent":True}])],x=.98,xanchor="right",y=1.055)],sliders=[dict(active=0,currentvalue={"prefix":"연도: ","font":{"size":12}},len=.92,x=.04,xanchor="left",y=-.105,steps=[dict(label=str(y),method="animate",args=[[str(y)],{"mode":"immediate","frame":{"duration":500,"redraw":True},"transition":{"duration":0}}]) for y in years])],annotations=anns)
    figure_write(fig,OUTPUT_DIR/"22_시간축_네트워크.html")

# ------------------------------------------------------------
# 지식그래프: event-centered
# ------------------------------------------------------------


def build_knowledge_graph(df, speakers, dev_keywords, boundary_keywords):
    if nx is None:
        skip("networkx가 없어 지식그래프 생략")
        return

    G = nx.Graph()
    top_speakers = list(speakers.head(TOP_NETWORK_NODES).index) if not speakers.empty else []
    top_df = df[df["SpeakerKey"].isin(top_speakers)].copy()
    relevant = top_df[(top_df["발전담론"] == "예") | (top_df["경계담론"] == "예")].copy()
    if relevant.empty:
        skip("관련 담론 발언이 없어 event-centered 지식그래프 생략")
        return

    # speaker -> metadata
    for sp_key in top_speakers:
        row = speakers.loc[sp_key]
        G.add_node(f"speaker::{sp_key}", label=row["Speaker"], type="의원")
        for dim in METADATA_DIMENSIONS:
            val = clean(row[dim]) or "미상"
            G.add_node(f"{dim}::{val}", label=val, type=dim)
            G.add_edge(f"speaker::{sp_key}", f"{dim}::{val}", relation=dim)

    for fam in FAMILIES:
        G.add_node(f"family::{fam}", label=fam, type="담론계열")

    keyword_counts = Counter()
    selected_speeches = relevant.sort_values(["Date", "MeetingID", "SeqNum"]).head(120)
    for _, row in selected_speeches.iterrows():
        sid = clean(row["SpeechID"])
        if not sid:
            continue
        G.add_node(f"speech::{sid}", label=f"발언 {sid}", type="발언", date=str(row["Date"].date()), speaker=row["Speaker"])
        G.add_edge(f"speech::{sid}", f"speaker::{row['SpeakerKey']}", relation="발언자")
        fams = set(split_semicolon(row["발전계열"]) + split_semicolon(row["경계계열"]))
        for fam in fams:
            if fam in FAMILIES:
                G.add_edge(f"speech::{sid}", f"family::{fam}", relation="담론계열")
        kws = set(split_semicolon(row["발전키워드"]) + split_semicolon(row["경계키워드"]))
        for kw in kws:
            if kw in KEYWORD_TO_FAMILY:
                keyword_counts[kw] += 1

    selected_kws = {k for k, _ in keyword_counts.most_common(TOP_NETWORK_KEYWORDS)}
    for kw in selected_kws:
        G.add_node(f"keyword::{kw}", label=kw, type="키워드")

    # speech -> keyword; family -> keyword mapping is treated as taxonomy, not empirical relation
    for _, row in selected_speeches.iterrows():
        sid = clean(row["SpeechID"])
        if not sid:
            continue
        for kw in set(split_semicolon(row["발전키워드"]) + split_semicolon(row["경계키워드"])):
            if kw in selected_kws:
                G.add_edge(f"speech::{sid}", f"keyword::{kw}", relation="키워드")

    if len(G.nodes) < 2:
        skip("event-centered 지식그래프 노드가 충분하지 않습니다.")
        return

    pos = nx.spring_layout(G, seed=42, iterations=160, weight=None, k=1.2)
    edge_x, edge_y = [], []
    for u, v in G.edges:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5), hoverinfo="none", showlegend=False)

    traces = [edge_trace]
    type_order = ["의원", "정당", "지역", "성별", "당선횟수", "당선방법", "발언", "담론계열", "키워드"]
    for typ in type_order:
        ns = [n for n in G.nodes if G.nodes[n].get("type") == typ]
        if not ns:
            continue
        if typ == "의원":
            label_nodes = set(sorted(ns, key=lambda n: len(G[n]), reverse=True)[:7])
        elif typ == "담론계열":
            label_nodes = set(ns)
        elif typ == "키워드":
            label_nodes = set(sorted(ns, key=lambda n: G.degree(n), reverse=True)[:8])
        else:
            label_nodes = set()
        traces.append(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
            mode="markers",
            name=typ,
            textfont=dict(size=9 if typ not in {"의원","담론계열"} else 10, color="#253443", family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
            marker=dict(size=[10 if typ == "의원" else 8 if typ == "담론계열" else 5.5 for _ in ns], opacity=0.80, line=dict(width=1.0, color="#ffffff")),
            hovertemplate=[f"{G.nodes[n]['label']}<br>type={typ}<extra></extra>" for n in ns],
        ))

    fig = go.Figure(traces)
    kg_label_anns = []
    for typ in type_order:
        ns = [n for n in G.nodes if G.nodes[n].get("type") == typ]
        if not ns:
            continue
        if typ == "의원":
            lnodes = set(sorted(ns, key=lambda n: len(G[n]), reverse=True)[:7])
        elif typ == "담론계열":
            lnodes = set(ns)
        elif typ == "키워드":
            lnodes = set(sorted(ns, key=lambda n: G.degree(n), reverse=True)[:8])
        else:
            lnodes = set()
        kg_label_anns.extend([dict(
            x=float(pos[n][0]), y=float(pos[n][1]), text=f"<b>{html_lib.escape(str(G.nodes[n]['label']))}</b>",
            showarrow=False, xanchor="center", yanchor="bottom", yshift=8, bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(183,191,200,0.80)", borderwidth=1, borderpad=2,
            font=dict(size=10, color="#253443", family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        ) for n in ns if n in lnodes])
    fig.update_layout(
        title="Event/Speech 중심 지식그래프",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        annotations=[dict(
            text="발언(Event)을 중심에 두고 의원·정당·지역·성별·당선정보·담론·키워드를 연결. 직접 상호작용 관계는 포함하지 않음.",
            x=0, y=-0.06, xref="paper", yref="paper", showarrow=False,
        )] + kg_label_anns,
    )
    figure_write(fig, OUTPUT_DIR / "23_지식그래프.html")

    pd.DataFrame([
        {"Source":u,"Target":v,"Relation":G.edges[u,v].get("relation", "") ,"SourceType":G.nodes[u].get("type"),"TargetType":G.nodes[v].get("type")}
        for u,v in G.edges
    ]).to_csv(OUTPUT_DIR / "network_knowledge_graph_edges.csv", encoding="utf-8-sig", index=False)
    return G


# ------------------------------------------------------------
# 네트워크 품질 검증
# ------------------------------------------------------------


def network_validation(df):
    issues = []
    # 동일 회의 내 순번 중복
    if df["SeqNum"].notna().any():
        dup = df[df["SeqNum"].notna()].duplicated(["MeetingID", "SeqNum"], keep=False)
        if dup.any():
            issues.append({"check": "meeting_seq_duplicate", "count": int(dup.sum()), "status": "WARN"})
        else:
            issues.append({"check": "meeting_seq_duplicate", "count": 0, "status": "OK"})

    # 발언자 메타데이터 다중값 이상 여부
    for dim in METADATA_DIMENSIONS:
        nunique = df[df["SpeakerKey"] != ""].groupby("SpeakerKey")[dim].agg(lambda s: len(set(clean(x) for x in s if clean(x))))
        bad = int((nunique > 1).sum())
        issues.append({"check": f"speaker_multiple_{dim}", "count": bad, "status": "WARN" if bad else "OK"})

    out = pd.DataFrame(issues)
    out.to_csv(OUTPUT_DIR / "validation_network.csv", encoding="utf-8-sig", index=False)
    return out


# ------------------------------------------------------------
# Annotation 미래 확장
# ------------------------------------------------------------


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def candidate_columns(df, patterns):
    cols = []
    for c in df.columns:
        for p in patterns:
            if re.search(p, c, flags=re.I):
                cols.append(c)
                break
    return cols


def cohens_kappa(a: pd.Series, b: pd.Series) -> float | None:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    pair["a"] = pair["a"].map(clean)
    pair["b"] = pair["b"].map(clean)
    pair = pair[(pair["a"] != "") & (pair["b"] != "")]
    if pair.empty:
        return None
    po = (pair["a"] == pair["b"]).mean()
    labels = sorted(set(pair["a"]) | set(pair["b"]))
    pe = 0.0
    n = len(pair)
    for label in labels:
        pa = (pair["a"] == label).sum() / n
        pb = (pair["b"] == label).sum() / n
        pe += pa * pb
    if abs(1 - pe) < 1e-12:
        return None
    return float((po - pe) / (1 - pe))


def macro_f1(y_true, y_pred):
    pairs = [(clean(a), clean(b)) for a, b in zip(y_true, y_pred) if clean(a) and clean(b)]
    if not pairs:
        return None
    labels = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    f1s = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in pairs)
        fp = sum(a != label and b == label for a, b in pairs)
        fn = sum(a == label and b != label for a, b in pairs)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return float(sum(f1s) / len(f1s)) if f1s else None


def annotation_reliability_over_time(df, annotators):
    if len(annotators) < 2:
        skip("2개 이상의 annotator 열이 없어 시간별 annotation reliability 생략")
        return
    work = df.copy()
    if "Month" not in work.columns:
        if "Date" in work.columns:
            work["_AnnotationMonth"] = pd.to_datetime(work["Date"], errors="coerce").dt.to_period("M").astype(str)
        elif "Year" in work.columns:
            work["_AnnotationMonth"] = work["Year"].astype(str)
        else:
            skip("Date/Year 열이 없어 시간별 annotation reliability 생략")
            return
    else:
        work["_AnnotationMonth"] = work["Month"]
    rows = []
    for month, part in work.groupby("_AnnotationMonth"):
        for a, b in itertools.combinations(annotators, 2):
            aa = part[a].map(clean); bb = part[b].map(clean)
            valid = (aa != "") & (bb != "")
            if valid.sum() == 0:
                continue
            rows.append({
                "Month": month, "AnnotatorA": a, "AnnotatorB": b,
                "N": int(valid.sum()),
                "Agreement": float((aa[valid] == bb[valid]).mean()),
                "CohenKappa": cohens_kappa(aa[valid], bb[valid]),
            })
    if not rows:
        skip("시간별 annotation 비교가 가능한 유효 라벨이 없어 reliability × 시간 생략")
        return
    out = pd.DataFrame(rows)
    out.to_csv(ANNOTATION_DIR / "annotator_reliability_by_time.csv", encoding="utf-8-sig", index=False)
    # 모든 pair 평균 κ를 월별로 요약
    plot = out.groupby("Month")["CohenKappa"].mean().reset_index()
    fig = go.Figure(go.Scatter(x=plot["Month"], y=plot["CohenKappa"], mode="lines+markers"))
    fig.update_layout(title="Annotation Reliability × Time (평균 Cohen's κ)", xaxis_title="연월", yaxis_title="Cohen's κ")
    figure_write(fig, ANNOTATION_DIR / "05_annotation_reliability_by_time.html")


def annotation_label_subtype_plot(df, human_label, subtype):
    if not human_label or not subtype:
        skip("human label + subtype 열이 없어 label×subtype 시각화 생략")
        return
    tab = pd.crosstab(df[human_label].map(clean).replace("", "미라벨"), df[subtype].map(clean).replace("", "미세분류없음"))
    tab.to_csv(ANNOTATION_DIR / "annotation_label_x_subtype.csv", encoding="utf-8-sig")
    fig = go.Figure(go.Heatmap(z=tab.values, x=tab.columns.astype(str), y=tab.index.astype(str), hovertemplate="Label=%{y}<br>Subtype=%{x}<br>%{z}<extra></extra>"))
    fig.update_layout(title="Annotation Label × Subtype", xaxis_title="Subtype", yaxis_title="Label")
    figure_write(fig, ANNOTATION_DIR / "06_annotation_label_x_subtype.html")


def annotation_model_metrics(df, human_label, model_label):
    if not human_label or not model_label:
        skip("Gold label + model label이 없어 모델 성능지표 생략")
        return
    y_true = df[human_label].map(clean)
    y_pred = df[model_label].map(clean)
    valid = (y_true != "") & (y_pred != "")
    if not valid.any():
        skip("Gold/Model 라벨의 유효한 겹침 데이터가 없어 모델 성능지표 생략")
        return
    acc = float((y_true[valid] == y_pred[valid]).mean())
    mf1 = macro_f1(y_true[valid], y_pred[valid])
    met = pd.DataFrame([{
        "N": int(valid.sum()), "Accuracy": acc, "MacroF1": mf1
    }])
    met.to_csv(ANNOTATION_DIR / "model_vs_gold_metrics.csv", encoding="utf-8-sig", index=False)
    fig = go.Figure(go.Bar(x=["Accuracy", "Macro F1"], y=[acc, mf1 if mf1 is not None else 0]))
    fig.update_layout(title="Gold vs Model 기본 성능지표", yaxis_title="Score", yaxis_range=[0,1])
    figure_write(fig, ANNOTATION_DIR / "07_model_vs_gold_metrics.html")


def annotation_future(df):
    annotation_dir = ANNOTATION_DIR
    annotation_dir.mkdir(exist_ok=True)

    # 이전 실행의 annotation 산출물을 먼저 정리한다.
    # 현재 실행에서 실제 데이터가 발견될 때만 다시 생성한다.
    stale_outputs = [
        "01_annotation_label_distribution.html", "02_annotator_agreement.html",
        "03_gold_vs_model_confusion.html", "04_model_confidence.html",
        "05_annotation_reliability_by_time.html", "06_annotation_label_x_subtype.html",
        "06_annotation_label_subtype.html", "07_model_vs_gold_metrics.html",
        "annotation_label_distribution.csv", "annotator_agreement.csv",
        "gold_vs_model_confusion.csv", "human_vs_ai_disagreements.csv",
        "annotator_reliability_by_time.csv", "annotation_label_x_subtype.csv",
        "model_vs_gold_metrics.csv",
    ]
    for name in stale_outputs:
        f = annotation_dir / name
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass

    human_label = first_existing(df, [
        "Gold_Label", "Final_Label", "Human_Label", "Annotation_Label",
        "어노테이션", "최종라벨", "인간라벨", "GoldLabel", "FinalLabel",
    ])
    subtype = first_existing(df, [
        "Gold_Subtype", "Final_Subtype", "Human_Subtype", "Annotation_Subtype",
        "세부라벨", "최종세부라벨",
    ])
    model_label = first_existing(df, [
        "Model_Label", "AI_Label", "GPT_Label", "ModelLabel", "AI라벨", "모델라벨",
    ])
    confidence = first_existing(df, [
        "Model_Confidence", "AI_Confidence", "GPT_Confidence", "Confidence",
        "모델신뢰도", "AI신뢰도",
    ])
    annotators = candidate_columns(df, [r"^Annotator[_ ]?\w+", r"^어노테이터", r"^라벨러", r"^annotator"])

    generated = False

    if not human_label:
        skip("인간/Gold annotation label 열이 없어 annotation label distribution 생략")
    else:
        counts = df[human_label].fillna("").map(clean).replace("", "미라벨").value_counts()
        counts.to_csv(annotation_dir / "annotation_label_distribution.csv", encoding="utf-8-sig", header=["Count"])
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values))
        fig.update_layout(title=f"Annotation Label Distribution ({human_label})", xaxis_title="Label", yaxis_title="Count")
        figure_write(fig, annotation_dir / "01_annotation_label_distribution.html")
        generated = True

    if len(annotators) >= 2:
        rows = []
        for a, b in itertools.combinations(annotators, 2):
            k = cohens_kappa(df[a], df[b])
            agreement = float((df[a].fillna("").map(clean) == df[b].fillna("").map(clean)).mean())
            rows.append({"AnnotatorA": a, "AnnotatorB": b, "Agreement": agreement, "CohenKappa": k})
        agree = pd.DataFrame(rows)
        agree.to_csv(annotation_dir / "annotator_agreement.csv", encoding="utf-8-sig", index=False)
        fig = go.Figure(go.Bar(
            x=[f"{r['AnnotatorA']} × {r['AnnotatorB']}" for r in rows],
            y=[r["CohenKappa"] if r["CohenKappa"] is not None else 0 for r in rows],
        ))
        fig.update_layout(title="Annotator Agreement: Cohen's κ", yaxis_title="κ")
        figure_write(fig, annotation_dir / "02_annotator_agreement.html")
        generated = True
    else:
        skip("2개 이상의 annotator 열이 없어 annotator agreement 생략")

    if human_label and model_label:
        conf = pd.crosstab(
            df[human_label].fillna("").map(clean).replace("", "미라벨"),
            df[model_label].fillna("").map(clean).replace("", "미라벨"),
        )
        conf.to_csv(annotation_dir / "gold_vs_model_confusion.csv", encoding="utf-8-sig")
        fig = go.Figure(go.Heatmap(z=conf.values, x=conf.columns, y=conf.index, hovertemplate="Gold=%{y}<br>Model=%{x}<br>%{z}<extra></extra>"))
        fig.update_layout(title="Gold vs Model Confusion Matrix", xaxis_title="Model", yaxis_title="Gold")
        figure_write(fig, annotation_dir / "03_gold_vs_model_confusion.html")
        mismatch = df[df[human_label].fillna("").map(clean) != df[model_label].fillna("").map(clean)].copy()
        mismatch.to_csv(annotation_dir / "human_vs_ai_disagreements.csv", encoding="utf-8-sig", index=False)
        generated = True
    else:
        skip("Gold label + model label이 모두 없어 Gold vs Model 분석 생략")

    if confidence:
        confv = pd.to_numeric(df[confidence], errors="coerce").dropna()
        if not confv.empty:
            fig = go.Figure(go.Histogram(x=confv))
            fig.update_layout(title=f"Model Confidence Distribution ({confidence})", xaxis_title="Confidence", yaxis_title="Count")
            figure_write(fig, annotation_dir / "04_model_confidence.html")
            generated = True
        else:
            skip("Confidence 열은 있으나 수치 데이터가 없어 confidence 시각화 생략")
    else:
        skip("model confidence 열이 없어 confidence 분석 생략")

    if subtype and human_label:
        annotation_label_subtype_plot(df, human_label, subtype)
        generated = True
    else:
        skip("human label + subtype 열이 없어 세부 라벨 분석 생략")

    annotation_reliability_over_time(df, annotators)
    annotation_model_metrics(df, human_label, model_label)

    if not generated:
        skip("현재 데이터에는 annotation 관련 열이 없어 annotation module은 사실상 비활성 상태입니다.")


# ------------------------------------------------------------
# 대시보드 index
# ------------------------------------------------------------


def make_dashboard(summary=None, insights=None):
    """생성된 시각화만 한 화면에 순차 배치한 통합 대시보드."""
    from html import escape
    from urllib.parse import quote

    summary = summary or {}
    insights = insights or {}

    sections = [
        ("01. 전체 흐름", [
            ("01_월별_담론량.html", "월별 담론량", "절대량"),
            ("02_월별_담론상대빈도.html", "월별 담론 상대빈도", "전체 발언량 보정"),
            ("03_6개계열_시간Heatmap.html", "6개 담론계열 × 시간", "6개 계열 변화"),
        ]),
        ("02. 어휘와 담론", [
            ("04_키워드_시간Heatmap.html", "키워드 × 시간", "시기별 키워드"),
            ("05_주요키워드_시계열.html", "주요 키워드 시계열", "핵심 어휘 추세"),
            ("06_키워드_시간구성비Heatmap.html", "키워드 시간 구성비", "키워드별 시기 집중"),
            ("07_키워드_popularity.html", "키워드 빈도", "전체 기간 분포"),
        ]),
        ("03. 정치사회적 행위자", [
            ("08_정당별_담론.html", "정당 × 담론", "정당별 구성"),
            ("09_지역별_담론.html", "지역 × 담론", "지역별 구성"),
            ("10_성별별_담론.html", "성별 × 담론", "성별별 구성"),
            ("11_당선횟수별_담론.html", "당선횟수 × 담론", "정치 경력별 구성"),
            ("12_당선방법별_담론.html", "당선방법 × 담론", "당선 방식별 구성"),
        ]),
        ("04. 의원 중심", [
            ("13_발언자_프로파일.html", "발언자 프로파일", "주요 발언자"),
            ("14_의원별_담론프로파일.html", "의원별 담론 프로파일", "의원별 담론 조합"),
            ("15_의원_담론_시간.html", "의원 × 담론 × 시간", "행위자와 시간"),
            ("16_발언자_전체발언_시간.html", "발언자 × 시간", "활동 시기"),
        ]),
        ("05. 정당 중심", [
            ("25_정당중심.html", "정당–주요 의원", "정당과 주요 의원"),
            ("26_정당별_담론프로파일.html", "정당별 담론 프로파일", "6개 담론계열"),
            ("27_정당_담론_시간.html", "정당 × 담론 × 시간", "정당별 시계열 구조"),
            ("28_정당_전체발언_시간.html", "정당 × 전체 발언 × 시간", "정당별 활동 시기"),
        ]),
        ("06. 관계·네트워크", [
            ("17_담론간_관계네트워크.html", "담론 간 관계", "담론 교차연결"),
            ("18_키워드_공출현네트워크.html", "키워드 공출현", "동일 발언의 공출현"),
            ("19_발언흐름_네트워크.html", "발언흐름", "회의 내 인접성"),
            ("20_중심성_분석.html", "중심성", "구조적 위치"),
            ("21_에고네트워크.html", "Ego Network", "국소 연결"),
            ("22_시간축_네트워크.html", "시간축 네트워크", "시간별 구조"),
            ("23_지식그래프.html", "Event/Speech 지식그래프", "통합 구조"),
            ("24_커뮤니티_탐지.html", "커뮤니티 탐지", "연결구조 군집"),
        ]),
    ]

    def chart_card(filename, title, desc, extra_class="", hero=False):
        actual = OUTPUT_DIR / filename
        if not actual.exists():
            return ""  # 없는 결과물은 카드 자체를 생성하지 않는다.
        insight = insights.get(filename, "데이터의 주요 패턴을 확인합니다.")
        hero_cls = " hero-chart" if hero else ""
        height_map = {
            "01_월별_담론량.html": 620, "02_월별_담론상대빈도.html": 620, "03_6개계열_시간Heatmap.html": 820,
            "04_키워드_시간Heatmap.html": 760, "05_주요키워드_시계열.html": 760, "06_키워드_시간구성비Heatmap.html": 820, "07_키워드_popularity.html": 820,
            "08_정당별_담론.html": 820, "09_지역별_담론.html": 820, "10_성별별_담론.html": 820, "11_당선횟수별_담론.html": 820, "12_당선방법별_담론.html": 820,
            "13_발언자_프로파일.html": 680, "14_의원별_담론프로파일.html": 760, "15_의원_담론_시간.html": 1050, "16_발언자_전체발언_시간.html": 700,
            "25_정당중심.html": 720, "26_정당별_담론프로파일.html": 760, "27_정당_담론_시간.html": 1050, "28_정당_전체발언_시간.html": 700,
            "17_담론간_관계네트워크.html": 760, "18_키워드_공출현네트워크.html": 760, "19_발언흐름_네트워크.html": 760,
            "20_중심성_분석.html": 2450, "21_에고네트워크.html": 960, "22_시간축_네트워크.html": 960, "23_지식그래프.html": 1250, "24_커뮤니티_탐지.html": 900,
        }
        h = height_map.get(filename, 760)
        return (
            f'<article class="card {extra_class}{hero_cls}">'
            f'<div class="card-head"><div><h3>{escape(title)}</h3><p>{escape(desc)}</p></div><span class="status ready">LIVE</span></div>'
            f'<div class="insight"><b>요점</b> {escape(insight)}</div>'
            f'<iframe src="{quote(filename)}" title="{escape(title)}" loading="eager" scrolling="no" style="height:{h}px"></iframe></article>'
        )

    featured_file = "02_월별_담론상대빈도.html"
    featured_html = chart_card(featured_file, "월별 담론 상대빈도", "전체 발언량 보정", "wide", hero=True)

    section_html = []
    wide_files = {"03_6개계열_시간Heatmap.html", "17_담론간_관계네트워크.html", "18_키워드_공출현네트워크.html", "19_발언흐름_네트워크.html", "20_중심성_분석.html", "21_에고네트워크.html", "22_시간축_네트워크.html", "23_지식그래프.html", "24_커뮤니티_탐지.html", "25_정당중심.html", "27_정당_담론_시간.html"}
    sec_no = 1
    for section_title, charts in sections:
        cards_html = [chart_card(fn, title, desc, "wide" if fn in wide_files else "") for fn, title, desc in charts]
        cards_html = [x for x in cards_html if x]
        if not cards_html:
            continue
        section_html.append(
            f'<section class="section" id="section-{sec_no}">'
            f'<div class="section-title"><span class="num">{sec_no:02d}</span><h2>{escape(section_title)}</h2></div>'
            f'<div class="grid">{"".join(cards_html)}</div></section>'
        )
        sec_no += 1

    ann_titles = {
        "01_annotation_label_distribution.html": ("Annotation 라벨 분포", "라벨 분포"),
        "02_annotator_agreement.html": ("Annotator agreement", "어노테이터 일치도"),
        "03_gold_vs_model_confusion.html": ("Gold vs Model", "혼동행렬"),
        "04_model_confidence.html": ("Model confidence", "모델 신뢰도"),
        "05_annotation_reliability_by_time.html": ("Reliability × Time", "시간별 신뢰도"),
        "06_annotation_label_x_subtype.html": ("Label × Subtype", "라벨과 세부유형"),
        "06_annotation_label_subtype.html": ("Label × Subtype", "라벨과 세부유형"),
        "07_model_vs_gold_metrics.html": ("Model metrics", "Accuracy / Macro-F1"),
    }
    ann_cards = []
    for name, (title, desc) in ann_titles.items():
        f = ANNOTATION_DIR / name
        if not f.exists():
            continue
        insight = insights.get(f"annotations/{name}", insights.get(name, "데이터의 주요 패턴을 확인합니다."))
        ann_cards.append(
            f'<article class="card wide"><div class="card-head"><div><h3>{escape(title)}</h3><p>{escape(desc)}</p></div><span class="status ready">LIVE</span></div>'
            f'<div class="insight"><b>요점</b> {escape(insight)}</div><iframe src="annotations/{quote(name)}" title="{escape(title)}" loading="eager" scrolling="no"></iframe></article>'
        )
    annotation_html = ''
    if ann_cards:
        annotation_html = f'<section class="section" id="section-annotation"><div class="section-title"><span class="num">A</span><h2>Annotation</h2></div><div class="grid">{"".join(ann_cards)}</div></section>'

    kpi = [
        ("발언", summary.get("rows", "—"), "분석 대상"),
        ("의원", summary.get("speakers", "—"), "식별된 발언자"),
        ("회의", summary.get("meetings", "—"), "회의/세션"),
        ("기간", f"{summary.get('start','—')} ~ {summary.get('end','—')}", "분석 범위"),
    ]
    kpi_html = ''.join(f'<div class="kpi"><div class="kpi-label">{escape(str(a))}</div><div class="kpi-value">{escape(str(b))}</div><div class="kpi-note">{escape(str(c))}</div></div>' for a,b,c in kpi)

    rows = int(summary.get("rows",0) or 0); dev = int(summary.get("development",0) or 0); bound = int(summary.get("boundary",0) or 0); both = int(summary.get("both",0) or 0)
    rel = dev + bound + both
    share = rel / rows * 100 if rows else 0
    summary_items = [
        ("전체 흐름", insights.get("SUMMARY_01", "발언량과 담론 비중의 시간 변화를 확인.")),
        ("어휘·담론", insights.get("SUMMARY_02", "핵심 어휘의 빈도와 시간적 변화를 확인.")),
        ("정치사회적 행위자", insights.get("SUMMARY_03", "행위자별 담론 분포를 확인.")),
        ("의원 중심", insights.get("SUMMARY_04", "주요 의원의 담론 프로파일을 확인.")),
        ("정당 중심", insights.get("SUMMARY_05", "정당별 담론 프로파일을 확인.")),
        ("관계·네트워크", insights.get("SUMMARY_06", "담론·키워드·발언흐름의 연결 구조를 확인.")),
    ]
    summary_text = ""

    data_links = [("network_speech_flow_edges.csv","발언흐름"),("network_keyword_cooccurrence_edges.csv","키워드 공출현"),("network_centrality.csv","중심성"),("network_knowledge_graph_edges.csv","지식그래프"),("validation_network.csv","검증"),("logs/run.log","로그")]
    data_html = ''.join(f'<a class="data-link" href="{quote(fn)}">{escape(label)}</a>' for fn,label in data_links if (OUTPUT_DIR/fn).exists())

    text=f'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>이승만 시기 국회회의록 — 통합 분석 대시보드</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--line:#dde2e7;--text:#18212b;--muted:#687482;--accent:#263f55;--soft:#eef2f5;--ok:#2f6b4f;--insight:#f7f9fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;line-height:1.45}}
.page{{max-width:1780px;margin:0 auto;padding:26px 26px 60px}}
.hero{{background:linear-gradient(135deg,#172736,#314b60);color:#fff;border-radius:22px;padding:30px 36px 26px;box-shadow:0 8px 30px rgba(20,35,50,.14)}} .hero h1{{margin:0;font-size:29px;letter-spacing:-.03em}} .hero p{{margin:7px 0 0;color:#dce6ed;font-size:13.5px}}
.executive{{margin-top:13px;background:#fff;border:1px solid var(--line);border-radius:16px;padding:15px 19px;box-shadow:0 3px 14px rgba(30,45,60,.045)}} .executive-label{{font-size:10px;font-weight:800;color:var(--accent);letter-spacing:.08em;text-transform:uppercase}} .executive h2{{font-size:17px;margin:4px 0 8px}} .summary-list{{display:grid;grid-template-columns:1fr 1fr;gap:5px 22px}} .summary-row{{display:grid;grid-template-columns:110px 1fr;gap:8px;font-size:12px;color:#35424e;padding:4px 0;border-bottom:1px solid #eef1f3}} .summary-row b{{color:var(--accent);white-space:nowrap}}
.featured{{margin-top:14px}} .section{{margin-top:29px}} .section-title{{display:flex;align-items:center;gap:11px;margin-bottom:10px}} .section-title .num{{width:33px;height:33px;border-radius:9px;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:11px;flex:0 0 auto}} .section-title h2{{margin:0;font-size:19px;letter-spacing:-.02em}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;min-width:0;box-shadow:0 2px 10px rgba(30,45,60,.035)}} .card.wide{{grid-column:1/-1}} .hero-chart{{box-shadow:0 6px 22px rgba(30,45,60,.07);border-color:#cdd7df}}
.card-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:11px 14px 7px;border-bottom:1px solid #edf0f2;min-height:56px}} .card-head h3{{font-size:15px;margin:0;font-weight:800}} .card-head p{{font-size:11px;color:var(--muted);margin:2px 0 0}} .status{{font-size:8px;border:1px solid #c8ded0;color:var(--ok);padding:3px 7px;border-radius:999px;background:#f2f8f4}}
.insight{{padding:9px 14px 10px;background:#eef4f8;border-left:4px solid #4b6b82;border-bottom:1px solid #dce5eb;font-size:12.5px;color:#263844;min-height:38px;line-height:1.5}} .insight b{{color:#1f4660;margin-right:7px;font-size:13.5px}} iframe{{display:block;width:100%;height:500px;overflow:hidden;border:0;background:#fff}} .card.wide iframe{{height:560px}} .hero-chart iframe{{height:560px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin:12px 0 4px}} .kpi{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:11px 13px;min-height:88px}} .kpi-label{{font-size:10px;color:var(--muted);font-weight:700}} .kpi-value{{font-size:20px;font-weight:800;margin:3px 0 2px}} .kpi-note{{font-size:10px;color:var(--muted)}}
.data-strip{{margin-top:26px;background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:12px;display:flex;flex-wrap:wrap;gap:7px;align-items:center}} .data-strip .label{{font-size:9px;font-weight:800;color:var(--muted)}} .data-link{{text-decoration:none;color:var(--accent);border:1px solid var(--line);background:var(--soft);border-radius:999px;padding:5px 8px;font-size:9px}} .footer{{margin-top:14px;color:var(--muted);font-size:9px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}}
@media(max-width:1200px){{.page{{padding:20px 15px 44px}}.grid{{grid-template-columns:1fr}}.card.wide{{grid-column:auto}}.kpis{{grid-template-columns:repeat(2,1fr)}} .summary-list{{grid-template-columns:1fr}}}}
@media(max-width:700px){{.hero{{padding:22px 18px}}.hero h1{{font-size:21px}}.kpis{{grid-template-columns:1fr 1fr}} .summary-row{{grid-template-columns:1fr}}}}
</style></head>
<body><main class="page">
<header class="hero"><h1>이승만 시기 국회회의록 — 통합 담론·네트워크 분석</h1><p>발전·경계 담론, 키워드, 의원 특성, 발언 이벤트, 네트워크를 한 화면에서 읽습니다.</p></header>
<section class="executive"><div class="executive-label">SUMMARY</div><h2>요약</h2><div class="summary-list">{''.join(f'<div class="summary-row"><b>{escape(k)}</b><span>{escape(v)}</span></div>' for k,v in summary_items)}</div></section>
<section class="featured"><div class="grid">{featured_html}</div></section>
<div class="kpis">{kpi_html}</div>
{''.join(section_html)}
{annotation_html}
<div class="data-strip"><span class="label">DATA</span>{data_html}</div>
<div class="footer"><span>분석 범위: {escape(str(summary.get('start','—')))} ~ {escape(str(summary.get('end','—')))}</span><span>발언(Event) 중심 · 핵심 네트워크 노드만 라벨 표시</span></div>
<script>
function fitDashboardFrames(){{
  document.querySelectorAll("iframe").forEach(function(frame){{
    function fit(){{ try {{ var d=frame.contentDocument; if(!d) return; var h=Math.max(d.body?d.body.scrollHeight:0,d.documentElement?d.documentElement.scrollHeight:0); if(h>80) frame.style.height=(h+14)+"px"; }} catch(e){{}} }}
    function observe(){{ try {{ var d=frame.contentDocument; if(!d||!d.body) return; if(frame._ro) frame._ro.disconnect(); if(d.defaultView&&d.defaultView.ResizeObserver){{ frame._ro=new d.defaultView.ResizeObserver(fit); frame._ro.observe(d.body); if(d.documentElement) frame._ro.observe(d.documentElement); }} }} catch(e){{}} fit(); }}
    frame.addEventListener("load",observe); setTimeout(observe,250); setTimeout(observe,1000); setTimeout(observe,2500);
  }});
}}
document.addEventListener("DOMContentLoaded",fitDashboardFrames);
window.addEventListener("resize",fitDashboardFrames);
</script>
</main></body></html>'''
    (OUTPUT_DIR / "대시보드.html").write_text(text, encoding="utf-8")


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------


def main():
    log("=" * 72)
    log("🚀 이승만 시기 국회회의록 고급 담론·네트워크 분석 시작")
    log("=" * 72)

    df = load_data()

    monthly = monthly_stats(df)
    family_counts, family_ratio = family_monthly_stats(df)
    keyword_month = keyword_monthly_stats(df)
    keyword_normalized = keyword_temporal_normalized_stats(keyword_month)
    group_tables = {dim: group_stats(df, dim) for dim in METADATA_DIMENSIONS}
    speakers = speaker_stats(df)
    dev_keywords, boundary_keywords = keyword_stats(df)

    # CSV
    monthly.to_csv(OUTPUT_DIR / "통계_월별.csv", encoding="utf-8-sig", index=False)
    family_counts.to_csv(OUTPUT_DIR / "통계_6개계열_월별_절대량.csv", encoding="utf-8-sig")
    family_ratio.to_csv(OUTPUT_DIR / "통계_6개계열_월별_상대빈도.csv", encoding="utf-8-sig")
    keyword_month.to_csv(OUTPUT_DIR / "통계_키워드_시간.csv", encoding="utf-8-sig")
    keyword_normalized.to_csv(OUTPUT_DIR / "통계_키워드_시간구성비.csv", encoding="utf-8-sig")
    for dim, table in group_tables.items():
        table.to_csv(OUTPUT_DIR / f"통계_{dim}.csv", encoding="utf-8-sig")
    speakers.to_csv(OUTPUT_DIR / "통계_발언자.csv", encoding="utf-8-sig")

    # 시각화
    plot_monthly(monthly)
    plot_family_heatmap(family_ratio)
    plot_keyword_heatmap(keyword_month)
    plot_keyword_trends(keyword_month)
    plot_keyword_temporal_normalized(keyword_month, keyword_normalized)
    plot_keyword_popularity(keyword_month)
    for i, dim in enumerate(METADATA_DIMENSIONS, start=8):
        plot_group(group_tables[dim], dim, i)
    plot_speaker_profile(speakers)
    plot_speaker_discourse_profile(df, speakers)
    plot_speaker_discourse_time(df, speakers)
    plot_speaker_time(df, speakers)
    party_center_analysis(df)

    # network
    cross_graph = build_discourse_cross_network(df)
    keyword_graph, _ = build_keyword_network(df)
    flow_graph, _ = build_speech_flow_network(df)
    centrality_df = centrality_analysis(flow_graph)
    ego_network_analysis(flow_graph, centrality_df)
    timeline_network(df, flow_graph)
    knowledge_graph = build_knowledge_graph(df, speakers, dev_keywords, boundary_keywords)
    community_df = community_detection_analysis(flow_graph)

    # validation
    network_validation(df)

    # future annotations
    annotation_future(df)

    # summary
    summary = {
        "rows": int(len(df)),
        "speakers": int(df.loc[df["SpeakerKey"] != "", "SpeakerKey"].nunique()),
        "meetings": int(df["MeetingID"].nunique()),
        "development": int((df["ExclusiveClass"] == "발전").sum()),
        "boundary": int((df["ExclusiveClass"] == "경계").sum()),
        "both": int((df["ExclusiveClass"] == "발전+경계").sum()),
        "unclassified": int((df["ExclusiveClass"] == "미분류").sum()),
        "start": str(df["Date"].min().date()) if not df.empty else None,
        "end": str(df["Date"].max().date()) if not df.empty else None,
    }
    insights = build_dashboard_insights(
        df, monthly, family_ratio, keyword_month, keyword_normalized, group_tables, speakers,
        cross_graph, keyword_graph, flow_graph, centrality_df, knowledge_graph
    )
    json_dump(summary, OUTPUT_DIR / "summary.json")
    json_dump(insights, OUTPUT_DIR / "dashboard_insights.json")
    make_dashboard(summary, insights)

    log("=" * 72)
    log("✅ 완료")
    log(f"출력 폴더: {OUTPUT_DIR.resolve()}")
    log(f"대시보드: {(OUTPUT_DIR / '대시보드.html').resolve()}")
    log("=" * 72)
    write_log()


if __name__ == "__main__":
    main()
