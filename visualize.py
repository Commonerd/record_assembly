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
6. Degree / Betweenness 중심성 추가
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
  20_중심성_분석.html
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
    insights = {}

    if not monthly.empty:
        rel_cols=[c for c in ["발전","경계","발전+경계"] if c in monthly.columns]
        if rel_cols:
            m=monthly.assign(관련=monthly[rel_cols].sum(axis=1)).sort_values("관련", ascending=False).iloc[0]
            dominant=max(rel_cols, key=lambda c: m[c])
            insights["01_월별_담론량.html"] = f"관련 담론 발언량이 가장 높은 시점은 {_short_date(m['Month'])}이며, 그 달에는 {dominant} 발언이 가장 많았습니다 ({int(m[dominant])}건)."
        if "발전_전체대비비중" in monthly.columns and "경계_전체대비비중" in monthly.columns:
            tmp=monthly.assign(관련비중=monthly["발전_전체대비비중"]+monthly["경계_전체대비비중"]).sort_values("관련비중", ascending=False).iloc[0]
            insights["02_월별_담론상대빈도.html"] = f"전체 발언량을 보정하면 {_short_date(tmp['Month'])}의 관련 담론 비중이 가장 높습니다({_pct(tmp['관련비중'])}); 절대량과 다른 집중 시점을 보여줍니다."
            total_class={c:int(monthly[c].sum()) for c in ["발전","경계","발전+경계"] if c in monthly.columns}
            if total_class:
                top_cls=max(total_class,key=total_class.get)
                insights["02_월별_담론상대빈도.html"] += f" 누적 발언이 가장 많은 분류는 {top_cls}({total_class[top_cls]:,}건)입니다."

    if not family_ratio.empty:
        stacked=family_ratio.stack().sort_values(ascending=False)
        if not stacked.empty:
            (fam, month), val = stacked.index[0], float(stacked.iloc[0])
            insights["03_6개계열_시간Heatmap.html"] = f"6개 계열 중 {_short_date(month)}의 {fam} 계열 밀도가 가장 높았습니다(1,000발언당 {val:.2f}건)."

    if not keyword_month.empty:
        top_cell=keyword_month.stack().sort_values(ascending=False)
        if not top_cell.empty:
            (kw, month), val=top_cell.index[0], float(top_cell.iloc[0])
            insights["04_키워드_시간Heatmap.html"] = f"최대 키워드 관측은 {_short_date(month)}의 ‘{kw}’로, 해당 시점에 {int(val)}개 발언에서 확인됩니다."
        totals=keyword_month.sum(axis=1).sort_values(ascending=False)
        if not totals.empty:
            kw=totals.index[0]
            peak_month=keyword_month.loc[kw].idxmax()
            insights["05_주요키워드_시계열.html"] = f"전체 기간에서 가장 자주 관측된 키워드는 ‘{kw}’이며, 최고점은 {_short_date(peak_month)}입니다."
            top3=", ".join([f"{k}({int(v)})" for k,v in totals.head(3).items()])
            insights["07_키워드_popularity.html"] = f"상위 3개 키워드는 {top3}이며, 전체 담론을 구성하는 핵심 어휘입니다."

    if not keyword_normalized.empty:
        concentration=keyword_normalized.max(axis=1).sort_values(ascending=False)
        if not concentration.empty:
            kw=concentration.index[0]; month=keyword_normalized.loc[kw].idxmax(); val=float(concentration.loc[kw])
            insights["06_키워드_시간구성비Heatmap.html"] = f"‘{kw}’는 전체 관측량의 {val:.1f}%가 {_short_date(month)}에 집중되어 시기 특이성이 가장 강합니다."

    dim_files={
        "정당":"08_정당별_담론.html", "지역":"09_지역별_담론.html", "성별":"10_성별별_담론.html",
        "당선횟수":"11_당선횟수별_담론.html", "당선방법":"12_당선방법별_담론.html"
    }
    for dim, fn in dim_files.items():
        tab=group_tables.get(dim)
        if tab is not None and not tab.empty:
            cand=tab[tab["전체"]>=max(3, int(len(df)*0.03))]
            if cand.empty: cand=tab
            row=cand.sort_values(["관련비중","전체"], ascending=False).iloc[0]
            insights[fn]=f"{dim}별로는 ‘{row.name}’ 집단의 관련 담론 비중이 가장 높았습니다({_pct(row['관련비중'])}, 전체 {int(row['전체'])}발언)."

    if not speakers.empty:
        row=speakers.sort_values(["관련발언","관련비중"], ascending=False).iloc[0]
        insights["13_발언자_프로파일.html"] = f"관련 담론 발언 수가 가장 많은 주요 발언자는 {row['Speaker']}이며, 전체 {int(row['전체발언'])}발언 가운데 {int(row['관련발언'])}건이 관련 담론입니다."
        st=df[df["SpeakerKey"]==row.name]
        fam_counts={fam:int((st["발전계열" if FAMILY_TO_DISCOURSE[fam]=="발전" else "경계계열"].apply(lambda x:fam in split_semicolon(x))).sum()) for fam in FAMILIES}
        if fam_counts:
            fam=max(fam_counts,key=fam_counts.get)
            insights["14_의원별_담론프로파일.html"] = f"주요 의원들의 프로파일에서는 ‘{fam}’ 계열이 가장 자주 나타나며, 의원별 담론 조합 차이를 비교할 수 있습니다."
        insights["15_의원_담론_시간.html"] = f"{row['Speaker']}의 관련 담론은 {st['Month'].nunique()}개 연월에 걸쳐 관측되어 집중과 지속 여부를 비교할 수 있습니다."
        insights["16_발언자_전체발언_시간.html"] = f"전체 발언량 기준 주요 발언자 중 {row['Speaker']}가 관련 담론 발언을 가장 많이 남겼습니다."

    if cross_graph is not None and getattr(cross_graph,'number_of_edges',lambda:0)()>0:
        e=max(cross_graph.edges(data=True), key=lambda x:x[2].get('weight',0))
        insights["17_담론간_관계네트워크.html"] = f"가장 강한 담론 교차연결은 ‘{e[0]}–{e[1]}’이며 동일 발언에서 {int(e[2].get('weight',0))}회 함께 나타났습니다."
    if keyword_graph is not None and getattr(keyword_graph,'number_of_edges',lambda:0)()>0:
        e=max(keyword_graph.edges(data=True), key=lambda x:x[2].get('weight',0))
        insights["18_키워드_공출현네트워크.html"] = f"가장 강한 키워드 공출현은 ‘{e[0]}–{e[1]}’이며 동일 발언에서 {int(e[2].get('weight',0))}회 연결됩니다."
    if flow_graph is not None and getattr(flow_graph,'number_of_edges',lambda:0)()>0:
        e=max(flow_graph.edges(data=True), key=lambda x:x[2].get('weight',0))
        insights["19_발언흐름_네트워크.html"] = f"회의 내 연속 발언 인접성이 가장 높은 쌍은 ‘{e[0]}–{e[1]}’입니다({int(e[2].get('weight',0))}회). 이는 직접 상호작용을 의미하지 않습니다."
    if centrality_df is not None and not centrality_df.empty:
        row=centrality_df.sort_values("BetweennessCentrality", ascending=False).iloc[0]
        insights["20_중심성_분석.html"] = f"관찰된 발언흐름 네트워크에서 Betweenness가 가장 높은 발언자는 {row['SpeakerKey']}이며, 연결 영역 사이의 구조적 위치가 상대적으로 높습니다."
        insights["21_에고네트워크.html"] = f"Ego Network는 {row['SpeakerKey']}처럼 중심성이 높은 발언자를 중심으로 주변 연결을 좁혀 보여줍니다."
    if flow_graph is not None and getattr(flow_graph,'number_of_edges',lambda:0)()>0:
        ys=[]
        for y,part in df.groupby("Year"):
            part=part.sort_values(["MeetingID","SeqNum","SpeechID"])
            n=0
            for _,g in part.groupby("MeetingID"):
                ss=[x for x in g["SpeakerKey"] if x]
                n += sum(a!=b for a,b in zip(ss,ss[1:]))
            ys.append((int(y),n))
        if ys:
            y,n=max(ys,key=lambda x:x[1])
            insights["22_시간축_네트워크.html"] = f"상위 발언자 네트워크의 시간적 연결 밀도는 {y}년에 가장 높게 관측되어 이 시기를 중심으로 구조 변화를 볼 수 있습니다."
    if knowledge_graph is not None:
        insights["23_지식그래프.html"] = f"발언(Event)을 중심으로 의원·메타데이터·담론·키워드를 연결한 구조이며 현재 {knowledge_graph.number_of_nodes():,}개 노드와 {knowledge_graph.number_of_edges():,}개 관계를 담고 있습니다."

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
        text=s["Speaker"], textposition="top center",
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
    if speakers.empty:
        skip("발언자가 없어 의원×담론×시간 분석 생략")
        return
    top = list(speakers.head(12).index)
    sub = df[df["SpeakerKey"].isin(top)].copy()
    months = sorted(sub["Month"].unique())
    matrices = {}
    labels = []
    for fam in FAMILIES:
        col = "발전계열" if FAMILY_TO_DISCOURSE[fam] == "발전" else "경계계열"
        mask = sub[col].apply(lambda x: fam in split_semicolon(x))
        piv = pd.crosstab(sub.loc[mask, "SpeakerKey"], sub.loc[mask, "Month"]).reindex(index=top, columns=months, fill_value=0)
        matrices[fam] = piv
        labels.append(fam)

    first = labels[0]
    fig = go.Figure(go.Heatmap(
        x=months, y=[speakers.loc[sp, "Speaker"] for sp in top],
        z=matrices[first].values,
        hovertemplate="%{y}<br>%{x}<br>발언 수=%{z}<extra></extra>",
    ))
    fig.update_layout(
        title=f"의원 × {first} × 시간", xaxis_title="연월", yaxis_title="의원",
        updatemenus=[{
            "type": "dropdown", "x": 1.0, "y": 1.14, "xanchor": "right",
            "buttons": [
                {"label": fam, "method": "restyle", "args": [{"z": [matrices[fam].values], "zauto": True}, {"title": f"의원 × {fam} × 시간"}]}
                for fam in labels
            ]
        }]
    )
    figure_write(fig, OUTPUT_DIR / "15_의원_담론_시간.html")


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
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes], y=[pos[n][1] for n in G.nodes],
        text=list(G.nodes), mode="markers+text", textposition="top center",
        marker=dict(size=[15 + math.sqrt(sum(d.get("weight", 0) for _, _, d in G.edges(n, data=True))) * 1.4 for n in G.nodes]),
        hovertemplate=[f"{n}<br>담론 교차연결={sum(d.get('weight', 0) for _, _, d in G.edges(n, data=True))}<extra></extra>" for n in G.nodes]
    )
    edge_ann = [dict(
        x=(pos[u][0]+pos[v][0])/2, y=(pos[u][1]+pos[v][1])/2,
        text=str(d["weight"]), showarrow=False, font=dict(size=11)
    ) for u, v, d in G.edges(data=True)]
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(title="6개 담론계열의 발언 내 교차연결", xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), annotations=edge_ann)
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
    label_nodes = {n for n, _ in sorted(((n, G.nodes[n]["count"]) for n in G.nodes), key=lambda x:x[1], reverse=True)[:10]}
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=[n if n in label_nodes else "" for n in G.nodes], textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=[min(19, 8 + math.sqrt(G.nodes[n]["count"]) * 1.8) for n in G.nodes], opacity=0.82),
        hovertemplate="%{hovertext}<extra></extra>", hovertext=node_text,
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(title="키워드 공출현 네트워크", xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), showlegend=False)
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
    for sp, n in ordered["SpeakerKey"].value_counts().items():
        G.add_node(sp, speech_count=int(n))
    for (u, v), w in edge_counter.items():
        G.add_edge(u, v, weight=int(w))

    # top-node subgraph for readability
    top_nodes = {n for n, _ in Counter(ordered["SpeakerKey"]).most_common(TOP_NETWORK_NODES)}
    H = G.subgraph(top_nodes).copy()

    if len(H.nodes) < 2:
        skip("발언흐름 네트워크의 주요 발언자가 2명 미만")
        return H, pd.DataFrame()

    pos = nx.spring_layout(H, seed=42, weight="weight")
    edge_x, edge_y = [], []
    hover_edges = []
    for u, v, d in H.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        hover_edges.append(d.get("weight", 1))

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none", showlegend=False)
    label_nodes = {n for n, _ in sorted(((n, H.nodes[n]["speech_count"]) for n in H.nodes), key=lambda x:x[1], reverse=True)[:10]}
    node_trace = go.Scatter(
        x=[pos[n][0] for n in H.nodes],
        y=[pos[n][1] for n in H.nodes],
        text=[n if n in label_nodes else "" for n in H.nodes],
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=[min(21, 8 + math.sqrt(H.nodes[n]["speech_count"]) * 1.8) for n in H.nodes], opacity=0.82),
        hovertemplate=[f"{n}<br>발언 수={H.nodes[n]['speech_count']}<extra></extra>" for n in H.nodes],
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="발언흐름 네트워크 (연속 발언 인접성)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        annotations=[dict(
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
    return H, edge_df


# ------------------------------------------------------------
# Centrality
# ------------------------------------------------------------


def centrality_analysis(G):
    if G is None or nx is None or len(G.nodes) < 2:
        skip("중심성 분석에 사용할 네트워크가 없습니다.")
        return pd.DataFrame()
    degree = nx.degree_centrality(G)
    between = nx.betweenness_centrality(G, weight=None, normalized=True)
    rows = []
    for n in G.nodes:
        rows.append({
            "SpeakerKey": n,
            "SpeechCount": G.nodes[n].get("speech_count", 0),
            "DegreeCentrality": degree.get(n, 0),
            "BetweennessCentrality": between.get(n, 0),
        })
    dfc = pd.DataFrame(rows).sort_values(["BetweennessCentrality", "DegreeCentrality"], ascending=False)
    dfc.to_csv(OUTPUT_DIR / "network_centrality.csv", encoding="utf-8-sig", index=False)

    label_nodes = set(dfc.sort_values("BetweennessCentrality", ascending=False).head(10)["SpeakerKey"])
    fig = go.Figure(go.Scatter(
        x=dfc["DegreeCentrality"], y=dfc["BetweennessCentrality"],
        mode="markers+text", text=[x if x in label_nodes else "" for x in dfc["SpeakerKey"]], textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=np.minimum(20, np.maximum(7, np.sqrt(dfc["SpeechCount"].values) * 1.7)), opacity=0.78),
        customdata=dfc["SpeakerKey"],
        hovertemplate="%{customdata}<br>Degree=%{x:.4f}<br>Betweenness=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(title="구조적 중심성: Degree × Betweenness", xaxis_title="Degree Centrality", yaxis_title="Betweenness Centrality")
    figure_write(fig, OUTPUT_DIR / "20_중심성_분석.html")
    return dfc


# ------------------------------------------------------------
# Ego network
# ------------------------------------------------------------


def ego_network_analysis(G, centrality_df):
    if G is None or nx is None or centrality_df.empty:
        skip("에고 네트워크 생성에 필요한 데이터가 없습니다.")
        return
    top = list(centrality_df.head(8)["SpeakerKey"])
    figs = []
    for center in top:
        if center not in G:
            continue
        ego = nx.ego_graph(G, center, radius=1)
        pos = nx.spring_layout(ego, seed=42)
        ex, ey = [], []
        for u, v in ego.edges:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ex += [x0, x1, None]; ey += [y0, y1, None]
        edge_trace = go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none", showlegend=False)
        label_nodes = {center} | {n for n, _ in sorted(ego.degree, key=lambda x:x[1], reverse=True)[:5]}
        node_trace = go.Scatter(
            x=[pos[n][0] for n in ego.nodes], y=[pos[n][1] for n in ego.nodes],
            text=[n if n in label_nodes else "" for n in ego.nodes], mode="markers+text", textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=[18 if n == center else 8 for n in ego.nodes], opacity=0.82),
            hovertext=list(ego.nodes), hovertemplate="%{hovertext}<extra></extra>",
        )
        fig = go.Figure([edge_trace, node_trace])
        fig.update_layout(title=f"Ego Network: {center}", xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
        figs.append((center, fig))

    # 여러 개의 figure를 한 HTML에 직렬화
    parts = [
        '<html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head><body>'
    ]
    for i, (name, fig) in enumerate(figs):
        inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"ego_{i}")
        parts.append(f"<h2>{name}</h2>{inner}")
    parts.append("</body></html>")
    (OUTPUT_DIR / "21_에고네트워크.html").write_text("\n".join(parts), encoding="utf-8")


# ------------------------------------------------------------
# Timeline network
# ------------------------------------------------------------


def timeline_network(df, flow_graph):
    if flow_graph is None or nx is None:
        skip("시간축 네트워크에 사용할 flow graph가 없습니다.")
        return

    # 전체 graph에서 핵심 발언자만 고정하여 모든 연도에서 동일 좌표 사용
    top = [n for n, _ in sorted(flow_graph.degree, key=lambda x: x[1], reverse=True)[:TOP_NETWORK_NODES]]
    G = flow_graph.subgraph(top).copy()
    if len(G.nodes) < 2:
        skip("시간축 네트워크용 주요 노드가 부족합니다.")
        return

    pos = nx.spring_layout(G, seed=42, weight="weight")
    years = sorted(int(y) for y in df["Year"].dropna().unique())

    # 연도별 실제 edge를 계산
    frame_data = []
    all_speakers = set(G.nodes)
    for year in years:
        part = df[(df["Year"] == year) & (df["SpeakerKey"].isin(all_speakers))].copy()
        part = part.sort_values(["MeetingID", "SeqNum"])
        edges = Counter()
        for _, meeting in part.groupby("MeetingID", sort=False):
            speakers = [s for s in meeting["SpeakerKey"] if s]
            for a, b in zip(speakers, speakers[1:]):
                if a == b:
                    continue
                u, v = sorted([a, b])
                edges[(u, v)] += 1

        ex, ey = [], []
        for (u, v), w in edges.items():
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]

        active_counts = part["SpeakerKey"].value_counts()
        label_nodes = set(top[:10])
        nx_trace = go.Scatter(
            x=[pos[n][0] for n in all_speakers if n in pos],
            y=[pos[n][1] for n in all_speakers if n in pos],
            text=[n if n in label_nodes else "" for n in all_speakers if n in pos],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=[min(18, 7 + math.sqrt(active_counts.get(n, 0)) * 2.1) for n in all_speakers if n in pos], opacity=0.8),
            hovertemplate=[f"{n}<br>{year}년 발언={active_counts.get(n,0)}<extra></extra>" for n in all_speakers if n in pos],
            showlegend=False,
        )
        edge_trace = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1), hoverinfo="none", showlegend=False)
        frame_data.append(go.Frame(data=[edge_trace, nx_trace], name=str(year)))

    if not frame_data:
        skip("연도별 frame을 만들 수 없습니다.")
        return

    first = frame_data[0]
    fig = go.Figure(data=first.data, frames=frame_data)
    fig.update_layout(
        title="시간축 네트워크: 연도별 발언흐름 구조",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        updatemenus=[{
            "type": "buttons", "showactive": True,
            "buttons": [{"label": "▶ 재생", "method": "animate", "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}]}],
            "x": 0, "y": 1.08,
        }],
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "연도: "},
            "steps": [{"label": str(y), "method": "animate", "args": [[str(y)], {"mode": "immediate", "frame": {"duration": 250, "redraw": True}, "transition": {"duration": 0}}]} for y in years]
        }],
        annotations=[dict(text="동일한 node 좌표를 고정하여 연도별 구조 변화를 비교", x=0, y=-0.05, xref="paper", yref="paper", showarrow=False)],
    )
    figure_write(fig, OUTPUT_DIR / "22_시간축_네트워크.html")


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
            label_nodes = set(sorted(ns, key=lambda n: len(G[n]), reverse=True)[:10])
        elif typ == "담론계열":
            label_nodes = set(ns)
        elif typ == "키워드":
            label_nodes = set(sorted(ns, key=lambda n: G.degree(n), reverse=True)[:12])
        else:
            label_nodes = set()
        traces.append(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
            text=[G.nodes[n]["label"] if n in label_nodes else "" for n in ns],
            mode="markers+text",
            textposition="top center", name=typ,
            textfont=dict(size=8 if typ not in {"의원","담론계열"} else 9),
            marker=dict(size=[11 if typ == "의원" else 9 if typ == "담론계열" else 5 for _ in ns], opacity=0.78),
            hovertemplate=[f"{G.nodes[n]['label']}<br>type={typ}<extra></extra>" for n in ns],
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title="Event/Speech 중심 지식그래프",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        annotations=[dict(
            text="발언(Event)을 중심에 두고 의원·정당·지역·성별·당선정보·담론·키워드를 연결. 직접 상호작용 관계는 포함하지 않음.",
            x=0, y=-0.06, xref="paper", yref="paper", showarrow=False,
        )],
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
    """모든 핵심 시각화를 한 페이지에 직접 임베드한 통합 대시보드.

    핵심 결과를 최상단에 고정하고, 각 카드에 데이터 기반 한 줄 인사이트를 표시한다.
    """
    from html import escape
    from urllib.parse import quote

    summary = summary or {}
    insights = insights or {}

    sections = [
        ("01. 전체 흐름", [
            ("01_월별_담론량.html", "월별 담론 절대량", "전체 발언 규모와 담론량의 기본 추세"),
            ("03_6개계열_시간Heatmap.html", "6개 담론계열 × 시간", "기술·소득·노동·이민·이주·외국인의 장기 변화"),
        ]),
        ("02. 어휘와 담론의 변화", [
            ("04_키워드_시간Heatmap.html", "키워드 × 시간", "실제 발언 어휘의 시기별 출현 변화"),
            ("05_주요키워드_시계열.html", "주요 키워드 시계열", "핵심 키워드의 장기 추세"),
            ("06_키워드_시간구성비Heatmap.html", "키워드 시간 구성비", "각 키워드의 시간적 집중·분산 패턴"),
            ("07_키워드_popularity.html", "키워드 빈도 구조", "전체 기간의 주요 키워드 분포"),
        ]),
        ("03. 정치사회적 행위자", [
            ("08_정당별_담론.html", "정당 × 담론", "정당별 담론 구성"),
            ("09_지역별_담론.html", "지역 × 담론", "지역별 담론 구성"),
            ("10_성별별_담론.html", "성별 × 담론", "성별별 담론 구성"),
            ("11_당선횟수별_담론.html", "당선횟수 × 담론", "정치 경력별 담론 구성"),
            ("12_당선방법별_담론.html", "당선방법 × 담론", "당선 방식별 담론 구성"),
        ]),
        ("04. 의원 중심 분석", [
            ("13_발언자_프로파일.html", "발언자 프로파일", "주요 의원의 전체 발언 구조"),
            ("14_의원별_담론프로파일.html", "의원별 담론 프로파일", "의원별 6개 담론계열 조합"),
            ("15_의원_담론_시간.html", "의원 × 담론 × 시간", "행위자와 담론 변화의 결합"),
            ("16_발언자_전체발언_시간.html", "발언자 × 시간", "주요 의원의 활동 시기"),
        ]),
        ("05. 관계·네트워크", [
            ("17_담론간_관계네트워크.html", "담론 간 관계 네트워크", "발언 수준에서 함께 결합되는 담론계열"),
            ("18_키워드_공출현네트워크.html", "키워드 공출현 네트워크", "동일 발언 안에서 함께 등장하는 핵심 어휘"),
            ("19_발언흐름_네트워크.html", "발언흐름 네트워크", "회의 내 발언 순서의 구조적 인접성"),
            ("20_중심성_분석.html", "중심성 분석", "Degree / Betweenness의 구조적 위치"),
            ("21_에고네트워크.html", "Ego Network", "핵심 의원 또는 노드 주변의 국소적 연결"),
            ("22_시간축_네트워크.html", "시간축 네트워크", "시간 변화에 따른 네트워크 구조"),
            ("23_지식그래프.html", "Event/Speech 중심 지식그래프", "의원·발언·담론·키워드·메타데이터의 통합 구조"),
        ]),
    ]

    def chart_card(filename, title, desc, extra_class="", hero=False):
        actual = OUTPUT_DIR / filename
        insight = insights.get(filename, "데이터에서 확인되는 핵심 패턴을 이 시각화로 확인합니다.")
        if not actual.exists():
            return f"<article class=\"card missing {extra_class}\"><div class=\"card-head\"><div><h3>{escape(title)}</h3><p>{escape(desc)}</p></div><span class=\"status\">SKIP</span></div><div class=\"insight skip-insight\"><b>해석 포인트</b> {escape(insight)}</div><div class=\"missing-box\">{escape(filename)} 파일이 없습니다.</div></article>"
        src = quote(filename)
        hero_cls = " hero-chart" if hero else ""
        return f"<article class=\"card {extra_class}{hero_cls}\"><div class=\"card-head\"><div><h3>{escape(title)}</h3><p>{escape(desc)}</p></div><span class=\"status ready\">VIEW</span></div><div class=\"insight\"><b>핵심 포인트</b> {escape(insight)}</div><iframe src=\"{src}\" title=\"{escape(title)}\" loading=\"lazy\"></iframe></article>"

    featured_file="02_월별_담론상대빈도.html"
    featured_html=chart_card(featured_file, "핵심 결과 · 월별 담론 상대빈도", "전체 발언량 차이를 보정한 담론 비중 변화", "wide", hero=True)

    section_html=[]
    wide_files={"03_6개계열_시간Heatmap.html","17_담론간_관계네트워크.html","18_키워드_공출현네트워크.html","19_발언흐름_네트워크.html","20_중심성_분석.html","22_시간축_네트워크.html","23_지식그래프.html"}
    for idx,(section_title,charts) in enumerate(sections,start=1):
        cards_html=[chart_card(fn,title,desc,"wide" if fn in wide_files else "") for fn,title,desc in charts]
        section_html.append(f"<section class=\"section\" id=\"section-{idx}\"><div class=\"section-title\"><span class=\"num\">{idx:02d}</span><div><h2>{escape(section_title)}</h2><p>위에서 아래로 읽으면 통계 → 어휘 → 행위자 → 관계 구조 순서로 해석됩니다.</p></div></div><div class=\"grid\">{''.join(cards_html)}</div></section>")

    ann_files=sorted(ANNOTATION_DIR.glob("*.html")) if ANNOTATION_DIR.exists() else []
    if ann_files:
        ann_titles={
            "01_annotation_label_distribution.html":("Annotation Label Distribution","인간 어노테이션 라벨 분포"),
            "02_annotator_agreement.html":("Annotator Agreement","어노테이터 간 일치도"),
            "03_gold_vs_model_confusion.html":("Gold vs Model","Human/Gold와 모델 예측의 혼동행렬"),
            "04_model_confidence.html":("Model Confidence","모델 confidence 분포"),
            "05_annotation_reliability_by_time.html":("Reliability × Time","시간에 따른 annotation reliability"),
            "06_annotation_label_subtype.html":("Label × Subtype","상위 라벨과 세부 라벨 관계"),
            "07_model_vs_gold_metrics.html":("Model Metrics","Accuracy / Macro-F1"),
        }
        ann_cards=[]
        for p in ann_files:
            title,desc=ann_titles.get(p.name,(p.stem,"향후 annotation 분석"))
            rel=f"annotations/{p.name}"
            ann_cards.append(chart_card(rel,title,desc,"wide"))
        annotation_html=f"<section class=\"section\" id=\"section-annotation\"><div class=\"section-title\"><span class=\"num\">A</span><div><h2>향후 Annotation 분석</h2><p>Annotation 데이터가 들어오면 동일한 화면에서 자동 활성화됩니다.</p></div></div><div class=\"grid\">{''.join(ann_cards)}</div></section>"
    else:
        annotation_html="<section class=\"section annotation-empty\" id=\"section-annotation\"><div class=\"section-title\"><span class=\"num\">A</span><div><h2>향후 Annotation 분석</h2><p>현재는 annotation 데이터가 없어 관련 시각화를 생성하지 않았습니다.</p></div></div><div class=\"skip-grid\"><div class=\"skip-card\"><strong>SKIP</strong><span>Annotation label distribution</span></div><div class=\"skip-card\"><strong>SKIP</strong><span>Annotator agreement / confusion matrix</span></div><div class=\"skip-card\"><strong>SKIP</strong><span>Gold vs model / Macro-F1</span></div><div class=\"skip-card\"><strong>SKIP</strong><span>Confidence / reliability × time</span></div></div></section>"

    kpi=[
        ("발언",summary.get("rows","—"),"분석 대상 발언"),("의원",summary.get("speakers","—"),"식별된 발언자"),
        ("회의",summary.get("meetings","—"),"회의/세션"),("기간",f"{summary.get('start','—')} ~ {summary.get('end','—')}","분석 범위"),
    ]
    kpi_html=''.join(f'<div class="kpi"><div class="kpi-label">{escape(str(a))}</div><div class="kpi-value">{escape(str(b))}</div><div class="kpi-note">{escape(str(c))}</div></div>' for a,b,c in kpi)

    rows=int(summary.get("rows",0) or 0)
    dev=int(summary.get("development",0) or 0); bound=int(summary.get("boundary",0) or 0); both=int(summary.get("both",0) or 0)
    rel=dev+bound+both
    share=rel/rows*100 if rows else 0
    first_sentence=f"분석 대상 {rows:,}개 발언 가운데 발전·경계 관련 발언은 {rel:,}건({share:.1f}%)이며, 발전 {dev:,}건·경계 {bound:,}건·중첩 {both:,}건으로 구성됩니다."
    second_sentence=insights.get("02_월별_담론상대빈도.html", "상대빈도는 전체 발언량 차이를 보정해 담론의 시기적 집중을 보여줍니다.")

    data_links=[
        ("network_speech_flow_edges.csv","발언흐름 edge"),("network_keyword_cooccurrence_edges.csv","키워드 공출현 edge"),("network_centrality.csv","중심성 결과"),("network_knowledge_graph_edges.csv","지식그래프 edge"),("validation_network.csv","네트워크 검증"),("logs/run.log","실행 로그"),
    ]
    data_html=''.join(f'<a class="data-link" href="{quote(fn)}">{escape(label)}</a>' for fn,label in data_links if (OUTPUT_DIR/fn).exists())

    text=f'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>이승만 시기 국회회의록 — 통합 분석 대시보드</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--line:#dde2e7;--text:#18212b;--muted:#687482;--accent:#263f55;--soft:#eef2f5;--ok:#2f6b4f;--skip:#7b8087;--insight:#f7f9fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;line-height:1.48}}
.page{{max-width:1780px;margin:0 auto;padding:28px 26px 64px}}
.hero{{background:linear-gradient(135deg,#172736,#314b60);color:#fff;border-radius:22px;padding:34px 40px 30px;box-shadow:0 8px 30px rgba(20,35,50,.14)}} .hero h1{{margin:0;font-size:31px;letter-spacing:-.03em}} .hero p{{margin:10px 0 0;color:#dce6ed;max-width:1050px}} .hero-note{{margin-top:12px;font-size:13px;color:#bfcdd7}}
.executive{{margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:18px;padding:20px 22px;box-shadow:0 3px 14px rgba(30,45,60,.045)}} .executive-label{{font-size:11px;font-weight:800;color:var(--accent);letter-spacing:.08em;text-transform:uppercase}} .executive h2{{font-size:19px;margin:5px 0 9px}} .executive p{{margin:4px 0;color:#35424e;font-size:14px;max-width:1200px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:16px 0 24px}} .kpi{{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:15px 17px;min-height:110px}} .kpi-label{{font-size:11px;color:var(--muted);font-weight:700}} .kpi-value{{font-size:24px;font-weight:800;margin:6px 0 4px;letter-spacing:-.02em}} .kpi-note{{font-size:12px;color:var(--muted)}}
.featured{{margin-top:20px}} .section{{margin-top:32px}} .section-title{{display:flex;align-items:flex-start;gap:14px;margin-bottom:13px}} .section-title .num{{flex:0 0 auto;width:38px;height:38px;border-radius:11px;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px}} .section-title h2{{margin:0;font-size:21px;letter-spacing:-.02em}} .section-title p{{margin:3px 0 0;color:var(--muted);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:15px;overflow:hidden;min-width:0;box-shadow:0 2px 10px rgba(30,45,60,.035)}} .card.wide{{grid-column:1/-1}} .hero-chart{{box-shadow:0 6px 22px rgba(30,45,60,.07);border-color:#cdd7df}}
.card-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:13px 16px 9px;border-bottom:1px solid #edf0f2;min-height:66px}} .card-head h3{{font-size:15px;margin:0;font-weight:800;letter-spacing:-.01em}} .card-head p{{font-size:11px;color:var(--muted);margin:4px 0 0}} .status{{flex:0 0 auto;font-size:9px;border:1px solid var(--line);color:var(--skip);padding:3px 7px;border-radius:999px}} .status.ready{{color:var(--ok);border-color:#c8ded0;background:#f2f8f4}}
.insight{{padding:10px 16px 11px;background:var(--insight);border-bottom:1px solid #edf0f2;font-size:12px;color:#3b4752;min-height:41px}} .insight b{{color:var(--accent);margin-right:5px}} .skip-insight{{color:var(--muted)}} iframe{{display:block;width:100%;height:500px;border:0;background:#fff}} .card.wide iframe{{height:590px}} .hero-chart iframe{{height:620px}}
.missing-box{{height:170px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:13px;background:#fafbfc}} .annotation-empty{{margin-top:38px}} .skip-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .skip-card{{border:1px dashed #cbd1d6;background:#fafbfc;border-radius:12px;padding:15px;display:flex;flex-direction:column;gap:7px}} .skip-card strong{{font-size:10px;color:var(--skip)}} .skip-card span{{font-size:12px;color:var(--muted)}}
.data-strip{{margin-top:34px;background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:15px;display:flex;flex-wrap:wrap;gap:9px;align-items:center}} .data-strip .label{{font-size:11px;font-weight:800;color:var(--muted);margin-right:2px}} .data-link{{text-decoration:none;color:var(--accent);border:1px solid var(--line);background:var(--soft);border-radius:999px;padding:6px 10px;font-size:10px}} .footer{{margin-top:18px;color:var(--muted);font-size:10px;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}}
@media(max-width:1200px){{.page{{padding:22px 16px 46px}}.grid{{grid-template-columns:1fr}}.card.wide{{grid-column:auto}}.kpis{{grid-template-columns:repeat(2,1fr)}}.skip-grid{{grid-template-columns:repeat(2,1fr)}} iframe,.card.wide iframe{{height:490px}} .hero-chart iframe{{height:540px}}}}
@media(max-width:700px){{.hero{{padding:25px 21px}}.hero h1{{font-size:23px}}.kpis{{grid-template-columns:1fr 1fr}}.skip-grid{{grid-template-columns:1fr}}.kpi-value{{font-size:19px}} iframe,.card.wide iframe,.hero-chart iframe{{height:420px}} .executive p{{font-size:13px}}}}
</style></head>
<body><main class="page">
<header class="hero"><h1>이승만 시기 국회회의록 — 통합 담론·네트워크 분석</h1><p>발전담론·경계담론, 실제 키워드, 의원의 정치사회적 특성, 발언 이벤트와 네트워크를 하나의 연속적인 화면에서 읽습니다.</p><div class="hero-note">발언흐름 네트워크는 직접적인 정치적 상호작용을 뜻하지 않고, 동일 회의에서 연속한 발언의 구조적 인접성을 나타냅니다. 네트워크 라벨은 가독성을 위해 핵심 노드 중심으로 표시하고 나머지는 hover에서 확인할 수 있습니다.</div></header>
<section class="executive"><div class="executive-label">Executive Summary</div><h2>가장 먼저 볼 결과</h2><p>{escape(first_sentence)}</p><p>{escape(second_sentence)}</p></section>
<div class="kpis">{kpi_html}</div>
<section class="featured"><div class="section-title"><span class="num">★</span><div><h2>핵심 시각화</h2><p>전체 발언량의 영향을 보정한 상대빈도 그래프를 첫 화면의 기준점으로 배치했습니다.</p></div></div><div class="grid">{featured_html}</div></section>
{''.join(section_html)}
{annotation_html}
<div class="data-strip"><span class="label">데이터 / 검증</span>{data_html}</div>
<div class="footer"><span>분석 범위: {escape(str(summary.get('start','—')))} ~ {escape(str(summary.get('end','—')))}</span><span>구조: 발언(Event) 중심 · 의원–메타데이터–담론–키워드 · 핵심 노드만 라벨 표시</span></div>
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

    # network
    cross_graph = build_discourse_cross_network(df)
    keyword_graph, _ = build_keyword_network(df)
    flow_graph, _ = build_speech_flow_network(df)
    centrality_df = centrality_analysis(flow_graph)
    ego_network_analysis(flow_graph, centrality_df)
    timeline_network(df, flow_graph)
    knowledge_graph = build_knowledge_graph(df, speakers, dev_keywords, boundary_keywords)

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
