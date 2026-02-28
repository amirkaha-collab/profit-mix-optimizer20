from __future__ import annotations
# -*- coding: utf-8 -*-
import itertools
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pandas.io.formats.style import Styler
import streamlit as st

# ----------------------------
# Page config (wide)
# ----------------------------
st.set_page_config(
    page_title="Profit Mix Optimizer",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# RTL + responsive theme CSS
# ----------------------------
RTL_CSS = """
<style>
/* RTL baseline */
html, body, [class*="css"]  {
  direction: rtl;
  text-align: right;
}

/* Keep sliders LTR so the thumb and ticks behave naturally */
div[data-baseweb="slider"]{
  direction: ltr !important;
}
div[data-baseweb="slider"] *{
  direction: ltr !important;
}

/* Header / cards */
.profit-title {
  font-size: 34px;
  font-weight: 800;
  margin-bottom: 2px;
}
.profit-subtitle {
  font-size: 15px;
  opacity: 0.85;
  margin-top: 0px;
  margin-bottom: 18px;
}
.kpi-card {
  border-radius: 18px;
  padding: 14px 16px;
  border: 1px solid rgba(120,120,120,0.20);
  background: rgba(255,255,255,0.55);
}
@media (prefers-color-scheme: dark) {
  .kpi-card { background: rgba(30,30,30,0.55); border: 1px solid rgba(255,255,255,0.12); }
}
/* Mobile: prefer dark background */
@media (max-width: 768px) {
  .stApp {
    background: #0f1116;
    color: #e8e8e8;
  }
  .kpi-card { background: rgba(25,25,30,0.7); border: 1px solid rgba(255,255,255,0.12); }
}

/* Make dataframe headers RTL */
div[data-testid="stDataFrame"] *{
  direction: rtl;
  text-align: right;
}

/* Wider text columns */
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# ----------------------------
# Password Gate
# ----------------------------
def _check_password() -> bool:
    """
    Password gate with Streamlit session_state.
    Recommended: set APP_PASSWORD in Streamlit secrets.
    """
    if st.session_state.get("auth_ok", False):
        return True

    correct = None
    # Prefer secrets; fallback to env; final fallback hardcoded (demo only)
    if hasattr(st, "secrets") and "APP_PASSWORD" in st.secrets:
        correct = str(st.secrets["APP_PASSWORD"])
    else:
        correct = os.getenv("APP_PASSWORD", "1234")

    st.markdown('<div class="profit-title">🔒 כניסה</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="profit-subtitle">האפליקציה מוגנת בסיסמה. הזן סיסמה כדי להמשיך.</div>',
        unsafe_allow_html=True,
    )
    pwd = st.text_input("סיסמה", type="password", placeholder="••••••••")

    c1, c2 = st.columns([1, 6])
    with c1:
        go = st.button("כניסה", use_container_width=True)
    if go:
        if pwd == correct:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("סיסמה שגויה.")

    st.stop()

_check_password()

# ----------------------------
# Constants / helpers
# ----------------------------
FUNDS_FILE = "funds_data.xlsx"
SERVICE_FILE = "service_scores.xlsx"

PARAM_ALIASES = {
    "stocks": ["סך חשיפה למניות", "מניות"],
    "foreign": ["סך חשיפה לנכסים המושקעים בחו\"ל", "סך חשיפה לנכסים המושקעים בחו׳ל", "חו\"ל", "חו׳ל"],
    "fx": ["חשיפה למט\"ח", "מט\"ח", "מט׳׳ח"],
    "illiquid": ["נכסים לא סחירים", "לא סחירים", "לא-סחיר", "לא סחיר"],
    "sharpe": ["מדד שארפ", "שארפ"],
    "israel_assets": ["נכסים בארץ", "נכסים בישראל", "בארץ", "ישראל"],
}

DISPLAY_NAMES = {
    "foreign": "יעד חו״ל (%)",
    "israel": "יעד ישראל (%)",
    "stocks": "יעד מניות (%)",
    "fx": "יעד מט״ח (%)",
    "illiquid": "מקסימום לא־סחיר (%)",
    "sharpe": "שארפ",
    "service": "שירות",
    "score": "Score (סטייה)",
}

def _to_float(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace(",", "")
    s = s.replace("%", "")
    s = s.replace("−", "-")
    s = re.sub(r"[^\d\.\-]+", "", s)
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _match_param(row_name: str, key: str) -> bool:
    rn = str(row_name).strip()
    for a in PARAM_ALIASES[key]:
        if a in rn:
            return True
    return False

def _extract_manager(fund_name: str) -> str:
    """
    Heuristic: take first token up to 'קרן'/'השתלמות'/'-' etc.
    Works for names like 'מנורה השתלמות כללי', 'כלל השתלמות כללי'.
    """
    name = str(fund_name).strip()
    # common splitters
    for splitter in [" קרן", " השתלמות", " -", "-", "  "]:
        if splitter in name:
            head = name.split(splitter)[0].strip()
            if head:
                return head
    # fallback: first word
    return name.split()[0] if name.split() else name

@dataclass
class FundRecord:
    track: str
    fund: str
    manager: str
    stocks: float
    foreign: float
    fx: float
    illiquid: float
    sharpe: float
    service: float

def _load_service_scores(path: str) -> Dict[str, float]:
    try:
        df = pd.read_excel(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if "provider" not in df.columns or "score" not in df.columns:
        # Try first two columns
        df = df.iloc[:, :2].copy()
        df.columns = ["provider", "score"]
    out = {}
    for _, r in df.iterrows():
        p = str(r["provider"]).strip()
        sc = _to_float(r["score"])
        if p and not math.isnan(sc):
            out[p] = float(sc)
    return out

@st.cache_data(show_spinner=False)
def load_funds_long(funds_path: str, service_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    svc = _load_service_scores(service_path)
    xls = pd.ExcelFile(funds_path)
    records: List[Dict] = []
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, header=None)
        if df.empty:
            continue
        # Expect first row as headers: first cell 'פרמטר'
        # Column 0: parameter names, columns 1..n: funds
        header_row = df.iloc[0].tolist()
        if not str(header_row[0]).strip().startswith("פרמטר"):
            # try find 'פרמטר' row
            idxs = df.index[df.iloc[:, 0].astype(str).str.contains("פרמטר", na=False)].tolist()
            if not idxs:
                continue
            df = df.iloc[idxs[0]:].reset_index(drop=True)
            header_row = df.iloc[0].tolist()
        fund_names = [c for c in header_row[1:] if str(c).strip() and str(c).strip() != "nan"]
        if not fund_names:
            continue

        # Build mapping param->row index
        param_col = df.iloc[1:, 0].astype(str).tolist()
        def row_for(key: str) -> Optional[int]:
            for i, rn in enumerate(param_col, start=1):
                if _match_param(rn, key):
                    return i
            return None

        ridx_stocks = row_for("stocks")
        ridx_foreign = row_for("foreign")
        ridx_fx = row_for("fx")
        ridx_ill = row_for("illiquid")
        ridx_sharpe = row_for("sharpe")

        if ridx_foreign is None and ridx_stocks is None:
            # Not a relevant sheet
            continue

        for j, fname in enumerate(fund_names, start=1):
            manager = _extract_manager(fname)
            rec = {
                "track": sh,
                "fund": str(fname).strip(),
                "manager": manager,
                "stocks": _to_float(df.iloc[ridx_stocks, j]) if ridx_stocks is not None else np.nan,
                "foreign": _to_float(df.iloc[ridx_foreign, j]) if ridx_foreign is not None else np.nan,
                "fx": _to_float(df.iloc[ridx_fx, j]) if ridx_fx is not None else np.nan,
                "illiquid": _to_float(df.iloc[ridx_ill, j]) if ridx_ill is not None else np.nan,
                "sharpe": _to_float(df.iloc[ridx_sharpe, j]) if ridx_sharpe is not None else np.nan,
            }
            # keep only rows that have at least foreign or stocks numeric
            if all(math.isnan(rec[k]) for k in ["foreign", "stocks", "fx", "illiquid", "sharpe"]):
                continue
            rec["service"] = float(svc.get(manager, 50.0))  # placeholder default
            records.append(rec)

    df_long = pd.DataFrame.from_records(records)
    # Clean NaNs to float
    for c in ["stocks", "foreign", "fx", "illiquid", "sharpe", "service"]:
        if c in df_long.columns:
            df_long[c] = pd.to_numeric(df_long[c], errors="coerce")

    return df_long, svc

# ----------------------------
# Load embedded files
# ----------------------------
if not os.path.exists(FUNDS_FILE):
    st.error(f"לא נמצא קובץ הנתונים '{FUNDS_FILE}' בתיקיית הפרויקט. העלה אותו לשורש הריפו ב-GitHub.")
    st.stop()
if not os.path.exists(SERVICE_FILE):
    st.error(f"לא נמצא קובץ השירות '{SERVICE_FILE}' בתיקיית הפרויקט. העלה אותו לשורש הריפו ב-GitHub.")
    st.stop()

df_long, service_map = load_funds_long(FUNDS_FILE, SERVICE_FILE)

# Basic validation
n_tracks = df_long["track"].nunique() if not df_long.empty else 0
n_records = len(df_long)

st.markdown('<div class="profit-title">📊 Profit Mix Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="profit-subtitle">כלי לחיפוש תמהיל השקעות אופטימלי בין מסלולי קרנות השתלמות, על בסיס קובץ הנתונים המובנה. '
    'מגדירים יעד חשיפות, מגבלות וקריטריוני דירוג – ומקבלים 3 חלופות שונות.</div>',
    unsafe_allow_html=True
)
st.info(f"✅ זוהו **{n_tracks}** מסלולי השקעה בקובץ | ✅ זוהו **{n_records}** קופות (מנהל×מסלול)")

if df_long.empty:
    st.error("לא הצלחתי לזהות טבלאות בקובץ. ודא שבכל גיליון יש שורה ראשונה 'פרמטר' ואז עמודות של קופות.")
    st.stop()

# ----------------------------
# UI
# ----------------------------
tab1, tab2, tab3 = st.tabs(["הגדרות יעד", "תוצאות (3 חלופות)", "פירוט חישוב / שקיפות"])

# Session defaults
def _init_state():
    st.session_state.setdefault("n_funds", 2)
    st.session_state.setdefault("mix_policy", "מותר לערבב מנהלים")
    st.session_state.setdefault("step", 5)
    st.session_state.setdefault("primary_rank", "דיוק")
    st.session_state.setdefault("targets", {"foreign": 30.0, "stocks": 40.0, "fx": 25.0, "illiquid": 20.0})
    st.session_state.setdefault("include", {"foreign": True, "stocks": True, "fx": False, "illiquid": False})
    st.session_state.setdefault("constraint", {
        "foreign": ("רך", "בדיוק"),
        "stocks": ("רך", "בדיוק"),
        "fx": ("רך", "לפחות"),
        "illiquid": ("קשיח", "לכל היותר"),
    })
    st.session_state.setdefault("score_weights", {"foreign": 1.0, "stocks": 1.0, "fx": 1.0, "illiquid": 1.0})
    st.session_state.setdefault("last_results", None)
    st.session_state.setdefault("last_note", "")

_init_state()

def _weights_for_n(n: int, step: int) -> List[Tuple[int, ...]]:
    step = max(1, int(step))
    if n == 1:
        return [(100,)]
    if n == 2:
        return [(w, 100 - w) for w in range(0, 101, step)]
    # n == 3
    out = []
    for w1 in range(0, 101, step):
        for w2 in range(0, 101 - w1, step):
            w3 = 100 - w1 - w2
            if w3 % step == 0:
                out.append((w1, w2, w3))
    return out

def _apply_israel_rule(targets: Dict[str, float]) -> Dict[str, float]:
    # If user sets Israel target, convert to foreign internally; we expose only foreign in UI (per your latest note),
    # but keep the rule here for consistency.
    return targets

def _compute_mix_metrics(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # arr shape (n_funds, n_metrics), weights (n_funds,)
    return np.nansum(arr * weights[:, None], axis=0)

def _hard_ok(value: float, target: float, mode: str) -> bool:
    # mode: "בדיוק", "לפחות", "לכל היותר"
    if math.isnan(value):
        return False
    if mode == "בדיוק":
        return abs(value - target) < 1e-9
    if mode == "לפחות":
        return value + 1e-9 >= target
    if mode == "לכל היותר":
        return value - 1e-9 <= target
    return True

def _soft_distance(value: float, target: float) -> float:
    if math.isnan(value) or math.isnan(target):
        return 0.0
    return abs(value - target) / 100.0  # normalize

def _make_advantage(primary: str, row: Dict, base_row: Optional[Dict]=None) -> str:
    if primary == "דיוק":
        return f"הכי מדויק ליעד, סטייה כוללת {row['score']:.4f}"
    if primary == "שארפ":
        if base_row is None:
            return f"שארפ משוקלל גבוה ({row['sharpe']:.2f})"
        delta = row["sharpe"] - base_row.get("sharpe", 0.0)
        return f"שארפ גבוה יותר ב-{delta:.2f} תוך סטייה {row['score']:.4f}"
    if primary == "שירות":
        if base_row is None:
            return f"ציון שירות משוקלל הגבוה ביותר ({row['service']:.1f})"
        delta = row["service"] - base_row.get("service", 0.0)
        return f"שירות גבוה יותר ב-{delta:.1f} תוך סטייה {row['score']:.4f}"
    return f"חלופה חזקה לפי {primary}"

def _prefilter_candidates(df: pd.DataFrame, include: Dict[str, bool], targets: Dict[str, float], cap: int) -> pd.DataFrame:
    # Quick score for single fund closeness to targets (soft only) to reduce search space
    # Keep those with smallest sum of deviations for selected soft metrics (foreign/stocks/fx)
    keys = [k for k, v in include.items() if v and k in ["foreign", "stocks", "fx", "illiquid"]]
    if not keys:
        keys = ["foreign", "stocks"]
    tmp = df.copy()
    score = np.zeros(len(tmp), dtype=float)
    for k in keys:
        score += np.abs(tmp[k].fillna(0.0).to_numpy() - float(targets.get(k, 0.0))) / 100.0
    tmp["_single_score"] = score
    tmp = tmp.sort_values("_single_score", ascending=True).head(cap).drop(columns=["_single_score"])
    return tmp

def find_best_solutions(
    df: pd.DataFrame,
    n_funds: int,
    step: int,
    mix_policy: str,
    include: Dict[str, bool],
    constraint: Dict[str, Tuple[str, str]],
    targets: Dict[str, float],
    primary_rank: str,
    max_solutions_scan: int = 60000,
) -> Tuple[pd.DataFrame, str]:
    """
    Returns a dataframe of candidate solutions (many), and a note string.
    We use a stable/rigorous scan but with a prefilter cap to keep Streamlit Cloud responsive.
    """
    # Validate include targets
    targets = {k: float(v) for k, v in targets.items()}

    # Pre-filter to keep search manageable
    cap = 80 if n_funds == 2 else 55 if n_funds == 3 else 120
    df_scan = _prefilter_candidates(df, include, targets, cap=cap)

    weights_list = _weights_for_n(n_funds, step)
    if not weights_list:
        return pd.DataFrame(), "לא נמצאו משקלים אפשריים. נסה צעד קטן יותר (למשל 5%)."

    # Decide columns for objective
    metric_keys = ["foreign", "stocks", "fx", "illiquid"]
    active_soft = [k for k in metric_keys if include.get(k, False)]
    if not active_soft:
        active_soft = ["foreign", "stocks"]

    # Hard constraints: any metric marked "קשיח"
    hard_keys = []
    for k in metric_keys:
        hardness, mode = constraint.get(k, ("רך", "בדיוק"))
        if hardness == "קשיח":
            hard_keys.append((k, mode))

    # Arrays
    A = df_scan[["foreign", "stocks", "fx", "illiquid", "sharpe", "service"]].to_numpy(dtype=float)
    # indices map to df rows
    records = df_scan.reset_index(drop=True)

    solutions = []
    scanned = 0
    # group by manager if needed
    if mix_policy == "אותו מנהל בלבד":
        groups = list(records.groupby("manager").groups.values())
        combos_iter = []
        for idxs in groups:
            if len(idxs) >= n_funds:
                combos_iter.append(itertools.combinations(list(idxs), n_funds))
        combo_source = itertools.chain.from_iterable(combos_iter)
    else:
        combo_source = itertools.combinations(range(len(records)), n_funds)

    for combo in combo_source:
        combo = tuple(combo)
        arr = A[list(combo), :]  # (n, 6)
        # quick skip if all nan for key metrics
        if np.all(np.isnan(arr[:, 0:4])):
            continue

        for w in weights_list:
            scanned += 1
            if scanned > max_solutions_scan:
                break

            weights = np.array(w, dtype=float) / 100.0
            mix = _compute_mix_metrics(arr[:, 0:6], weights)
            foreign, stocks, fx, illiq, sharpe, service = mix.tolist()
            israel = 100.0 - foreign if not math.isnan(foreign) else np.nan

            # Hard constraints
            ok = True
            for k, mode in hard_keys:
                val = {"foreign": foreign, "stocks": stocks, "fx": fx, "illiquid": illiq}.get(k, np.nan)
                tgt = targets.get(k, 0.0)
                if not _hard_ok(val, tgt, mode):
                    ok = False
                    break
            if not ok:
                continue

            # Score for "דיוק": sum of normalized deviations for active soft keys (even if they are hard, score still informative)
            score = 0.0
            for k in active_soft:
                val = {"foreign": foreign, "stocks": stocks, "fx": fx, "illiquid": illiq}.get(k, np.nan)
                score += _soft_distance(val, targets.get(k, 0.0))

            # Gather labels
            fund_labels = [records.loc[i, "fund"] for i in combo]
            track_labels = [records.loc[i, "track"] for i in combo]
            managers = [records.loc[i, "manager"] for i in combo]
            manager_set = " | ".join(sorted(set(managers)))

            solutions.append({
                "combo": combo,
                "weights": w,
                "מנהלים": manager_set,
                "מסלולים": " | ".join(track_labels),
                "קופות": " | ".join(fund_labels),
                "חו״ל (%)": foreign,
                "ישראל (%)": israel,
                "מניות (%)": stocks,
                "מט״ח (%)": fx,
                "לא־סחיר (%)": illiq,
                "שארפ משוקלל": sharpe,
                "שירות משוקלל": service,
                "score": score,
            })
        if scanned > max_solutions_scan:
            break

    if not solutions:
        return pd.DataFrame(), f"לא נמצאו פתרונות שעומדים במגבלות. נסה לרכך מגבלות קשיחות או להגדיל צעד/מספר קופות."

    df_sol = pd.DataFrame(solutions)

    note = f"נסרקו {min(scanned, max_solutions_scan):,} קומבינציות (לאחר סינון מוקדם ל-{len(records)} קופות)."

    # Sorting by primary rank for candidate ordering
    if primary_rank == "דיוק":
        df_sol = df_sol.sort_values(["score", "שארפ משוקלל", "שירות משוקלל"], ascending=[True, False, False])
    elif primary_rank == "שארפ":
        df_sol = df_sol.sort_values(["שארפ משוקלל", "score"], ascending=[False, True])
    elif primary_rank == "שירות":
        df_sol = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])
    else:
        df_sol = df_sol.sort_values(["score"], ascending=[True])

    return df_sol, note

def pick_three_distinct(df_sol: pd.DataFrame, primary_rank: str) -> pd.DataFrame:
    """
    Always return 3 solutions with distinct manager sets.
    #1: best by primary_rank ordering already in df_sol
    #2: best by Sharpe (distinct managers)
    #3: best by Service (distinct managers)
    """
    if df_sol.empty:
        return df_sol

    picked = []
    used_manager_sets = set()

    def manager_key(row) -> str:
        return str(row["מנהלים"]).strip()

    # 1) primary
    for _, r in df_sol.iterrows():
        mk = manager_key(r)
        if mk not in used_manager_sets:
            picked.append(r)
            used_manager_sets.add(mk)
            break

    base = picked[0] if picked else None

    # 2) Sharpe
    df_sh = df_sol.sort_values(["שארפ משוקלל", "score"], ascending=[False, True])
    for _, r in df_sh.iterrows():
        mk = manager_key(r)
        if mk not in used_manager_sets:
            picked.append(r)
            used_manager_sets.add(mk)
            break

    # 3) Service
    df_sv = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])
    for _, r in df_sv.iterrows():
        mk = manager_key(r)
        if mk not in used_manager_sets:
            picked.append(r)
            used_manager_sets.add(mk)
            break

    # Fill if still missing (rare)
    if len(picked) < 3:
        for _, r in df_sol.iterrows():
            mk = manager_key(r)
            if mk not in used_manager_sets:
                picked.append(r)
                used_manager_sets.add(mk)
            if len(picked) == 3:
                break

    df_out = pd.DataFrame(picked).reset_index(drop=True)

    # Add "חלופה" + "יתרון"
    rows = []
    for i in range(len(df_out)):
        row = df_out.iloc[i].to_dict()
        if i == 0:
            row["חלופה"] = "חלופה 1 (דירוג ראשי)"
            row["יתרון"] = _make_advantage(primary_rank, row)
        elif i == 1:
            row["חלופה"] = "חלופה 2 (שארפ)"
            row["יתרון"] = _make_advantage("שארפ", row, base_row=base.to_dict() if base is not None else None)
        else:
            row["חלופה"] = "חלופה 3 (שירות)"
            row["יתרון"] = _make_advantage("שירות", row, base_row=base.to_dict() if base is not None else None)
        rows.append(row)
    return pd.DataFrame(rows)

def _color_rows(df: pd.DataFrame, targets: Dict[str, float], constraint: Dict[str, Tuple[str, str]]) -> 'Styler':
    # Conditional formatting via Styler (works in st.dataframe as static if use st.dataframe? better use st.dataframe without styler.
    # We'll use st.dataframe normally and add per-cell highlights by HTML-free notes in KPI cards.
    return df.style

def _render_kpi_cards(alt_rows: pd.DataFrame):
    if alt_rows.empty:
        return
    cols = st.columns(3)
    for i in range(min(3, len(alt_rows))):
        r = alt_rows.iloc[i]
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
              <div style="font-weight:800; font-size:16px; margin-bottom:6px;">{r['חלופה']}</div>
              <div style="font-size:14px; margin-bottom:6px;">Score: <b>{r['score']:.4f}</b></div>
              <div style="font-size:13px; opacity:0.95;">
                חו״ל: <b>{r['חו״ל (%)']:.2f}%</b> ·
                מניות: <b>{r['מניות (%)']:.2f}%</b> ·
                מט״ח: <b>{r['מט״ח (%)']:.2f}%</b> ·
                לא־סחיר: <b>{r['לא־סחיר (%)']:.2f}%</b>
              </div>
              <div style="font-size:13px; margin-top:6px;">
                שארפ: <b>{r['שארפ משוקלל']:.2f}</b> · שירות: <b>{r['שירות משוקלל']:.1f}</b>
              </div>
              <div style="font-size:12.5px; margin-top:8px; opacity:0.9;">{r['יתרון']}</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Tab 1: Inputs (main, not sidebar)
# ----------------------------
with tab1:
    st.subheader("הגדרות בסיס")
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.2, 1.2])

    with c1:
        st.session_state["n_funds"] = st.selectbox(
            "כמה קופות לשלב?",
            options=[1, 2, 3],
            index=[1, 2, 3].index(st.session_state["n_funds"]),
        )
    with c2:
        st.session_state["mix_policy"] = st.selectbox(
            "מדיניות מנהלים",
            options=["מותר לערבב מנהלים", "אותו מנהל בלבד"],
            index=0 if st.session_state["mix_policy"] == "מותר לערבב מנהלים" else 1,
        )
    with c3:
        st.session_state["step"] = st.selectbox(
            "צעד משקלים (%)",
            options=[1, 2, 5, 10, 20],
            index=[1, 2, 5, 10, 20].index(st.session_state["step"]),
            help="בצעד קטן החיפוש יסודי יותר אך כבד יותר.",
        )
    with c4:
        st.session_state["primary_rank"] = st.selectbox(
            "דירוג ראשי",
            options=["דיוק", "שארפ", "שירות"],
            index=["דיוק", "שארפ", "שירות"].index(st.session_state["primary_rank"]),
        )

    st.divider()
    st.subheader("יעדים ומגבלות – סט אחד לכל המשתנים")

    # One unified set: include + target + hard/soft + inequality mode
    rows = []
    mcols = st.columns([1.2, 1.2, 1.2, 1.0, 1.0])
    with mcols[0]:
        st.markdown("**משתנה**")
    with mcols[1]:
        st.markdown("**לכלול בדירוג**")
    with mcols[2]:
        st.markdown("**יעד (%)**")
    with mcols[3]:
        st.markdown("**קשיחות**")
    with mcols[4]:
        st.markdown("**כיוון**")

    def metric_row(key: str, label: str, default_mode: str):
        cols = st.columns([1.2, 1.2, 1.2, 1.0, 1.0])
        with cols[0]:
            st.write(label)
        with cols[1]:
            inc = st.checkbox(" ", value=st.session_state["include"].get(key, False), key=f"inc_{key}")
        with cols[2]:
            val = st.slider(
                " ", min_value=0.0, max_value=120.0 if key in ("foreign", "fx") else 100.0,
                value=float(st.session_state["targets"].get(key, 0.0)),
                step=0.5, key=f"tgt_{key}",
                label_visibility="collapsed"
            )
        with cols[3]:
            hard = st.selectbox(
                " ", options=["רך", "קשיח"],
                index=0 if st.session_state["constraint"].get(key, ("רך", default_mode))[0] == "רך" else 1,
                key=f"hard_{key}",
                label_visibility="collapsed"
            )
        with cols[4]:
            mode = st.selectbox(
                " ", options=["בדיוק", "לפחות", "לכל היותר"],
                index=["בדיוק", "לפחות", "לכל היותר"].index(st.session_state["constraint"].get(key, ("רך", default_mode))[1]),
                key=f"mode_{key}",
                label_visibility="collapsed"
            )
        st.session_state["include"][key] = inc
        st.session_state["targets"][key] = float(val)
        st.session_state["constraint"][key] = (hard, mode)

    metric_row("foreign", "חו״ל", "בדיוק")
    metric_row("stocks", "מניות", "בדיוק")
    metric_row("fx", "מט״ח", "לפחות")
    metric_row("illiquid", "לא־סחיר", "לכל היותר")

    st.divider()
    st.subheader("הרצה")
    run = st.button("חשב 3 חלופות", type="primary", use_container_width=True)

    if run:
        with st.spinner("מריץ חיפוש יסודי..."):
            sols, note = find_best_solutions(
                df=df_long,
                n_funds=st.session_state["n_funds"],
                step=st.session_state["step"],
                mix_policy=st.session_state["mix_policy"],
                include=st.session_state["include"],
                constraint=st.session_state["constraint"],
                targets=_apply_israel_rule(st.session_state["targets"]),
                primary_rank=st.session_state["primary_rank"],
                max_solutions_scan=90000 if st.session_state["n_funds"] <= 2 else 70000,
            )
            st.session_state["last_note"] = note
            if sols.empty:
                st.session_state["last_results"] = None
            else:
                top3 = pick_three_distinct(sols, st.session_state["primary_rank"])
                st.session_state["last_results"] = {
                    "solutions_all": sols.head(5000),  # keep limited for transparency tab
                    "top3": top3,
                }
        if st.session_state["last_results"] is None:
            st.error("לא נמצאו פתרונות.")
        else:
            st.success("מוכן! עבור לטאב 'תוצאות'.")

# ----------------------------
# Tab 2: Results (show full table immediately)
# ----------------------------
with tab2:
    st.subheader("תוצאות (3 חלופות)")
    if st.session_state.get("last_results") is None:
        st.info("כדי לראות תוצאות, עבור לטאב 'הגדרות יעד' ולחץ 'חשב 3 חלופות'.")
    else:
        top3 = st.session_state["last_results"]["top3"].copy()

        # KPI cards
        _render_kpi_cards(top3)

        st.markdown("#### טבלה מלאה")
        st.caption(st.session_state.get("last_note", ""))

        # Column order and wider "קופות"/"מסלולים"
        cols_order = [
            "חלופה", "יתרון",
            "מנהלים", "קופות", "מסלולים",
            "חו״ל (%)", "ישראל (%)", "מניות (%)", "מט״ח (%)", "לא־סחיר (%)",
            "שארפ משוקלל", "שירות משוקלל", "score", "weights",
        ]
        for c in cols_order:
            if c not in top3.columns:
                top3[c] = np.nan
        view = top3[cols_order].copy()

        # Expand weight labels
        def _weights_str(w):
            if isinstance(w, (tuple, list)):
                return " / ".join([f"{int(x)}%" for x in w])
            return str(w)
        view["משקלים"] = view["weights"].apply(_weights_str)
        view = view.drop(columns=["weights"])
        # nicer score name
        view = view.rename(columns={"score": "Score (סטייה)"})

        # Conditional flags for quick reading
        # We avoid heavy Styler; we add emoji columns
        ill_max = float(st.session_state["targets"].get("illiquid", 20.0))
        view["חריג לא־סחיר"] = np.where(view["לא־סחיר (%)"].to_numpy() > ill_max + 1e-9, "🔴", "")
        # score thresholds
        view["סטייה גבוהה"] = np.where(view["Score (סטייה)"].to_numpy() > 0.08, "🟠", "")

        # Make text columns first and wider
        column_config = {
            "קופות": st.column_config.TextColumn(width="large"),
            "מסלולים": st.column_config.TextColumn(width="large"),
            "מנהלים": st.column_config.TextColumn(width="medium"),
            "יתרון": st.column_config.TextColumn(width="large"),
            "חלופה": st.column_config.TextColumn(width="medium"),
        }
        st.dataframe(view, use_container_width=True, hide_index=True, column_config=column_config)

# ----------------------------
# Tab 3: Transparency
# ----------------------------
with tab3:
    st.subheader("פירוט חישוב / שקיפות")
    st.caption("כדי לא להעמיס – הפירוט מוצג בתוך Expander.")
    with st.expander("לחץ להצגת פירוט"):
        st.write("**פרטי קלט:**")
        st.json({
            "מספר קופות": st.session_state["n_funds"],
            "מדיניות מנהלים": st.session_state["mix_policy"],
            "צעד משקלים": st.session_state["step"],
            "דירוג ראשי": st.session_state["primary_rank"],
            "כולל בדירוג": st.session_state["include"],
            "יעדים": st.session_state["targets"],
            "קשיחות/כיוון": st.session_state["constraint"],
            "הערת ריצה": st.session_state.get("last_note", ""),
        }, expanded=False)

        if st.session_state.get("last_results") is None:
            st.info("אין פתרונות להצגה.")
        else:
            st.markdown("**דוגמאות מתוך רשימת המועמדים (עד 200 שורות):**")
            cand = st.session_state["last_results"]["solutions_all"].head(200).copy()
            # show only relevant cols
            cand = cand[[
                "מנהלים", "קופות", "מסלולים",
                "חו״ל (%)", "מניות (%)", "מט״ח (%)", "לא־סחיר (%)",
                "שארפ משוקלל", "שירות משוקלל", "score", "weights"
            ]].copy()
            cand["משקלים"] = cand["weights"].apply(lambda w: " / ".join([f"{int(x)}%" for x in w]) if isinstance(w, (tuple, list)) else str(w))
            cand = cand.drop(columns=["weights"]).rename(columns={"score": "Score (סטייה)"})
            st.dataframe(cand, use_container_width=True, hide_index=True, column_config={
                "קופות": st.column_config.TextColumn(width="large"),
                "מסלולים": st.column_config.TextColumn(width="large"),
            })

st.caption("© Profit Mix Optimizer – חישוב ממוצע משוקלל על בסיס הקובץ המובנה. ישראל = 100 − חו״ל.")
