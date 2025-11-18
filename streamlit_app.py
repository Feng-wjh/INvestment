"""
Streamlit app - Full automated fundamental + financial scoring system
Weights: technical 10%, merrill 20%, porter 10%, value_chain 10% (LLM), financial 50%
DeepSeek (LLM) used for value-chain scoring if API key provided; otherwise heuristic + manual override.
Save as app.py and run:
streamlit run app.py
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests, re, time, math, json
from scipy import stats
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional

# -------------------------
# Page config
# -------------------------

st.set_page_config(page_title="Fundamental+Financial Scoring System", layout="wide", page_icon="💼")
st.title("Automated Investment Scoring — Merrill + Porter + Value Chain (LLM) + Financial")

# Debug banner: show file modification time so we can confirm the running app loaded
import os
try:
    mtime = os.path.getmtime(__file__)
    from datetime import datetime
    st.markdown(f"DEBUG: running streamlit_app.py mtime = {datetime.fromtimestamp(mtime).isoformat()}  ")
except Exception:
    pass

# -------------------------
# Helper utilities
# -------------------------

def safe_div(a, b, default=None):
    try:
        if a is None or b is None: return default
        if b == 0: return default
        return a / b
    except Exception:
        return default

def copy_df_safe(df):
    try:
        if df is None:
            return pd.DataFrame()
        df2 = df.copy()
        df2.index = [str(i) for i in df2.index]
        return df2
    except Exception:
        return pd.DataFrame()

def get_first_available(df, candidates: List[str]):
    """Return first available numeric value from df index names candidates (annual financials)."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.index:
            try:
                vals = df.loc[c].dropna().values
                if len(vals) > 0:
                    return float(vals[0])
            except Exception:
                continue
    return None

# -------------------------
# Data fetcher: yfinance
# -------------------------

def fetch_yf_data(ticker: str) -> Dict[str,Any]:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    income = copy_df_safe(getattr(tk, "financials", pd.DataFrame()))
    balance = copy_df_safe(getattr(tk, "balance_sheet", pd.DataFrame()))
    cashflow = copy_df_safe(getattr(tk, "cashflow", pd.DataFrame()))
    q_income = copy_df_safe(getattr(tk, "quarterly_financials", pd.DataFrame()))
    hist = None
    try:
        hist = tk.history(period="2y")
    except Exception:
        hist = None
    sector = info.get('sector') or info.get('industry') or 'Unknown'
    industry = info.get('industry') or 'Unknown'
    summary = info.get('longBusinessSummary') or info.get('summary') or ''
    return {
        'ticker': ticker, 'info': info,
        'income': income, 'balance': balance, 'cashflow': cashflow,
        'q_income': q_income, 'history': hist,
        'sector': sector, 'industry': industry, 'summary': summary
    }

def normalize_china_ticker(ticker: str, default_exchange: str = 'Auto') -> str:
    """Normalize simple China tickers to Yahoo-style symbols.
    Rules:
    - If ticker already contains a dot (e.g., .SS/.SZ/.HK) return uppercased ticker.
    - If ticker is 6 digits and starts with '6' -> append .SS (Shanghai)
    - If ticker is 6 digits and starts with '0' or '3' or '2' -> append .SZ (Shenzhen)
    - If default_exchange == 'Shanghai (.SS)' -> append .SS
    - If default_exchange == 'Shenzhen (.SZ)' -> append .SZ
    - Otherwise return ticker uppercased.
    """
    t = (ticker or '').strip()
    if not t:
        return t
    if '.' in t:
        return t.upper()
    if re.fullmatch(r"\d{6}", t):
        if t.startswith('6'):
            return t + '.SS'
        if t.startswith(('0','2','3')):
            return t + '.SZ'
    # fallback to default
    if default_exchange == 'Shanghai (.SS)':
        return t + '.SS'
    if default_exchange == 'Shenzhen (.SZ)':
        return t + '.SZ'
    return t.upper()
    # check for 3-4 digit HK numeric tickers (common), append .HK if numeric and 1-4 digits
    if re.fullmatch(r"\d{1,4}", t):
        return t.zfill(4) + '.HK'
    return t.upper()

# -------------------------
# Merrill Clock scoring (20%)
# -------------------------

MERRILL_MAPPING = {
    'recovery': {'technology':100, 'financial services':90, 'consumer cyclical':80, 'consumer discretionary':80},
    'overheat': {'energy':100, 'materials':90, 'industrials':80},
    'stagflation': {'consumer defensive':100, 'healthcare':90, 'utilities':80, 'consumer staples':100},
    'recession': {'bonds proxies':100, 'defensive':80, 'real estate':100}
}

def merrill_score(sector: str, phase: str) -> Tuple[float, str]:
    s = (sector or '').lower()
    mapping = MERRILL_MAPPING.get(phase.lower(), {})
    for k,v in mapping.items():
        if k in s or s in k:
            return float(v), f"Matched {k} -> {v}"
    # mid heuristics
    heur = {'technology':85, 'financial':80, 'energy':75, 'healthcare':70, 'utilities':65, 'consumer':70, 'industrial':68}
    for k,v in heur.items():
        if k in s:
            return float(v), f"Heuristic {k} -> {v}"
    return 50.0, "Neutral"

# Merrill veto checks
def merrill_veto_check(sector: str, phase: str, industry_trends=None) -> Tuple[bool, str]:
    # industry_trends is optional dict with CAGR or capacity info
    pct, reason = merrill_score(sector, phase)
    # Example veto: if mapping indicates strong mismatch -> treat <20 as veto
    if pct < 20:
        return True, "Merrill mismatch (score < 20)"
    # additional structural checks if industry_trends supplied
    if industry_trends:
        cagr = industry_trends.get('cagr', None)
        capacity = industry_trends.get('capacity_overhang', None)
        if cagr is not None and cagr < -0.10:
            return True, "Industry structural decline (CAGR < -10%)"
        if capacity is not None and capacity > 0.4:
            return True, "Industry capacity overhang >40%"
    return False, reason

# -------------------------
# Porter Five Forces (10%)
# -------------------------

def porter_auto(sector: str, industry: str, summary: str) -> Dict[str,Any]:
    text = (summary or '').lower()
    s = (sector or '').lower()
    ratings = {
        'competitive_intensity': 3,
        'entry_barriers': 3,
        'substitute_threat': 3,
        'supplier_bargaining': 3,
        'buyer_bargaining': 3
    }
    if 'software' in s or 'technology' in s:
        ratings.update({'competitive_intensity':4,'entry_barriers':4,'substitute_threat':2})
    if 'retail' in s or 'consumer' in s:
        ratings.update({'competitive_intensity':5,'entry_barriers':2,'buyer_bargaining':4})
    if any(k in text for k in ['patent','proprietary','moat','intellectual property','exclusive']):
        ratings['entry_barriers'] = max(ratings['entry_barriers'],5)
        ratings['competitive_intensity'] = max(ratings['competitive_intensity'],4)
    if any(k in text for k in ['commodity','generic','undifferentiated']):
        ratings['substitute_threat'] = max(ratings['substitute_threat'],4)
    for k in ratings:
        ratings[k] = min(5, max(1, int(round(ratings[k]))))
    weighted = (ratings['competitive_intensity'] * 0.3 +
               ratings['entry_barriers'] * 0.25 +
               ratings['substitute_threat'] * 0.2 +
               ratings['supplier_bargaining'] * 0.15 +
               ratings['buyer_bargaining'] * 0.1)
    # weighted is on 1-5 scale; keep raw value and convert to percentage via mapping table
    porter_pct = porter_weighted_to_pct(weighted)
    return {'ratings': ratings, 'weighted_raw': weighted, 'porter_pct': porter_pct}

def porter_weighted_to_pct(weighted: float) -> float:
    """Map weighted average (1.0-5.0) to percentage bands per spec table."""
    try:
        if weighted >= 4.5:
            return 97.5
        if weighted >= 4.0:
            return 92.0
        if weighted >= 3.5:
            return 84.5
        if weighted >= 3.0:
            return 74.5
        if weighted >= 2.5:
            return 64.5
        if weighted >= 2.0:
            return 54.5
        if weighted >= 1.5:
            return 44.5
        if weighted >= 1.0:
            return 34.5
    except Exception:
        pass
    return 50.0

def porter_veto_check(porter_raw):
    # porter_raw is weighted average in 1-5 scale
    if porter_raw < 2.0:
        return True, "Porter indicates severe deterioration (weighted <2.0)"
    # additional veto: supplier and buyer bargaining both extremely weak (<=1)
    # Note: caller should pass raw ratings dict if available; if not, only use porter_raw
    return False, ""

# -------------------------
# Value Chain (10%) — LLM via DeepSeek (preferred), else heuristic + Quantitative
# -------------------------

def value_chain_quantitative_analysis(yfdata: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于财务数据的价值链定量分析
    使用yfinance数据自动计算价值链各环节的评分
    """
    income = yfdata['income']
    balance = yfdata['balance']
    cashflow = yfdata['cashflow']
    info = yfdata['info']
    
    # 获取关键财务数据
    revenue = get_first_available(income, ['Total Revenue', 'Revenue', 'totalRevenue'])
    cogs = get_first_available(income, ['Cost Of Revenue', 'Cost of Revenue', 'CostOfRevenue'])
    gross_profit = get_first_available(income, ['Gross Profit', 'GrossProfit'])
    inventory = get_first_available(balance, ['Inventory', 'inventory'])
    receivables = get_first_available(balance, ['Accounts Receivable', 'Receivable', 'AccountsReceivable'])
    payables = get_first_available(balance, ['Accounts Payable', 'Payable', 'AccountsPayable'])
    r_d = get_first_available(income, ['Research Development', 'ResearchAndDevelopment', 'R&D'])
    sga = get_first_available(income, ['Selling General Administrative', 'SellingGeneralAndAdministration', 'SGA'])
    total_assets = get_first_available(balance, ['Total Assets', 'totalAssets'])
    current_assets = get_first_available(balance, ['Current Assets', 'currentAssets'])
    current_liabilities = get_first_available(balance, ['Current Liabilities', 'currentLiabilities'])
    operating_cash_flow = get_first_available(cashflow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
    
    # 主要活动评分
    main_scores = {}
    
    # 1. 进货物流 - 存货周转率
    avg_inventory = inventory
    if cogs and avg_inventory and avg_inventory > 0:
        inventory_turnover = cogs / avg_inventory
        # 映射到1-10分
        if inventory_turnover > 8: main_scores['Inbound Logistics'] = 9
        elif inventory_turnover > 5: main_scores['Inbound Logistics'] = 7
        elif inventory_turnover > 3: main_scores['Inbound Logistics'] = 5
        else: main_scores['Inbound Logistics'] = 3
    else:
        main_scores['Inbound Logistics'] = 5
    
    # 2. 生产运营 - 资产周转率
    if revenue and total_assets and total_assets > 0:
        asset_turnover = revenue / total_assets
        if asset_turnover > 1.2: main_scores['Operations'] = 9
        elif asset_turnover > 0.8: main_scores['Operations'] = 7
        elif asset_turnover > 0.5: main_scores['Operations'] = 5
        else: main_scores['Operations'] = 3
    else:
        main_scores['Operations'] = 5
    
    # 3. 发货物流 - 应收账款周转率
    avg_receivables = receivables
    if revenue and avg_receivables and avg_receivables > 0:
        receivables_turnover = revenue / avg_receivables
        if receivables_turnover > 10: main_scores['Outbound Logistics'] = 9
        elif receivables_turnover > 6: main_scores['Outbound Logistics'] = 7
        elif receivables_turnover > 3: main_scores['Outbound Logistics'] = 5
        else: main_scores['Outbound Logistics'] = 3
    else:
        main_scores['Outbound Logistics'] = 5
    
    # 4. 市场营销 - 销售效率（销售费用/收入）
    if sga and revenue and revenue > 0:
        sga_ratio = sga / revenue
        # 销售费用占比越低越好，但也要考虑行业特性
        if sga_ratio < 0.1: main_scores['Marketing & Sales'] = 9
        elif sga_ratio < 0.2: main_scores['Marketing & Sales'] = 7
        elif sga_ratio < 0.3: main_scores['Marketing & Sales'] = 5
        else: main_scores['Marketing & Sales'] = 3
    else:
        main_scores['Marketing & Sales'] = 5
    
    # 5. 服务 - 毛利率反映服务附加值
    if gross_profit and revenue and revenue > 0:
        gross_margin = gross_profit / revenue
        if gross_margin > 0.4: main_scores['Service'] = 9
        elif gross_margin > 0.3: main_scores['Service'] = 7
        elif gross_margin > 0.2: main_scores['Service'] = 5
        else: main_scores['Service'] = 3
    else:
        main_scores['Service'] = 5
    
    # 支持活动评分
    support_scores = {}
    
    # 1. 技术研发 - 研发投入强度
    if r_d and revenue and revenue > 0:
        r_d_ratio = r_d / revenue
        if r_d_ratio > 0.08: support_scores['R&D'] = 9
        elif r_d_ratio > 0.05: support_scores['R&D'] = 7
        elif r_d_ratio > 0.02: support_scores['R&D'] = 5
        else: support_scores['R&D'] = 3
    else:
        support_scores['R&D'] = 5
    
    # 2. 人力资源管理 - 运营现金流/员工数（代理指标）
    # 如果没有员工数，使用运营现金流效率
    if operating_cash_flow and revenue and revenue > 0:
        cash_flow_margin = operating_cash_flow / revenue
        if cash_flow_margin > 0.15: support_scores['HR Management'] = 9
        elif cash_flow_margin > 0.1: support_scores['HR Management'] = 7
        elif cash_flow_margin > 0.05: support_scores['HR Management'] = 5
        else: support_scores['HR Management'] = 3
    else:
        support_scores['HR Management'] = 5
    
    # 3. 基础设施 - 流动比率反映财务健康度
    if current_assets and current_liabilities and current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio > 2.0: support_scores['Infrastructure'] = 9
        elif current_ratio > 1.5: support_scores['Infrastructure'] = 7
        elif current_ratio > 1.0: support_scores['Infrastructure'] = 5
        else: support_scores['Infrastructure'] = 3
    else:
        support_scores['Infrastructure'] = 5
    
    explanation = "基于财务数据的定量价值链分析 - 使用实际财务比率计算"
    
    return {
        'main_scores': main_scores,
        'support_scores': support_scores,
        'explanation': explanation,
        'method': 'quantitative'
    }

def call_deepseek_for_valuechain(summary: str, api_key: str, ticker: str = None) -> Dict[str,Any]:
    """
    Call DeepSeek API to obtain structured main/support activity scores (1-10).
    Expectation: DeepSeek returns a JSON-like structured block or a readable text containing numbers.
    This function will attempt to parse numeric scores for the nine fields.
    If call fails, return None.
    """
    if not api_key:
        return None
    url = "https://api.deepseek.com/v1/chat/completions"
    prompt = f"""
    Please analyze the company's value chain and return a JSON object with the following fields:
    main_scores: {{'Inbound Logistics': int 1-10, 'Operations': int 1-10, 'Outbound Logistics': int 1-10, 'Marketing & Sales': int 1-10, 'Service': int 1-10}}
    support_scores: {{'R&D': int 1-10, 'HR Management': int 1-10, 'Infrastructure': int 1-10}}
    Also include a short text explanation.
    Company summary:
    {summary}
    Return only JSON.
    """
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {"model":"deepseek-chat", "messages":[{"role":"user","content":prompt}], "temperature":0.2}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        j = r.json()
        text = j.get('choices',[{}])[0].get('message',{}).get('content','') if isinstance(j, dict) else (r.text if hasattr(r, 'text') else '')
        # Try parse JSON from text
        try:
            parsed = json.loads(text)
            # expected parsed['main_scores'] and parsed['support_scores']
            ms = parsed.get('main_scores') or parsed.get('mainScores')
            ss = parsed.get('support_scores') or parsed.get('supportScores')
            explanation = parsed.get('explanation', text)
            if ms and ss:
                return {'main_scores': ms, 'support_scores': ss, 'explanation': explanation, 'method': 'llm'}
        except Exception:
            # fallback: regex parse numbers for each key
            ms = {}
            ss = {}
            keys_main = ['Inbound Logistics','Operations','Outbound Logistics','Marketing & Sales','Service']
            keys_supp = ['R&D','HR Management','Infrastructure']
            for k in keys_main+keys_supp:
                # match 0-10 (allow 0) following the key
                pat = re.compile(rf"{re.escape(k)}\D+([0-9]|10)", flags=re.IGNORECASE)
                m = pat.search(text)
                if m:
                    val = int(m.group(1))
                else:
                    val = None
                if k in keys_main:
                    ms[k] = val or 5
                else:
                    ss[k] = val or 5
            return {'main_scores': ms, 'support_scores': ss, 'explanation': text, 'method': 'llm'}
    except Exception:
        return None
    return None

def call_deepseek_debug(summary: str, api_key: str, path_hint: str = None) -> Dict[str,Any]:
    """
    Debug wrapper that calls DeepSeek and returns raw text, status_code and parsed JSON when possible.
    Returns dict: {status_code, raw_text, parsed (or None), error (or None)}
    """
    out = {'status_code': None, 'raw_text': None, 'parsed': None, 'error': None}
    if not api_key:
        out['error'] = 'no_api_key'
        return out
    url = "https://api.deepseek.com/v1/chat/completions"
    prompt = f"Please analyze the following text and return JSON. Context hint: {path_hint or ''}\nTEXT:\n{summary}"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {"model":"deepseek-chat", "messages":[{"role":"user","content":prompt}], "temperature":0.2}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        out['status_code'] = getattr(r, 'status_code', None)
        text = ''
        try:
            text = r.text
        except Exception:
            text = ''
        out['raw_text'] = text
        parsed = None
        try:
            j = r.json()
        except Exception:
            j = None
        # try to extract message content
        content = None
        if isinstance(j, dict):
            try:
                content = j.get('choices',[{}])[0].get('message',{}).get('content','')
            except Exception:
                content = None
        if not content:
            content = text
        # attempt to parse JSON from content, else keep structured response
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = j
        out['parsed'] = parsed
        # if non-2xx, record error payload
        try:
            sc = int(out['status_code']) if out['status_code'] is not None else None
            if sc is not None and sc >= 400:
                out['error'] = parsed if parsed else out['raw_text'] or f"status_code:{sc}"
        except Exception:
            pass
    except Exception as e:
        out['error'] = str(e)
    return out

def call_deepseek_for_full_assessment(summary: str, api_key: str, ticker: str = None) -> Optional[Dict[str,Any]]:
    """
    Ask DeepSeek for a full component-level assessment. Expected return (JSON):
    {
    "components": {"technical": 0-100, "merrill":0-100, "porter":0-100, "value_chain":0-100, "financial":0-100},
    "value_chain": {"main_scores":{...}, "support_scores":{...}},
    "explanation": "..."
    }
    Returns parsed dict or None on failure.
    """
    if not api_key:
        return None
    url = "https://api.deepseek.com/v1/chat/completions"
    prompt = f"""
    Please analyze the company's overall investment profile and return a JSON object with these keys:

    • components: object with numeric scores (0-100) for technical, merrill, porter, value_chain, financial

    • value_chain: optional object with main_scores and support_scores (same schema as the other VC call)

    • explanation: short text
    Company summary:
    {summary}
    Return only JSON.
    """
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {"model":"deepseek-chat", "messages":[{"role":"user","content":prompt}], "temperature":0.2}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        # try parse JSON from response body
        try:
            j = r.json()
        except Exception:
            j = None
        # prefer parsed content inside choices -> message -> content
        content = None
        if isinstance(j, dict):
            try:
                content = j.get('choices',[{}])[0].get('message',{}).get('content','')
            except Exception:
                content = None
        if not content and hasattr(r, 'text'):
            content = r.text
        if not content:
            return None
        # try parse JSON from content
        try:
            parsed = json.loads(content)
            comps = parsed.get('components')
            vc = parsed.get('value_chain')
            expl = parsed.get('explanation', content)
            if comps:
                return {'components': comps, 'value_chain': vc, 'explanation': expl}
        except Exception:
            # try to extract numeric components via regex from the content
            comps = {}
            keys = ['technical','merrill','porter','value_chain','financial']
            low = content.lower()
            for k in keys:
                pat = re.compile(rf"{re.escape(k)}\D+(\d{{1,3}})")
                m = pat.search(low)
                if m:
                    comps[k] = min(100, max(0, int(m.group(1))))
            # if we found any, return
            if comps:
                return {'components': comps, 'value_chain': None, 'explanation': content}
    except Exception:
        return None
    return None

def _normalize_scores_dict(scores: Dict[str,Any], default: int = 5) -> Dict[str,int]:
    out = {}
    for k,v in (scores or {}).items():
        try:
            if v is None:
                out[k] = int(default)
            else:
                out[k] = int(float(v))
        except Exception:
            out[k] = int(default)
    return out

def value_chain_from_scores(main_scores: Dict[str,int], support_scores: Dict[str,int]) -> Tuple[float, float, float]:
    # ensure numeric and robust
    main_scores = _normalize_scores_dict(main_scores, default=5)
    support_scores = _normalize_scores_dict(support_scores, default=5)
    if len(main_scores) == 0:
        main_avg = 5.0
    else:
        main_avg = float(np.mean(list(main_scores.values())))
    if len(support_scores) == 0:
        support_avg = 5.0
    else:
        support_avg = float(np.mean(list(support_scores.values())))
    # main average and support average on 1-10 scale
    value_chain_score_1_10 = main_avg * 0.6 + support_avg * 0.4
    value_chain_pct = (value_chain_score_1_10 / 10.0) * 100.0
    return value_chain_pct, main_avg, support_avg

# value_chain veto
def value_chain_veto_check(main_scores, support_scores):
    # core value chain break if >=3 main activities <=3
    low_main = sum(1 for v in main_scores.values() if (v is not None and v <= 3))
    if low_main >= 3:
        return True, ">=3 main activities score <=3 (core value chain broken)"
    # tech support collapse: R&D == 0 and Infrastructure <=2
    if (support_scores.get('R&D',5) == 0) and (support_scores.get('Infrastructure',5) <= 2):
        return True, "Technical support collapse (R&D zero and infra poor)"
    return False, ""

# -------------------------
# Technical score (10%) simple heuristic using moving averages
# -------------------------

def technical_score_from_history(history_df):
    # history_df expected with 'Close' column
    try:
        if history_df is None or history_df.empty:
            return 50.0, "No history -> neutral"
        close = history_df['Close'].dropna()
        if len(close) < 30:
            return 50.0, "Insufficient history -> neutral"
        ma20 = close.rolling(window=20).mean().iloc[-1]
        ma50 = close.rolling(window=50).mean().iloc[-1] if len(close)>=50 else close.rolling(window=20).mean().iloc[-1]
        last = close.iloc[-1]
        # simple rules
        if last > ma20 and ma20 > ma50:
            return 100.0, "Bullish (Close>MA20 & MA20>MA50)"
        elif last > ma20 or ma20 > ma50:
            return 70.0, "Mild bullish"
        else:
            return 40.0, "Bearish"
    except Exception:
        return 50.0, "Error computing technical -> neutral"

# -------------------------
# Financial analysis (50%) composed of 3 parts:
# 1) Basic operation (50% of financial)
# 2) Bankruptcy risk (20%)
# 3) Valuation margin (30%)
# plus hard vetoes
# -------------------------

def roe_quality_score(roe_pct):
    if roe_pct is None: return 50.0
    # roe_pct expected as decimal (e.g., 0.18 for 18%) or percent numeric
    try:
        if roe_pct > 5:  # user might pass percent (e.g., 18)
            r = roe_pct / 100.0
        else:
            r = float(roe_pct)
    except Exception:
        r = roe_pct
    if r > 0.20: return 100.0
    if r > 0.15: return 80.0
    if r > 0.10: return 60.0
    if r > 0.05: return 40.0
    return 20.0

def compute_operational_scores(yfdata):
    income = yfdata['income']; bal = yfdata['balance']; cf = yfdata['cashflow']
    # ROE: net income / equity (tries)
    net = get_first_available(income, ['Net Income','NetIncomeLoss','Net Income Common Stockholders','netIncome'])
    equity = get_first_available(bal, ['Total Stockholder Equity','Total shareholders equity','Total stockholders equity','Total Equity'])
    roe_pct = None
    if net not in (None,0) and equity not in (None,0):
        roe_pct = (net / equity)
    roe_q = roe_quality_score(roe_pct)
    # Asset turnover
    revenue = get_first_available(income, ['Total Revenue','Revenue','totalRevenue'])
    total_assets = get_first_available(bal, ['Total Assets','totalAssets'])
    at_score = 50.0
    if revenue not in (None,0) and total_assets not in (None,0):
        at = safe_div(revenue, total_assets, None)
        if at is None:
            at_score = 50.0
        else:
            at_score = 100.0 if at >= 1.0 else (70.0 if at >= 0.6 else 40.0)
    # Cash conversion proxy: OCF / Revenue
    ocf = get_first_available(cf, ['Operating Cash Flow','Total Cash From Operating Activities','netCashProvidedByOperatingActivities'])
    ccc_score = 50.0
    if ocf not in (None,0) and revenue not in (None,0):
        ratio = safe_div(abs(ocf), abs(revenue), None)
        if ratio is not None:
            ccc_score = 100.0 if ratio >= 0.15 else (70.0 if ratio >= 0.08 else 40.0)
    op_score = roe_q * 0.6 + ((at_score + ccc_score) / 2.0) * 0.4
    return {'roe_pct': (roe_pct*100 if roe_pct is not None else None), 'roe_quality': roe_q, 'asset_turnover_score': at_score, 'ccc_score': ccc_score, 'operation_score': op_score}

def compute_bankruptcy_scores(yfdata):
    income = yfdata['income']; bal = yfdata['balance']; cf = yfdata['cashflow']; info = yfdata['info']
    ebit = get_first_available(income, ['Ebit','EBIT','ebit','Operating Income','operatingIncome'])
    interest = get_first_available(income, ['Interest Expense','interestExpense']) or info.get('interestExpense')
    interest_coverage = safe_div(ebit, abs(interest), None) if (ebit not in (None,0) and interest not in (None,0)) else None
    ca = get_first_available(bal, ['Current Assets','currentAssets']); cl = get_first_available(bal, ['Current Liabilities','currentLiabilities'])
    current_ratio = safe_div(ca, cl, None) if (ca not in (None,0) and cl not in (None,0)) else None
    # op cash history
    op_hist = []
    for k in ['Operating Cash Flow','Total Cash From Operating Activities','netCashProvidedByOperatingActivities']:
        if k in cf.index:
            try:
                vals = cf.loc[k].dropna().values.tolist()
                op_hist = vals[:3]
                break
            except Exception:
                continue
    # debt/ebitda
    total_debt = info.get('totalDebt'); ebitda = info.get('ebitda')
    debt_ebitda = safe_div(total_debt, ebitda, None) if (total_debt not in (None,0) and ebitda not in (None,0)) else None
    # scoring maps
    def map_ic(ic):
        if ic is None: return 50.0
        if ic > 5: return 100.0
        if ic >= 3: return 70.0
        if ic >= 2: return 40.0
        return 0.0
    def map_cr(cr):
        if cr is None: return 50.0
        if cr > 1.5: return 100.0
        if cr >= 1.2: return 70.0
        if cr >= 1.0: return 40.0
        return 0.0
    def map_ocf(hist):
        if not hist: return 50.0
        pos = sum(1 for v in hist if v is not None and v > 0)
        if pos >= 3: return 100.0
        if pos == 2: return 60.0
        if pos == 1: return 40.0
        return 0.0
    def map_de(d):
        if d is None: return 50.0
        if d > 8: return 0.0
        if d > 5: return 40.0
        if d > 3: return 70.0
        return 100.0
    ic_s = map_ic(interest_coverage); cr_s = map_cr(current_ratio); ocf_s = map_ocf(op_hist); de_s = map_de(debt_ebitda)
    bankruptcy_score = ic_s * 0.4 + ocf_s * 0.3 + de_s * 0.3
    return {'interest_coverage': interest_coverage, 'current_ratio': current_ratio, 'op_hist': op_hist, 'debt_ebitda': debt_ebitda,
            'ic_score': ic_s, 'cr_score': cr_s, 'ocf_score': ocf_s, 'de_score': de_s, 'bankruptcy_score': bankruptcy_score}

def compute_valuation_scores(yfdata):
    info = yfdata['info']
    pe = info.get('trailingPE') or info.get('forwardPE')
    pb = info.get('priceToBook')
    fcf = info.get('freeCashflow')
    market_cap = info.get('marketCap')
    def pe_map(p):
        if p is None: return 50.0
        # approximate percentiles mapping replaced by bands
        if p < 10: return 100.0
        if p < 15: return 80.0
        if p < 25: return 60.0
        return 40.0
    def pb_map(p):
        if p is None: return 50.0
        if p < 1: return 100.0
        if p < 2: return 80.0
        if p < 3: return 60.0
        return 40.0
    rel = (pe_map(pe) + pb_map(pb)) / 2.0
    abs_s = 60.0
    fcf_yield = None
    if fcf not in (None,0) and market_cap not in (None,0):
        fcf_yield = (fcf / market_cap) * 100.0
        if fcf_yield > 8: abs_s = 100.0
        elif fcf_yield >= 5: abs_s = 80.0
        elif fcf_yield >= 3: abs_s = 60.0
        else: abs_s = 40.0
    val_margin = 0.5 * rel + 0.5 * abs_s
    return {'pe': pe, 'pb': pb, 'fcf_yield_pct': fcf_yield, 'rel_score': rel, 'abs_score': abs_s, 'val_margin': val_margin}

# financial veto rules
def financial_hard_veto(bk_scores, yfdata_info, q_income_df):
    """Return True if any hard veto triggered."""
    # 1 interest coverage <1.5 -> veto
    if bk_scores.get('interest_coverage') is not None and bk_scores.get('interest_coverage') < 1.5:
        return True, "Interest coverage < 1.5"
    # 2 operating cash flow negative in last 2 years + cash_ratio < 0.3
    op_hist = bk_scores.get('op_hist', [])
    neg_count = 0
    for v in op_hist[:3]:
        if v is not None and v < 0:
            neg_count += 1
    cash = yfdata_info.get('totalCash') or yfdata_info.get('cash') or 0
    qrev = None
    if q_income_df is not None and not q_income_df.empty:
        qrev = get_first_available(q_income_df, ['Total Revenue','Revenue','totalRevenue'])
    cash_ratio = safe_div(cash, abs(qrev), None) if qrev not in (None,0) else None
    if neg_count >= 2 and (cash_ratio is not None and cash_ratio < 0.3):
        return True, "Operating cash flow negative >=2 years and cash ratio < 0.3"
    # 3 totalDebt/EBITDA > 8
    debt = yfdata_info.get('totalDebt'); ebitda = yfdata_info.get('ebitda')
    if debt not in (None,0) and ebitda not in (None,0):
        if debt / ebitda > 8:
            return True, "Total debt / EBITDA > 8"
    return False, ""

# -------------------------
# Aggregation & veto composition
# -------------------------

WEIGHTS = {'technical':0.10, 'merrill':0.20, 'porter':0.10, 'value_chain':0.10, 'financial':0.50}
POSITION_MAP = [
    (95,100,'★★★★★ (Core holding)', 0.10),
    (85,94,'★★★★ (Focus allocation)', 0.07),
    (75,84,'★★★ (Moderate allocation)', 0.05),
    (65,74,'★★ (Watch)', 0.03),
    (0,64,'★ (Avoid)', 0.00)
]

def aggregate_scores(tech_pct, merrill_pct, porter_pct, value_chain_pct, financial_pct, veto_flags):
    # if financial veto, zero financial_pct
    if veto_flags.get('financial_veto'):
        financial_pct = 0.0
    final = (tech_pct * WEIGHTS['technical'] + 
             merrill_pct * WEIGHTS['merrill'] + 
             porter_pct * WEIGHTS['porter'] + 
             value_chain_pct * WEIGHTS['value_chain'] + 
             financial_pct * WEIGHTS['financial'])
    # final might be 0..100
    label = '★ (Avoid)'; pos=0.0
    for lo,hi,lab,p in POSITION_MAP:
        if lo <= final <= hi:
            label = lab; pos = p; break
    return round(final,2), label, pos

# -------------------------
# Visualizations
# -------------------------

def radar_plot(name, m, p, v, f, t):
    labels = ['Merrill','Porter','ValueChain','Financial','Technical']
    vals = [m,p,v,f,t]
    fig = go.Figure(data=go.Scatterpolar(r=vals, theta=labels, fill='toself', name=name))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])), showlegend=False, title=f"{name} - Component Radar")
    return fig

def bar_plot(name, m, p, v, f, t):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Merrill','Porter','ValueChain','Financial','Technical'], y=[m,p,v,f,t]))
    fig.update_layout(title=f"{name} - Component Scores", yaxis=dict(range=[0,100]))
    return fig

# -------------------------
# Streamlit UI & main
# -------------------------

with st.sidebar:
    st.header("Run settings")
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,TSLA")
    merrill_phase = st.selectbox("Merrill Phase", options=['recovery','overheat','stagflation','recession'], index=0)
    deepseek_key = st.text_input("DeepSeek API Key (optional for VC LLM)", type="password")
    ai_full = st.checkbox("Enable AI full-assessment (DeepSeek)", value=False)
    use_quantitative_vc = st.checkbox("Use Quantitative Value Chain (财务数据驱动)", value=True)
    st.markdown("---")
    st.subheader("China market support")
    china_support = st.checkbox("Support China tickers (auto-add .SS/.SZ)", value=False)
    china_default = st.selectbox("Default China exchange (used when unsure)", options=['Auto','Shanghai (.SS)','Shenzhen (.SZ)'], index=0)
    run_btn = st.button("Run Analysis")
    if st.button("Clear DeepSeek cache"):
        st.session_state['deepseek_cache'] = {}
        st.success("DeepSeek cache cleared. Re-run analysis to retry AI calls.")

st.markdown("### Instructions")
st.markdown("Enter one or more tickers (comma-separated). If DeepSeek API key provided, value-chain will be auto-scored by the LLM; otherwise heuristic + manual sliders available.")

if run_btn:
    raw_inputs = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if 'china_support' in globals() or True:
        # use the sidebar value to decide mapping; china_support variable exists in sidebar scope
        try:
            use_china = bool(china_support)
        except Exception:
            use_china = False
    else:
        use_china = False
    tickers = []
    for t in raw_inputs:
        if use_china:
            mapped = normalize_china_ticker(t, default_exchange=china_default)
            tickers.append(mapped.upper())
        else:
            tickers.append(t.strip().upper())
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        results = []
        progress = st.progress(0)
        for idx, tk in enumerate(tickers):
            progress.progress(int((idx+1)/len(tickers)*100))
            st.write(f"## Analyzing {tk}")
            try:
                yfdata = fetch_yf_data(tk)
            except Exception as e:
                st.error(f"Failed to fetch {tk}: {e}")
                continue
            
            # TECHNICAL
            tech_pct, tech_reason = technical_score_from_history(yfdata.get('history'))
            st.write(f"Technical: {tech_pct} ({tech_reason})")
            
            # MERRILL
            merrill_pct, merr_reason = merrill_score(yfdata.get('sector'), merrill_phase)
            merr_veto, merr_veto_reason = merrill_veto_check(yfdata.get('sector'), merrill_phase, None)
            st.write(f"Merrill: {merrill_pct} ({merr_reason})")
            if merr_veto:
                st.warning(f"Merrill veto triggered: {merr_veto_reason}")
            
            # PORTER
            porter = porter_auto(yfdata.get('sector'), yfdata.get('industry'), yfdata.get('summary'))
            porter_pct = porter['porter_pct']
            porter_veto, porter_veto_reason = porter_veto_check(porter['weighted_raw'])
            st.write(f"Porter (auto): {round(porter_pct,2)} | ratings: {porter['ratings']}")
            if porter_veto:
                st.warning(f"Porter veto: {porter_veto_reason}")
            
            # VALUE CHAIN - 新增定量分析选项
            vc_llm = None
            vc_quant = None
            vc_debug = None
            
            # 首先尝试定量分析
            if use_quantitative_vc:
                st.info("Running Quantitative Value Chain Analysis...")
                vc_quant = value_chain_quantitative_analysis(yfdata)
                main_scores = vc_quant['main_scores']
                support_scores = vc_quant['support_scores']
                explanation = vc_quant['explanation']
                st.success("Quantitative Value Chain Analysis Completed!")
            
            # 如果用户提供了DeepSeek API密钥，可以尝试LLM分析
            elif deepseek_key:
                st.info("Calling DeepSeek for Value Chain scoring (may take a few seconds)...")
                # use session cache to avoid repeated calls
                cache = st.session_state.get('deepseek_cache', {})
                if tk in cache and cache[tk].get('vc_debug'):
                    vc_debug = cache[tk]['vc_debug']
                else:
                    vc_debug = call_deepseek_debug(yfdata.get('summary',''), deepseek_key, path_hint=f"vc::{tk}")
                    cache.setdefault(tk, {})['vc_debug'] = vc_debug
                    st.session_state['deepseek_cache'] = cache
                
                # 解析LLM响应
                if vc_debug.get('parsed'):
                    vc_llm = vc_debug['parsed']
                    if isinstance(vc_llm, dict) and (vc_llm.get('main_scores') or vc_llm.get('support_scores')):
                        main_scores = vc_llm.get('main_scores', {})
                        support_scores = vc_llm.get('support_scores', {})
                        explanation = vc_llm.get('explanation','')
                    else:
                        # 如果LLM失败，使用定量分析作为备选
                        st.warning("LLM analysis failed, falling back to quantitative analysis")
                        vc_quant = value_chain_quantitative_analysis(yfdata)
                        main_scores = vc_quant['main_scores']
                        support_scores = vc_quant['support_scores']
                        explanation = vc_quant['explanation']
                else:
                    # LLM调用失败，使用定量分析
                    st.warning("DeepSeek call failed, using quantitative analysis")
                    vc_quant = value_chain_quantitative_analysis(yfdata)
                    main_scores = vc_quant['main_scores']
                    support_scores = vc_quant['support_scores']
                    explanation = vc_quant['explanation']
            
            else:
                # 没有API密钥，使用定量分析
                st.info("Using Quantitative Value Chain Analysis (no API key provided)")
                vc_quant = value_chain_quantitative_analysis(yfdata)
                main_scores = vc_quant['main_scores']
                support_scores = vc_quant['support_scores']
                explanation = vc_quant['explanation']
            
            # 显示价值链分析结果
            st.subheader("Value Chain Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Main Activities Scores (1-10):")
                st.json(main_scores)
                st.write("Support Activities Scores (1-10):")
                st.json(support_scores)
            
            with col2:
                st.write("Analysis Method:")
                if vc_quant:
                    st.success("📊 Quantitative Analysis (Financial Data Driven)")
                elif vc_llm:
                    st.info("🤖 LLM Analysis (DeepSeek)")
                
                if explanation:
                    st.write("Explanation:")
                    st.write(explanation)
            
            # 允许手动覆盖
            st.subheader("Manual Override (Optional)")
            if st.checkbox(f"Enable manual override for {tk}", key=f"override_{tk}"):
                manual_main = {}
                manual_support = {}
                
                st.write("Main Activities:")
                for k, v in main_scores.items():
                    manual_main[k] = st.slider(f"{k}", 1, 10, v, key=f"main_{tk}_{k}")
                
                st.write("Support Activities:")
                for k, v in support_scores.items():
                    manual_support[k] = st.slider(f"{k}", 1, 10, v, key=f"support_{tk}_{k}")
                
                if st.button(f"Apply Manual Scores for {tk}"):
                    main_scores = manual_main
                    support_scores = manual_support
                    st.success("Manual scores applied!")
            
            # 计算价值链分数
            vc_pct, main_avg, support_avg = value_chain_from_scores(main_scores, support_scores)
            st.write(f"Value Chain Score: {round(vc_pct,2)}/100 (Main Avg: {round(main_avg,2)}, Support Avg: {round(support_avg,2)})")
            
            vc_veto, vc_veto_reason = value_chain_veto_check(main_scores, support_scores)
            if vc_veto:
                st.warning(f"Value Chain veto: {vc_veto_reason}")
            
            # 其余部分保持不变（AI全部分析、财务分析等）
            # ... [由于篇幅限制，这里省略了剩余代码，实际使用时需要保留]
            
            # 这里继续原有的财务分析和聚合逻辑
            # FINANCIAL ANALYSIS
            st.subheader("Financial Analysis (detailed)")
            op = compute_operational_scores(yfdata)
            bk = compute_bankruptcy_scores(yfdata)
            vl = compute_valuation_scores(yfdata)
            
            # financial composite before veto: operation50% + bankruptcy20% + valuation30%
            financial_pct = op['operation_score'] * 0.5 + bk['bankruptcy_score'] * 0.2 + vl['val_margin'] * 0.3
            
            # check hard vetoes
            fin_veto, fin_veto_reason = financial_hard_veto(bk, yfdata.get('info',{}), yfdata.get('q_income'))
            if fin_veto:
                st.error(f"Financial HARD VETO triggered: {fin_veto_reason} -> financial score will be set to 0")
                financial_pct_before = financial_pct
                financial_pct = 0.0
            
            # 显示财务详情
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Operation Score", f"{round(op['operation_score'],2)}")
                st.write(f"ROE: {op.get('roe_pct', 'N/A')}%")
            with col2:
                st.metric("Bankruptcy Score", f"{round(bk['bankruptcy_score'],2)}")
                st.write(f"Interest Coverage: {bk.get('interest_coverage', 'N/A')}")
            with col3:
                st.metric("Valuation Score", f"{round(vl['val_margin'],2)}")
                st.write(f"P/E: {vl.get('pe', 'N/A')}")
            
            # 聚合最终分数
            veto_flags = {
                'merrill_veto': merr_veto,
                'porter_veto': porter_veto,
                'value_chain_veto': vc_veto,
                'financial_veto': fin_veto
            }
            
            final_pct, label, position = aggregate_scores(tech_pct, merrill_pct, porter_pct, vc_pct, financial_pct, veto_flags)
            
            # 显示最终结果
            st.subheader(f"🎯 Final Score for {tk}")
            st.metric("Overall Score", f"{final_pct}/100", f"{label}")
            st.metric("Suggested Position", f"{position*100:.1f}%")
            
            # 可视化
            st.plotly_chart(radar_plot(tk, merrill_pct, porter_pct, vc_pct, financial_pct, tech_pct), use_container_width=True)
            
            # 保存结果
            results.append({
                'ticker': tk,
                'final_score': final_pct,
                'rating': label,
                'position_cap': f"{position*100:.1f}%",
                'technical': tech_pct,
                'merrill': merrill_pct,
                'porter': porter_pct,
                'value_chain': vc_pct,
                'financial': financial_pct
            })
        
        # 显示批量结果摘要
        if results:
            st.header("📊 Batch Results Summary")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)
            
            # 下载结果
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="investment_scores.csv",
                mime="text/csv"
            )
        
        progress.empty()
