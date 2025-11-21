# FinGuide.py
# FinGuide — Smart Financial Assistance (single-file Streamlit app)
# Accepts CSV / Excel / PDF bank statements, cleans, shows KPIs, charts, and gives 1-step ML predictions.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re
from datetime import datetime
from textwrap import dedent

# Optional libs
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    SKL_AVAILABLE = True
except Exception:
    SKL_AVAILABLE = False

# ---------- Page config ----------
st.set_page_config(page_title="FinGuide — Smart Financial Assistance",
                   page_icon="🧭",
                   layout="wide")

# ---------- Small CSS for a clean website-like look ----------
st.markdown("""
    <style>
      .hero { background: #09b5a8; padding: 20px; border-radius: 8px; color: white; margin-bottom:10px; }
      .card { background: #0f1721; border-radius:8px; padding:12px; color: #e6eef8; margin-bottom:12px; }
      body { background: #07101a; color: #e6eef8; }
      .muted { color: rgba(230,238,248,0.7); font-size:13px; }
      .kpi { background: linear-gradient(180deg,#08313a,#0b2a36); padding:10px; border-radius:8px; text-align:center; color:white; }
      .small { font-size:13px; color: #cbd5e1; }
      .stDownloadButton>button { background: #09b5a8; color: white; }
    </style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="hero"><h1>FinGuide — Smart Financial Assistance</h1>'
            '<div class="small">Upload financial data → Clean → KPIs → Charts → Predictions</div></div>',
            unsafe_allow_html=True)

# ---------- Tabs ----------
tabs = st.tabs(["Upload", "Cleaned Data", "KPIs", "Charts", "Predictions", "Power BI"])

# ---------- Utility: safe number parse ----------
def to_numeric_safe(series):
    s = series.astype(str).str.replace(",", "").str.replace("₹", "").str.replace("(", "-").str.replace(")", "")
    return pd.to_numeric(s, errors="coerce")

# ---------- PDF parsing helper ----------
def parse_pdf_table_like(file_bytes):
    """
    Try to extract table-like rows from PDFs:
    - If pdfplumber available, try to read tables then fallback to text parsing.
    - Return DataFrame or None.
    """
    text = ""
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                # try to extract tables from each page and concat
                dfs = []
                for page in pdf.pages:
                    try:
                        tables = page.extract_tables()
                        for t in tables:
                            if t and len(t) > 0:
                                df = pd.DataFrame(t[1:], columns=t[0])
                                dfs.append(df)
                    except Exception:
                        pass
                if dfs:
                    return pd.concat(dfs, ignore_index=True)
                # else fallback to text
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception:
            # fallback to text extraction without tables
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception:
                text = ""
    else:
        # fallback: try to extract plain text using naive approach (PyPDF2 not required)
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for p in reader.pages:
                try:
                    text += p.extract_text() or ""
                except Exception:
                    pass
        except Exception:
            text = ""

    # If we have text, attempt to find typical bank columns: Date Description Debit Credit Balance
    if text:
        # Normalize
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        # Attempt to capture rows with a date at start
        rows = []
        date_re = re.compile(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})|(\d{4}[-/]\d{2}[-/]\d{2})')
        for ln in lines:
            if date_re.search(ln):
                rows.append(ln)
        # Try to split using multi spaces
        parsed = []
        for r in rows:
            parts = re.split(r'\s{2,}', r)
            if len(parts) >= 3:
                parsed.append(parts)
            else:
                # try space split and keep last 3 as amounts
                p2 = r.split()
                if len(p2) >= 4:
                    # date, description, maybe debit/credit/balance
                    parsed.append([p2[0], " ".join(p2[1:-2]), p2[-2], p2[-1]])
        if parsed:
            # Make columns flexible
            max_cols = max(len(p) for p in parsed)
            cols = [f"col_{i}" for i in range(max_cols)]
            df = pd.DataFrame([p + [""]*(max_cols-len(p)) for p in parsed], columns=cols)
            return df
    return None

# ---------- Generic cleaning function ----------
def clean_general(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)
    # try to coerce numeric-ish columns commonly named
    for c in df.columns:
        if df[c].dtype == object:
            # if many numeric-like entries, coerce
            sample = df[c].astype(str).str.replace(",", "").str.replace("₹", "").str.replace("(", "-").str.replace(")", "")
            num_count = sample.str.match(r"^-?\d+(\.\d+)?$").sum()
            if num_count / max(1, len(sample)) > 0.6:
                df[c] = pd.to_numeric(sample, errors="coerce")
    # create Profit if Revenue & Expense exist
    rev = next((c for c in df.columns if "revenue" in c.lower() or "credit"==c.lower() or "credit" in c.lower()), None)
    exp = next((c for c in df.columns if "expense" in c.lower() or "debit"==c.lower() or "debit" in c.lower() or "cost" in c.lower()), None)
    if rev and exp and "Profit" not in df.columns:
        # avoid string subtraction error
        try:
            df["Profit"] = pd.to_numeric(df[rev], errors="coerce") - pd.to_numeric(df[exp], errors="coerce")
        except Exception:
            pass
    # Try to parse dates
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            pass
    return df

# ---------- Helper: detect file type and load ----------
def load_file(uploaded_file):
    filename = uploaded_file.name.lower()
    data = None
    if filename.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif filename.endswith(".xls") or filename.endswith(".xlsx") or filename.endswith(".xlsb"):
        try:
            data = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception:
            data = pd.read_excel(uploaded_file)
    elif filename.endswith(".pdf"):
        bytes_data = uploaded_file.read()
        # Try parse tables first
        df = parse_pdf_table_like(bytes_data)
        if df is not None:
            data = df
        else:
            st.warning("PDF uploaded but we couldn't parse clean tables automatically. We'll extract text and try to infer columns.")
            # fallback: store raw text as single-column DF
            text = ""
            if PDFPLUMBER_AVAILABLE:
                try:
                    with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
                        for p in pdf.pages:
                            text += p.extract_text() or ""
                except Exception:
                    text = ""
            else:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
                    for p in reader.pages:
                        text += p.extract_text() or ""
                except Exception:
                    text = ""
            data = pd.DataFrame({"raw_text":[text]})
    else:
        st.error("Unsupported file type. Please upload CSV, XLSX, or PDF.")
    return data

# ---------- Sample data helper ----------
def sample_data():
    data = {
        "Date": pd.date_range("2024-01-01", periods=12, freq="M"),
        "Shop_Name": ["Om Electronics", "City Electronics", "Mass Traders", "Om Electronics"] * 3,
        "Category": ["Mobile", "Laptop", "TV", "Accessories"] * 3,
        "Product": ["Phone A", "Laptop B", "LED TV 42", "Earphones"] * 3,
        "Units_Sold": [10, 5, 3, 30, 12, 7, 4, 22, 15, 9, 6, 18],
        "Revenue": [120000, 250000, 60000, 30000, 140000, 180000, 50000, 80000, 150000, 90000, 60000, 120000],
        "Expense": [90000, 200000, 45000, 20000, 100000, 140000, 35000, 60000, 120000, 70000, 40000, 90000],
    }
    return pd.DataFrame(data)

# ---------- UPLOAD Tab ----------
with tabs[0]:
    st.markdown("<div class='card'><h3>📥 Upload your financial file</h3>", unsafe_allow_html=True)
    st.markdown('<div class="muted">Supported: CSV, Excel (XLS/XLSX), and PDF bank statements.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drag & drop file here (or click Browse)", type=["csv","xlsx","xls","pdf"], accept_multiple_files=False)
    use_sample = st.button("Use sample sales data (quick demo)")

    if use_sample:
        df_raw = sample_data()
        st.success("Sample data loaded.")
        st.session_state["df_raw"] = df_raw
        st.session_state["source_type"] = "sales"
    elif uploaded:
        try:
            df_loaded = load_file(uploaded)
            if df_loaded is None:
                st.error("Could not load file.")
            else:
                st.session_state["df_raw"] = df_loaded
                # naive detection: if columns contain Debit/Credit likely bank
                cols = [c.lower() for c in df_loaded.columns]
                if any("debit" in c for c in cols) or any("credit" in c for c in cols) or any("balance" in c for c in cols):
                    st.session_state["source_type"] = "bank"
                else:
                    st.session_state["source_type"] = "sales"
                st.success("File loaded. Go to Cleaned Data tab.")
        except Exception as e:
            st.error(f"Load failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- CLEANED DATA Tab ----------
with tabs[1]:
    st.markdown("<div class='card'><h3>🧹 Cleaned Data Preview</h3>", unsafe_allow_html=True)
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None:
        st.info("No file loaded. Upload in Upload tab or use sample.")
    else:
        try:
            df_clean = clean_general(df_raw)
            st.session_state["df_clean"] = df_clean
            st.dataframe(df_clean.head(50), use_container_width=True, height=360)
            csv = df_clean.to_csv(index=False)
            st.download_button("⬇ Download cleaned CSV", data=csv, file_name="fin_guide_cleaned.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Cleaning failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- KPIs Tab ----------
with tabs[2]:
    st.markdown("<div class='card'><h3>📌 KPIs</h3>", unsafe_allow_html=True)
    df_clean = st.session_state.get("df_clean", None)
    source = st.session_state.get("source_type", None)
    if df_clean is None:
        st.info("No cleaned data to show KPIs.")
    else:
        # Try detect columns
        cols = [c.lower() for c in df_clean.columns]
        rev = next((c for c in df_clean.columns if "revenue" in c.lower() or "credit" in c.lower()), None)
        exp = next((c for c in df_clean.columns if "expense" in c.lower() or "debit" in c.lower() or "cost" in c.lower()), None)
        bal = next((c for c in df_clean.columns if "balance" in c.lower()), None)
        prof = next((c for c in df_clean.columns if "profit" in c.lower()), None)

        # convert to numeric if possible
        if rev is not None:
            df_clean[rev] = to_numeric_safe(df_clean[rev])
        if exp is not None:
            df_clean[exp] = to_numeric_safe(df_clean[exp])
        if bal is not None:
            df_clean[bal] = to_numeric_safe(df_clean[bal])

        # Branch KPIs based on detected type
        if source == "bank" or (rev is None and exp is None and bal is not None):
            # BANK STUDENT KPIs
            total_credit = float(df_clean[rev].sum()) if rev and pd.api.types.is_numeric_dtype(df_clean[rev]) else 0.0
            total_debit = float(df_clean[exp].sum()) if exp and pd.api.types.is_numeric_dtype(df_clean[exp]) else 0.0
            net = total_credit - total_debit
            highest_credit = float(df_clean[rev].max()) if rev and pd.api.types.is_numeric_dtype(df_clean[rev]) else None
            highest_debit = float(df_clean[exp].max()) if exp and pd.api.types.is_numeric_dtype(df_clean[exp]) else None
            # Monthly averages if date exists
            date_col = next((c for c in df_clean.columns if "date" in c.lower()), None)
            avg_month_income = avg_month_expense = None
            if date_col and pd.api.types.is_datetime64_any_dtype(df_clean[date_col]):
                monthly = df_clean.set_index(date_col).resample("M").agg({rev: "sum", exp: "sum"})
                avg_month_income = monthly[rev].mean() if rev in monthly else None
                avg_month_expense = monthly[exp].mean() if exp in monthly else None

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='kpi'><h4>Total Income</h4><h2>₹{int(total_credit):,}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='kpi'><h4>Total Expense</h4><h2>₹{int(total_debit):,}</h2></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='kpi'><h4>Net Savings</h4><h2>₹{int(net):,}</h2></div>", unsafe_allow_html=True)
            st.write("")
            st.markdown("**Other useful stats**")
            st.write(f"- Highest credit (single txn): ₹{int(highest_credit):,}" if highest_credit else "- Highest credit: N/A")
            st.write(f"- Highest debit (single txn): ₹{int(highest_debit):,}" if highest_debit else "- Highest debit: N/A")
            if avg_month_income is not None:
                st.write(f"- Avg monthly income (approx): ₹{int(avg_month_income):,}")
            if avg_month_expense is not None:
                st.write(f"- Avg monthly expense (approx): ₹{int(avg_month_expense):,}")

        else:
            # SHOP KPIs
            total_rev = float(df_clean[rev].sum()) if rev and pd.api.types.is_numeric_dtype(df_clean[rev]) else None
            total_exp = float(df_clean[exp].sum()) if exp and pd.api.types.is_numeric_dtype(df_clean[exp]) else None
            if prof and pd.api.types.is_numeric_dtype(df_clean[prof]):
                total_profit = float(df_clean[prof].sum())
            elif rev is not None and exp is not None:
                total_profit = total_rev - total_exp
            else:
                total_profit = None
            margin = round(total_profit / total_rev * 100, 2) if total_rev and total_profit is not None else None

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='kpi'><h4>Total Revenue</h4><h2>₹{int(total_rev):,}</h2></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='kpi'><h4>Total Expense</h4><h2>₹{int(total_exp):,}</h2></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='kpi'><h4>Total Profit</h4><h2>₹{int(total_profit):,}</h2></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='kpi'><h4>Profit Margin</h4><h2>{margin if margin is not None else 'N/A'}%</h2></div>", unsafe_allow_html=True)
            st.write("")
            # Breakdown by category if possible
            group_col = next((c for c in df_clean.columns if "category" in c.lower()), None)
            if group_col and rev:
                st.markdown("**Revenue by Category**")
                cat = df_clean.groupby(group_col)[rev].sum().reset_index().sort_values(by=rev, ascending=False)
                st.table(cat.head(8))
            else:
                st.write("No category breakdown found.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- CHARTS Tab ----------
with tabs[3]:
    st.markdown("<div class='card'><h3>📊 Charts</h3>", unsafe_allow_html=True)
    df_clean = st.session_state.get("df_clean", None)
    source = st.session_state.get("source_type", None)
    if df_clean is None:
        st.info("No data to chart.")
    else:
        # find columns
        rev = next((c for c in df_clean.columns if "revenue" in c.lower() or "credit" in c.lower()), None)
        exp = next((c for c in df_clean.columns if "expense" in c.lower() or "debit" in c.lower()), None)
        bal = next((c for c in df_clean.columns if "balance" in c.lower()), None)
        date_col = next((c for c in df_clean.columns if "date" in c.lower()), None)

        # ensure numeric
        if rev is not None:
            df_clean[rev] = to_numeric_safe(df_clean[rev])
        if exp is not None:
            df_clean[exp] = to_numeric_safe(df_clean[exp])
        if bal is not None:
            df_clean[bal] = to_numeric_safe(df_clean[bal])

        # Simple charts: Income vs Expense pie, line of balance or revenue trend
        colA, colB = st.columns(2)
        with colA:
            st.markdown("### Income vs Expense (clear simple pie)")
            income_val = float(df_clean[rev].sum()) if rev and pd.api.types.is_numeric_dtype(df_clean[rev]) else 0.0
            expense_val = float(df_clean[exp].sum()) if exp and pd.api.types.is_numeric_dtype(df_clean[exp]) else 0.0
            # avoid NaN; make small fallback
            inc = max(income_val, 0.0)
            exc = max(expense_val, 0.0)
            if inc == 0 and exc == 0:
                st.info("No numerical income/expense found to build a pie chart.")
            else:
                fig, ax = plt.subplots(figsize=(4,3))
                labels = ["Income", "Expense"]
                sizes = [inc, exc]
                # hide tiny slices
                sizes = [s if s > 0 else 0.0 for s in sizes]
                colors = ["#06b6d4", "#f97316"]
                ax.pie(sizes, labels=labels, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', colors=colors, startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

        with colB:
            st.markdown("### Trend (Line chart)")
            if date_col and pd.api.types.is_datetime64_any_dtype(df_clean[date_col]) and rev:
                try:
                    monthly = df_clean.set_index(date_col).resample("M")[rev].sum()
                    fig2, ax2 = plt.subplots(figsize=(6,3))
                    ax2.plot(monthly.index, monthly.values, marker="o", color="#06b6d4")
                    ax2.set_xlabel("Month")
                    ax2.set_ylabel("Revenue")
                    fig2.autofmt_xdate()
                    st.pyplot(fig2)
                except Exception:
                    st.info("Could not build monthly trend. Try date parsing.")
            elif rev:
                # fallback: index-based trend
                fig2, ax2 = plt.subplots(figsize=(6,3))
                ax2.plot(df_clean.index, df_clean[rev].fillna(0), marker="o", color="#06b6d4")
                ax2.set_xlabel("Record #")
                ax2.set_ylabel("Revenue")
                st.pyplot(fig2)
            else:
                st.info("No revenue/credit column found to plot trends.")

        # If balance exists, show balance line
        if bal:
            st.markdown("#### Balance over time (if present)")
            fig3, ax3 = plt.subplots(figsize=(10,2.5))
            ax3.plot(df_clean.index, df_clean[bal].fillna(method="ffill").fillna(0), color="#10b981")
            ax3.set_xlabel("Record #")
            ax3.set_ylabel("Balance")
            st.pyplot(fig3)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTIONS Tab ----------
with tabs[4]:
    st.markdown("<div class='card'><h3>🔮 Predictions</h3>", unsafe_allow_html=True)
    df_clean = st.session_state.get("df_clean", None)
    if df_clean is None:
        st.info("No data to predict on.")
    else:
        rev = next((c for c in df_clean.columns if "revenue" in c.lower() or "credit" in c.lower()), None)
        date_col = next((c for c in df_clean.columns if "date" in c.lower()), None)

        if not SKL_AVAILABLE:
            st.warning("scikit-learn not installed. Predictions disabled. Install with `pip install scikit-learn`.")
        elif rev is None:
            st.info("No revenue/credit column found for prediction.")
        else:
            # Build monthly aggregate if date exists; else use index-based series
            try:
                series = None
                if date_col and pd.api.types.is_datetime64_any_dtype(df_clean[date_col]):
                    monthly = df_clean.set_index(date_col).resample("M")[rev].sum().dropna()
                    if len(monthly) >= 1:
                        series = monthly
                if series is None:
                    # fallback: use record-based sum/resampling
                    tmp = df_clean[rev].fillna(0)
                    if tmp.sum() == 0 or len(tmp) < 2:
                        st.info("Not enough numeric revenue data for prediction.")
                    else:
                        series = pd.Series(tmp.values, index=pd.RangeIndex(len(tmp)))

                if series is not None and len(series) >= 2:
                    # prepare X as 0..n-1
                    X = np.arange(len(series)).reshape(-1,1)
                    y = series.values
                    model = LinearRegression()
                    model.fit(X, y)
                    next_idx = np.array([[len(series)]])
                    pred = model.predict(next_idx)[0]
                    st.metric("Predicted next period revenue (one-step)", f"₹{int(pred):,}")
                    # plot series + prediction
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(np.arange(len(series)), y, marker="o", label="Actual")
                    ax.plot(next_idx.flatten(), pred, marker="X", color="red", markersize=9, label="Prediction")
                    ax.set_xlabel("Period index")
                    ax.set_ylabel("Revenue")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("Not enough aggregated points to train a regression (need >=2).")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- POWER BI Tab ----------
with tabs[5]:
    st.markdown("<div class='card'><h3>Power BI</h3>", unsafe_allow_html=True)
    st.write("You can export the cleaned CSV and import into Power BI Desktop for richer dashboards.")
    if st.button("Download cleaned CSV for Power BI") and st.session_state.get("df_clean") is not None:
        st.download_button("Download CSV", data=st.session_state["df_clean"].to_csv(index=False), file_name="fin_guide_cleaned.csv")
    st.markdown("<div class='muted'>powerBi</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.write("")
st.markdown("<div class='muted'>FinGuide prototype</div>", unsafe_allow_html=True)
