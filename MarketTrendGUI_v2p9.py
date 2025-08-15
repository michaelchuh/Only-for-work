
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Trend Analyzer — v2.9
- NEW: Cascading (nested) selectors on Sales Trend:
  Make → Model → Body style (Proton) → Seats → Drivetrain type → Pure electric range (kms)
  Each downstream list only shows values valid under all upstream selections.
- Keeps v2.8 capabilities:
  • Sales Trend: Country VS + time granularity + Details panel (hover/click)
  • Market Structure: secondary filter + Details panel
  • Competitor Compare: Details panel with drilldown
"""
import os, re, sys
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import mplcursors
except Exception:
    mplcursors = None

# ---------- Month detection ----------
MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct":10, "october":10,
    "nov":11, "november":11,
    "dec":12, "december":12,
}

def _normalize_colname(x: Union[str, object]) -> str:
    if isinstance(x, pd.Timestamp):
        return x.strftime("%b '%y")
    s = str(x).strip()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s)
    return s

def _is_total_or_ytd(s: str) -> bool:
    s_low = s.lower()
    return ("tot" in s_low) or ("ytd" in s_low) or ("year to date" in s_low)

def _parse_month_from_header(s: str) -> Optional[Tuple[int,int]]:
    if not s or _is_total_or_ytd(s):
        return None
    s0 = _normalize_colname(s).lower()

    m = re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:uary|ch|il|e|y|ust|tember|ober|ember)?[ \-_/\.]*'?(\\d{2}|\\d{4})$".replace("\\\\","\\"), s0)
    if m:
        mon = MONTH_NAMES[m.group(1)]; yr = int(m.group(2))
        if yr < 100: yr += 2000 if yr <= 79 else 1900
        return (yr, mon)

    m = re.match(r"^(\\d{4})[ \-_/\.]*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:uary|ch|il|e|y|ust|tember|ober|ember)?$".replace("\\\\","\\"), s0)
    if m:
        yr = int(m.group(1)); mon = MONTH_NAMES[m.group(2)]
        return (yr, mon)

    m = re.match(r"^(\\d{4})[ \\-_/\\.]([01]?\\d)$".replace("\\\\","\\"), s0)
    if m:
        yr = int(m.group(1)); mon = int(m.group(2))
        if 1 <= mon <= 12: return (yr, mon)

    m = re.match(r"^([01]?\\d)[ \\-_/\\.]([12]\\d{3})$".replace("\\\\","\\"), s0)
    if m:
        mon = int(m.group(1)); yr = int(m.group(2))
        if 1 <= mon <= 12: return (yr, mon)

    m = re.match(r"^(\\d{4})\\s*(\\d{2})$".replace("\\\\","\\"), s0)
    if m:
        yr = int(m.group(1)); mon = int(m.group(2))
        if 1 <= mon <= 12: return (yr, mon)

    m = re.match(r"^(\\d{4})\\s*年\\s*(\\d{1,2})\\s*月$".replace("\\\\","\\"), s0)
    if m:
        yr = int(m.group(1)); mon = int(m.group(2))
        if 1 <= mon <= 12: return (yr, mon)

    try:
        dt = pd.to_datetime(s0, errors="raise", dayfirst=False)
        return (int(dt.year), int(dt.month))
    except Exception:
        pass
    return None

def detect_month_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if _parse_month_from_header(str(c)) is not None]

# ---------- Period aggregation ----------
def _period_key(colname: str, granularity: str) -> Optional[Tuple[str, Tuple[int,int]]]:
    ym = _parse_month_from_header(colname)
    if ym is None:
        return None
    y, m = ym
    if granularity == "Monthly":
        return (f"{y}-{m:02d}", (y, m))
    if granularity == "Quarterly":
        q = (m-1)//3 + 1
        return (f"{y} Q{q}", (y, q))
    return (f"{y}", (y, 1))

def aggregate_time(df_vals: pd.DataFrame, months: List[str], granularity: str) -> Tuple[pd.DataFrame, List[str]]:
    if granularity == "Monthly":
        return df_vals[months].copy(), months
    groups: Dict[str, List[str]] = {}
    order: Dict[str, Tuple[int,int]] = {}
    for c in months:
        out = _period_key(c, granularity)
        if out is None: 
            continue
        label, key = out
        groups.setdefault(label, []).append(c)
        order[label] = key
    labels = sorted(groups.keys(), key=lambda k: order[k])
    agg = pd.DataFrame(index=df_vals.index)
    for lab in labels:
        cols = [c for c in groups[lab] if c in df_vals.columns]
        if cols:
            agg[lab] = df_vals[cols].sum(axis=1)
    return agg, labels

def metric_transform_period(df_ser: pd.DataFrame, metric: str, granularity: str) -> pd.DataFrame:
    periods = df_ser.columns[1:]
    vals = df_ser[periods].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out = vals.copy()
    if metric in ("YoY %", "MoM %"):
        if granularity == "Monthly":
            yoy_lag, prev_lag = 12, 1
        elif granularity == "Quarterly":
            yoy_lag, prev_lag = 4, 1
        else:
            yoy_lag, prev_lag = 1, 1
        lag = yoy_lag if metric == "YoY %" else prev_lag
        for i in range(len(periods)):
            if i-lag >= 0:
                prev = vals.iloc[:,i-lag]
                out.iloc[:,i] = np.where(prev==0, np.nan, (vals.iloc[:,i]/prev - 1.0) * 100.0)
            else:
                out.iloc[:,i] = np.nan
    return pd.concat([df_ser[["Series"]], out], axis=1)

def safe_xtick_step(n: int) -> int:
    return max(1, n // 12) if n > 0 else 1

# ---------- Reusable UI: details panel ----------
def build_detail_panel(parent) -> Dict[str, object]:
    card = ttk.LabelFrame(parent, text="Details", padding=8)
    card.pack(fill=tk.BOTH, expand=True)
    kv = ttk.Treeview(card, columns=("k","v"), show="headings", height=8)
    kv.heading("k", text="Field"); kv.heading("v", text="Value")
    kv.column("k", width=140, anchor="w"); kv.column("v", width=180, anchor="w")
    kv.pack(fill=tk.X, expand=False, pady=(0,8))

    br = ttk.Treeview(card, columns=("name","val"), show="headings", height=12)
    br.heading("name", text="Breakdown"); br.heading("val", text="Units / %")
    br.column("name", width=200, anchor="w"); br.column("val", width=100, anchor="e")
    br.pack(fill=tk.BOTH, expand=True)

    status = tk.StringVar(value="Hover or click a point…")
    ttk.Label(card, textvariable=status, anchor="w").pack(fill=tk.X, pady=(6,0))
    return {"kv":kv, "br":br, "status":status}

def clear_detail(panel):
    panel["kv"].delete(*panel["kv"].get_children())
    panel["br"].delete(*panel["br"].get_children())

def push_detail(panel, info: Dict[str, Union[str,float]], breakdown: Optional[pd.DataFrame] = None):
    clear_detail(panel)
    for k, v in info.items():
        if isinstance(v,(int,float,np.floating)) and not pd.isna(v):
            panel["kv"].insert("", "end", values=(k, f"{v:,.0f}"))
        else:
            panel["kv"].insert("", "end", values=(k, v))
    if breakdown is not None and not breakdown.empty:
        for _, r in breakdown.iterrows():
            val = r.iloc[1]
            if isinstance(val,(int,float,np.floating)):
                panel["br"].insert("", "end", values=(str(r.iloc[0]), f"{float(val):,.0f}"))
            else:
                panel["br"].insert("", "end", values=(str(r.iloc[0]), str(val)))
        panel["status"].set("")
    else:
        panel["status"].set("")

# -------------- App --------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Market Trend Analyzer — v2.9")
        self.geometry("1660x980")
        self.minsize(1300, 780)

        self.df: Optional[pd.DataFrame] = None
        self.months: List[str] = []
        self._last_series = None

        # cascade config (Sales Trend)
        self.CASCADE_ORDER = ["Make", "Model", "Body style (Proton)", "Seats", "Drivetrain type", "Pure electric range (kms)"]

        self._build_ui()

        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            self.load_file(sys.argv[1])

    # ---------------- UI ----------------
    def _build_ui(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Excel…", command=self.on_open)
        filemenu.add_separator()
        filemenu.add_command(label="Export Chart as PNG…", command=self.on_export_png)
        filemenu.add_command(label="Export Series as CSV…", command=self.on_export_csv)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        toolsmenu = tk.Menu(menubar, tearoff=0)
        toolsmenu.add_command(label="Show column headers", command=self.on_show_headers)
        toolsmenu.add_command(label="Select month columns…", command=self.on_pick_months)
        menubar.add_cascade(label="Tools", menu=toolsmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.on_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        self.path_var = tk.StringVar(value="(No file loaded)")
        ttk.Label(top, text="Data file:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.path_var, width=120, anchor="w").pack(side=tk.LEFT, padx=(6, 10))
        ttk.Button(top, text="Open Excel…", command=self.on_open).pack(side=tk.LEFT)

        self.tabs = ttk.Notebook(self); self.tabs.pack(fill=tk.BOTH, expand=True)

        # Tab 1 — Sales Trend (w/ details)
        self.tab_trend = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_trend, text="Sales Trend")
        self._build_trend(self.tab_trend)

        # Tab 2 — Market Structure (w/ details)
        self.tab_struct = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_struct, text="Market Structure")
        self._build_structure(self.tab_struct)

        # Tab 3 — Competitor Compare (w/ details)
        self.tab_comp = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_comp, text="Competitor Compare")
        self._build_comp(self.tab_comp)

    # ---------- helpers ----------
    def _slice_months(self, start: str, end: str) -> List[str]:
        if not self.months: return []
        s = self.months.index(start) if start in self.months else 0
        e = self.months.index(end) if end in self.months else len(self.months)-1
        if s>e: s,e = e,s
        return self.months[s:e+1]

    def _get_selected(self, lb: tk.Listbox) -> Optional[List[str]]:
        if lb is None or str(lb['state']) != 'normal':
            return None
        vals = [lb.get(i) for i in lb.curselection()]
        if not vals or "(All)" in vals:
            return None
        return vals

    def _apply_filters(self, df: pd.DataFrame, filter_map: Dict[str, tk.Listbox], skip_cols: Optional[List[str]]=None) -> pd.DataFrame:
        skip = set(skip_cols or [])
        for col, lb in filter_map.items():
            if col in skip: 
                continue
            if col not in df.columns or str(lb['state']) != 'normal': continue
            vals = self._get_selected(lb)
            if vals is None: continue
            df = df[df[col].astype(str).isin(vals)]
        return df

    def _aggregate(self, df: pd.DataFrame, group_col: Optional[str], months: List[str]) -> pd.DataFrame:
        if df.empty: return pd.DataFrame(columns=["Series"]+months)
        if group_col is None or group_col == "Total":
            series = df[months].sum(axis=0).to_frame().T
            series.insert(0, "Series", "Total")
            return series
        g = df.groupby(group_col)[months].sum().reset_index().rename(columns={group_col:"Series"})
        return g

    def _attach_hover(self, ax, lines, labels, unit_label, cb=None):
        if mplcursors is None:
            return
        try:
            cursor = mplcursors.cursor(lines, hover=True)
            @cursor.connect("add")
            def _(sel):
                x, y = sel.target
                idx = int(round(x))
                idx = max(0, min(idx, len(labels)-1))
                label = sel.artist.get_label()
                txt = f"{label}\\n{labels[idx]}\\n{y:.2f}%" if unit_label=="%" else f"{label}\\n{labels[idx]}\\n{y:,.0f}"
                sel.annotation.set(text=txt)
                if cb: 
                    try: cb(label, idx)
                    except Exception: pass
        except Exception:
            pass

    def _safe_xtick_step(self, n: int) -> int:
        return safe_xtick_step(n)

    # ---------- Month picker ----------
    def on_pick_months(self):
        if self.df is None:
            messagebox.showinfo("Info", "Open a file first."); return
        top = tk.Toplevel(self)
        top.title("Select month columns"); top.geometry("520x520")
        ttk.Label(top, text="请选择月份列（多选）：").pack(anchor="w", padx=8, pady=(8,2))
        frm = ttk.Frame(top); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        lb = tk.Listbox(frm, selectmode=tk.MULTIPLE, exportselection=False)
        sb = ttk.Scrollbar(frm, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.LEFT, fill=tk.Y)

        cols = [str(c) for c in self.df.columns]
        for i, c in enumerate(cols):
            lb.insert(tk.END, c)
            if c in self.months:
                lb.selection_set(i)

        btns = ttk.Frame(top); btns.pack(fill=tk.X, padx=8, pady=8)
        def select_all(): lb.selection_set(0, tk.END)
        def select_none(): lb.selection_clear(0, tk.END)
        ttk.Button(btns, text="全选", command=select_all).pack(side=tk.LEFT)
        ttk.Button(btns, text="清空", command=select_none).pack(side=tk.LEFT, padx=6)

        def apply_and_close():
            picks = [lb.get(i) for i in lb.curselection()]
            if not picks:
                messagebox.showwarning("No selection", "请至少选择一列作为月份列"); return
            self.months = picks
            # refresh comboboxes
            self.trend_start_cb["values"]=self.months; self.trend_end_cb["values"]=self.months
            self.struct_start_cb["values"]=self.months; self.struct_end_cb["values"]=self.months
            self.comp_start_cb["values"]=self.months; self.comp_end_cb["values"]=self.months
            self.trend_start.set(self.months[0]); self.trend_end.set(self.months[-1])
            self.struct_start.set(self.months[0]); self.struct_end.set(self.months[-1])
            self.comp_start.set(self.months[0]); self.comp_end.set(self.months[-1])
            top.destroy()
            try:
                self.on_plot_trend(); self.on_plot_structure(); self.on_plot_comp()
            except Exception as e:
                messagebox.showwarning("Plot error", f"Plotting failed:\\n{e}")

        ttk.Button(btns, text="确定", command=apply_and_close).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=top.destroy).pack(side=tk.RIGHT, padx=6)

    # ---------- Tab 1: Sales Trend (with details + cascade) ----------
    def _build_trend(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))

        pr = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        pr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        center = ttk.Frame(pr); right = ttk.Frame(pr, width=360)
        pr.add(center, weight=5); pr.add(right, weight=2)

        def mk_box(label):
            frame = ttk.LabelFrame(left, text=label, padding=6)
            lb = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=False, height=6)
            sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            lb.grid(row=0, column=0, sticky="nsew"); sb.grid(row=0, column=1, sticky="ns")
            frame.columnconfigure(0, weight=1); frame.rowconfigure(0, weight=1)
            return frame, lb

        self.lb_trend: Dict[str, tk.Listbox] = {}
        self._trend_widgets: Dict[str, ttk.LabelFrame] = {}

        sections = [
            ("Country", "Country"),
            ("Data type", "Data type"),
            ("Vehicle type", "Vehicle type"),
            ("Segment", "Segment - Proton"),
            ("Segment - Luxury (IDN only)", "Segment - Luxury"),
            ("Data source", "Data source"),
            ("Make", "Make"),
            ("Model", "Model"),
            ("Body style", "Body style (Proton)"),
            ("Seats", "Seats"),
            ("Drivetrain type", "Drivetrain type"),
            ("Pure electric range (kms)", "Pure electric range (kms)"),
            ("Price band", "Price band"),
        ]

        for i,(lab,col) in enumerate(sections):
            f, lb = mk_box(lab); f.grid(row=i//2, column=i%2, sticky="nsew", padx=4, pady=4)
            self.lb_trend[col]=lb
            self._trend_widgets[col]=f
        for r in range((len(sections)+1)//2): left.rowconfigure(r, weight=1)
        for c in range(2): left.columnconfigure(c, weight=1)

        opts = ttk.LabelFrame(left, text="Options", padding=6)
        opts.grid(row=7, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        self.trend_agg = tk.StringVar(value="Make")
        ttk.Label(opts, text="Aggregate by:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.trend_agg, state="readonly",
                     values=["Total","Make","Segment - Proton","Segment - Luxury","Model"]).grid(row=0,column=1,sticky="ew",padx=6)
        self.trend_metric = tk.StringVar(value="Trend")
        ttk.Label(opts, text="Metric:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(opts, textvariable=self.trend_metric, state="readonly",
                     values=["Trend","YoY %","MoM %"]).grid(row=0,column=3,sticky="ew",padx=6)

        ttk.Label(opts, text="Start month:").grid(row=1, column=0, sticky="w")
        ttk.Label(opts, text="End month:").grid(row=1, column=2, sticky="w")
        self.trend_start = tk.StringVar(); self.trend_end = tk.StringVar()
        self.trend_start_cb = ttk.Combobox(opts, textvariable=self.trend_start, state="readonly")
        self.trend_end_cb   = ttk.Combobox(opts, textvariable=self.trend_end,   state="readonly")
        self.trend_start_cb.grid(row=1,column=1,sticky="ew",padx=6); self.trend_end_cb.grid(row=1,column=3,sticky="ew",padx=6)

        ttk.Label(opts, text="Time granularity:").grid(row=2, column=0, sticky="w")
        self.trend_freq = tk.StringVar(value="Monthly")
        ttk.Combobox(opts, textvariable=self.trend_freq, state="readonly",
                     values=["Monthly","Quarterly","Yearly"]).grid(row=2, column=1, sticky="ew", padx=6)

        self.country_vs = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Country VS (compare across countries)", variable=self.country_vs).grid(row=2, column=2, columnspan=2, sticky="w")

        for i in range(4): opts.columnconfigure(i, weight=1)

        b = ttk.Frame(left); b.grid(row=8, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        ttk.Button(b, text="Plot", command=self.on_plot_trend).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Reset", command=self.on_reset_trend).pack(side=tk.LEFT, padx=4)

        self.fig_trend = Figure(figsize=(8,5), dpi=100)
        self.ax_trend = self.fig_trend.add_subplot(111)
        self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, master=center)
        self.canvas_trend.draw(); self.canvas_trend.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        kpi = ttk.Frame(center); kpi.pack(fill=tk.X, pady=(8,0))
        self.kpi_trend = tk.StringVar(value="KPIs —"); ttk.Label(kpi, textvariable=self.kpi_trend, anchor="w").pack(fill=tk.X)

        self.trend_detail = build_detail_panel(right)
        self._trend_ctx = None

        def on_country_select(event=None):
            lb = self.lb_trend.get("Country")
            if not lb or str(lb['state']) != 'normal': return
            sel = [lb.get(i) for i in lb.curselection()]
            enable = ("(All)" in sel) or any(str(x).lower()=="indonesia" for x in sel)
            lux_col = "Segment - Luxury"
            frame = self._trend_widgets.get(lux_col)
            lux_lb = self.lb_trend.get(lux_col)
            if frame and lux_lb:
                state = "normal" if enable and (self.df is not None and (lux_col in self.df.columns)) else "disabled"
                lux_lb.configure(state=state)
                try:
                    frame.configure(text="Segment - Luxury (IDN only)" + ("" if state=="normal" else " — disabled"))
                except Exception:
                    pass
        self._on_country_select = on_country_select

        # ---- cascade bindings
        for col in self.CASCADE_ORDER:
            lb = self.lb_trend.get(col)
            if lb is not None:
                lb.bind("<<ListboxSelect>>", self._on_cascade_change)

    # ----- cascade logic
    def _cascade_selected_dict(self) -> Dict[str, Optional[List[str]]]:
        sel = {}
        for col in self.CASCADE_ORDER:
            lb = self.lb_trend.get(col)
            sel[col] = self._get_selected(lb)
        return sel

    def _filtered_by_upstream(self, upto_col: str) -> pd.DataFrame:
        """Filter df by upstream selections up to and including upto_col."""
        if self.df is None:
            return pd.DataFrame()
        df = self.df.copy()
        for col in self.CASCADE_ORDER:
            vals = self._get_selected(self.lb_trend.get(col))
            if vals is not None:
                df = df[df[col].astype(str).isin(vals)]
            if col == upto_col:
                break
        return df

    def _refresh_downstream(self, changed_col: str):
        """When `changed_col` selection changes, refresh all downstream listboxes values."""
        if self.df is None:
            return
        if changed_col not in self.CASCADE_ORDER:
            return
        idx = self.CASCADE_ORDER.index(changed_col)
        # progressively filter and refresh downstream
        df = self._filtered_by_upstream(changed_col)
        for col in self.CASCADE_ORDER[idx+1:]:
            lb = self.lb_trend.get(col)
            frame = self._trend_widgets.get(col)
            if lb is None:
                continue
            if col not in self.df.columns:
                lb.delete(0, tk.END); lb.insert(tk.END, "(N/A)"); lb.configure(state="disabled")
                if frame:
                    try: frame.configure(text=frame.cget("text") + " — not found")
                    except Exception: pass
                continue
            # compute options valid under upstream filter
            options = sorted(df[col].dropna().astype(str).unique().tolist(), key=lambda x: str(x))
            prev = self._get_selected(lb)
            lb.configure(state="normal")
            lb.delete(0, tk.END); lb.insert(tk.END, "(All)")
            for v in options: lb.insert(tk.END, v)
            # restore previous selections if still valid; otherwise default to (All)
            if prev:
                valid = set(options)
                indices = [i+1 for i,v in enumerate(options) if v in prev and v in valid]
                if indices:
                    for i in indices: lb.selection_set(i)
                else:
                    lb.selection_set(0)
            else:
                lb.selection_set(0)
            # next iteration uses df filtered by this col too (if specific selected)
            sel_vals = self._get_selected(lb)
            if sel_vals is not None:
                df = df[df[col].astype(str).isin(sel_vals)]

    def _on_cascade_change(self, event=None):
        # find which listbox triggered
        for col, lb in self.lb_trend.items():
            if event.widget is lb:
                self._refresh_downstream(col)
                break

    # ---------- plotting for Sales Trend (same as v2.8) ----------
    def on_plot_trend(self):
        if self.df is None or not self.months: return
        months = self._slice_months(self.trend_start.get() or self.months[0], self.trend_end.get() or self.months[-1])

        # filters; if Country VS, skip Country filter to compare across countries
        skip_cols = ["Country"] if self.country_vs.get() else []
        fdf = self._apply_filters(self.df.copy(), self.lb_trend, skip_cols=skip_cols)

        # aggregation target
        agg_col = "Country" if self.country_vs.get() and ("Country" in fdf.columns) else (None if self.trend_agg.get()=="Total" else self.trend_agg.get())

        # base aggregation on monthly columns then periodize
        base = self._aggregate(fdf, agg_col, months)

        # time aggregation
        per_vals, per_labels = aggregate_time(base.iloc[:, 1:], months, self.trend_freq.get())
        base_period = pd.concat([base[["Series"]], per_vals], axis=1)

        # metric transform on periodized series
        ser = metric_transform_period(base_period, self.trend_metric.get(), self.trend_freq.get())

        # plot
        self.ax_trend.clear()
        x = np.arange(len(per_labels))
        lines=[]
        for _, row in ser.iterrows():
            y = row[per_labels].astype(float).values
            ln, = self.ax_trend.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=str(row["Series"]))
            lines.append(ln)
        ylabel = "Units" if self.trend_metric.get()=="Trend" else "%"
        title_agg = "Country" if (agg_col=="Country") else self.trend_agg.get()
        self.ax_trend.set_title(f"Sales {self.trend_metric.get()} — {title_agg} ({self.trend_freq.get()})")
        self.ax_trend.set_xlabel("Period"); self.ax_trend.set_ylabel(ylabel)
        step = self._safe_xtick_step(len(x))
        self.ax_trend.set_xticks(x[::step])
        self.ax_trend.set_xticklabels([per_labels[i] for i in range(0,len(per_labels),step)], rotation=45, ha="right")
        self.ax_trend.grid(True, linestyle="--", alpha=0.4); self.ax_trend.legend(fontsize=8)
        self.fig_trend.tight_layout(); self.canvas_trend.draw()
        # hover → details
        def on_hover(series_label, idx):
            row = ser[ser["Series"].astype(str)==series_label]
            if row.empty or idx>=len(per_labels): return
            val = float(row[per_labels[idx]].iloc[0]) if not pd.isna(row[per_labels[idx]].iloc[0]) else np.nan
            info = {"Series": series_label, "Period": per_labels[idx], "Value": val}
            push_detail(self.trend_detail, info)
        self._attach_hover(self.ax_trend, lines, per_labels, ylabel, cb=on_hover)

        # enable click
        for ln in lines:
            try: ln.set_picker(True); ln.set_pickradius(8)
            except Exception: pass

        # keep context for click drill
        self._trend_ctx = {"ser": ser, "per_labels": per_labels, "ylabel": ylabel,
                           "agg_col": agg_col, "base_df": fdf, "gran": self.trend_freq.get(),
                           "metric": self.trend_metric.get()}
        if not hasattr(self, "_mpl_pick_id_trend"):
            self._mpl_pick_id_trend = self.canvas_trend.mpl_connect('pick_event', self._on_pick_trend)

        # KPIs
        s = ser.iloc[0,1:].astype(float) if not ser.empty else pd.Series(dtype=float)
        total = float(np.nansum(s.values)) if len(s)>0 else 0.0
        latest = next((float(v) for v in reversed(s.values) if not (pd.isna(v))), np.nan) if len(s)>0 else np.nan
        self.kpi_trend.set(f"KPIs — Total in range: {total:,.0f} | Latest period: {latest:,.0f}")
        self._last_series = ser

    def _on_pick_trend(self, event):
        if self._trend_ctx is None: return
        label = event.artist.get_label()
        if not hasattr(event, "ind") or len(event.ind)==0: return
        idx = int(event.ind[0])
        ser = self._trend_ctx["ser"]; per_labels = self._trend_ctx["per_labels"]
        ylabel = self._trend_ctx["ylabel"]; base_df = self._trend_ctx["base_df"]
        agg_col = self._trend_ctx["agg_col"]; gran = self._trend_ctx["gran"]
        if idx<0 or idx>=len(per_labels): return
        period = per_labels[idx]

        row = ser[ser["Series"].astype(str)==label]
        if row.empty: return
        val = float(row[period].iloc[0]) if not pd.isna(row[period].iloc[0]) else np.nan

        if gran == "Monthly":
            yoy_lag, prev_lag = 12, 1
        elif gran == "Quarterly":
            yoy_lag, prev_lag = 4, 1
        else:
            yoy_lag, prev_lag = 1, 1
        mom = np.nan; yoy = np.nan
        if idx-prev_lag >= 0 and not pd.isna(row[per_labels[idx-prev_lag]].iloc[0]):
            prev = float(row[per_labels[idx-prev_lag]].iloc[0])
            if prev != 0: mom = (val/prev - 1.0) * 100.0
        if idx-yoy_lag >= 0 and not pd.isna(row[per_labels[idx-yoy_lag]].iloc[0]):
            prevy = float(row[per_labels[idx-yoy_lag]].iloc[0])
            if prevy != 0: yoy = (val/prevy - 1.0) * 100.0

        if ylabel == "Units":
            base = self._aggregate(base_df, agg_col, self.months)
            per_vals, labs = aggregate_time(base.iloc[:,1:], self.months, gran)
            base_period = pd.concat([base[["Series"]], per_vals], axis=1)
            r = base_period[base_period["Series"].astype(str)==label]
            if not r.empty and period in r.columns:
                idx2 = labs.index(period)
                r3 = float(r[labs[max(0,idx2-2):idx2+1]].sum(axis=1).iloc[0])
                r12 = float(r[labs[max(0,idx2-11):idx2+1]].sum(axis=1).iloc[0])
            else:
                r3 = r12 = np.nan
            info = {"Series": label, "Period": period, "Value": val, "MoM %": mom, "YoY %": yoy, "3P Sum": r3, "12P Sum": r12}
        else:
            s = row[per_labels].astype(float).values
            r3 = float(np.nanmean(s[max(0,idx-2):idx+1])) if len(s)>0 else np.nan
            r12 = float(np.nanmean(s[max(0,idx-11):idx+1])) if len(s)>0 else np.nan
            info = {"Series": label, "Period": period, "Value": val, "MoM %": mom, "YoY %": yoy, "3P Avg %": r3, "12P Avg %": r12}

        child_col = None
        if agg_col == "Country":
            child_col = "Make" if "Make" in base_df.columns else None
        elif agg_col == "Make":
            child_col = "Model" if "Model" in base_df.columns else None
        elif agg_col in ("Segment - Proton","Segment - Luxury"):
            child_col = "Make" if "Make" in base_df.columns else None
        elif agg_col is None:
            child_col = "Make" if "Make" in base_df.columns else None

        if child_col is not None:
            cols = []
            if gran == "Monthly":
                cols = [period]
            else:
                for m in self.months:
                    lab = _period_key(m, granularity=gran)
                    if lab and lab[0] == period:
                        cols.append(m)
            if not cols:
                breakdown = None
            else:
                dfsub = base_df.copy()
                if agg_col is not None:
                    dfsub = dfsub[dfsub[agg_col].astype(str)==label]
                br = dfsub.groupby(child_col)[cols].sum()
                br["val"] = br.sum(axis=1)
                br = br[["val"]].reset_index().sort_values(by="val", ascending=False).head(12)
                breakdown = br[[child_col, "val"]]
        else:
            breakdown = None

        push_detail(self.trend_detail, info, breakdown)

    def on_reset_trend(self):
        if self.df is None: return
        for col, lb in self.lb_trend.items():
            if str(lb['state']) == 'normal':
                lb.selection_clear(0, tk.END); lb.selection_set(0)
        if self.months:
            self.trend_start.set(self.months[0]); self.trend_end.set(self.months[-1])
        self.trend_agg.set("Make"); self.trend_metric.set("Trend")
        self.trend_freq.set("Monthly"); self.country_vs.set(False)
        if hasattr(self, "_on_country_select"):
            self._on_country_select()
        # refresh cascade from top
        self._refresh_downstream(self.CASCADE_ORDER[0])
        self.on_plot_trend()

    # ---------- Tab 2: Market Structure (unchanged from v2.8, still with details) ----------
    def _build_structure(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))

        pr = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        pr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        center = ttk.Frame(pr); right = ttk.Frame(pr, width=360)
        pr.add(center, weight=5); pr.add(right, weight=2)

        self.struct_dim = tk.StringVar(value="Drivetrain type")
        ttk.Label(left, text="Share dimension:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.struct_dim, state="readonly",
                     values=["Drivetrain type","Body style (Proton)","Price band"]).pack(fill=tk.X)

        self.struct_agg = tk.StringVar(value="Total")
        ttk.Label(left, text="Aggregate by (optional drilldown):").pack(anchor="w", pady=(8,0))
        cb = ttk.Combobox(left, textvariable=self.struct_agg, state="readonly",
                     values=["Total","Country","Make","Segment - Proton","Vehicle type"])
        cb.pack(fill=tk.X)

        ttk.Label(left, text="Filter value:").pack(anchor="w", pady=(8,0))
        self.struct_filter_val = tk.StringVar(value="(All)")
        self.struct_filter_cb = ttk.Combobox(left, textvariable=self.struct_filter_val, state="readonly")
        self.struct_filter_cb.pack(fill=tk.X)

        ttk.Label(left, text="Start month:").pack(anchor="w", pady=(8,0))
        self.struct_start = tk.StringVar()
        ttk.Label(left, text="End month:").pack(anchor="w")
        self.struct_end = tk.StringVar()
        self.struct_start_cb = ttk.Combobox(left, textvariable=self.struct_start, state="readonly"); self.struct_start_cb.pack(fill=tk.X)
        self.struct_end_cb   = ttk.Combobox(left, textvariable=self.struct_end,   state="readonly"); self.struct_end_cb.pack(fill=tk.X)

        ttk.Button(left, text="Plot Share Trend", command=self.on_plot_structure).pack(pady=10, fill=tk.X)

        self.fig_struct = Figure(figsize=(8,5), dpi=100)
        self.ax_struct = self.fig_struct.add_subplot(111)
        self.canvas_struct = FigureCanvasTkAgg(self.fig_struct, master=center)
        self.canvas_struct.draw(); self.canvas_struct.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.kpi_struct = tk.StringVar(value="Structure —"); ttk.Label(center, textvariable=self.kpi_struct).pack(anchor="w", pady=(8,0))

        self.struct_detail = build_detail_panel(right)

        def _refresh_struct_filter(*_):
            if self.df is None: 
                self.struct_filter_cb["values"] = ["(All)"]; self.struct_filter_val.set("(All)"); return
            g = self.struct_agg.get()
            if g == "Total" or g not in self.df.columns:
                self.struct_filter_cb["values"] = ["(All)"]; self.struct_filter_val.set("(All)")
            else:
                vals = sorted(self.df[g].astype(str).dropna().unique().tolist(), key=lambda x: str(x))
                self.struct_filter_cb["values"] = ["(All)"] + vals
                self.struct_filter_val.set("(All)")
        cb.bind("<<ComboboxSelected>>", _refresh_struct_filter)
        self._refresh_struct_filter = _refresh_struct_filter

        self._struct_ctx = None

    def on_plot_structure(self):
        if self.df is None or not self.months: return
        dim = self.struct_dim.get()
        if dim not in self.df.columns:
            messagebox.showwarning("Missing column", f"'{dim}' is not in this worksheet."); return
        months = self._slice_months(self.struct_start.get() or self.months[0], self.struct_end.get() or self.months[-1])

        gcol = None if self.struct_agg.get()=="Total" else self.struct_agg.get()
        df = self.df.copy()

        if gcol is not None and gcol in df.columns:
            sel = self.struct_filter_val.get()
            if sel and sel != "(All)":
                df = df[df[gcol].astype(str) == sel]

        if gcol is None:
            num = df.groupby(dim)[months].sum()
            denom = df[months].sum(axis=0).replace(0, np.nan)
            share = (num.div(denom, axis=1) * 100.0).reset_index().rename(columns={dim:"Series"})
        else:
            num = df.groupby([gcol, dim])[months].sum()
            denom = df.groupby(gcol)[months].sum().replace(0, np.nan)
            share = num.div(denom).mul(100.0)
            share = share.reset_index().rename(columns={dim:"Series", gcol:"_Group"})
            share["Series"] = share["_Group"].astype(str) + " — " + share["Series"].astype(str)
            share = share.drop(columns=["_Group"])

        self.ax_struct.clear()
        x = np.arange(len(months))
        lines=[]
        for _, row in share.iterrows():
            y = row[months].astype(float).values
            ln, = self.ax_struct.plot(x, y, marker="o", linewidth=1.6, markersize=3, label=str(row["Series"]))
            lines.append(ln)
        self.ax_struct.set_title(f"Market Share by {dim} — {self.struct_agg.get()} (Filter: {self.struct_filter_val.get()})")
        self.ax_struct.set_xlabel("Month"); self.ax_struct.set_ylabel("%")
        step = self._safe_xtick_step(len(x))
        self.ax_struct.set_xticks(x[::step])
        self.ax_struct.set_xticklabels([months[i] for i in range(0,len(months),step)], rotation=45, ha="right")
        self.ax_struct.grid(True, linestyle="--", alpha=0.4); self.ax_struct.legend(fontsize=8)
        self.fig_struct.tight_layout(); self.canvas_struct.draw()

        def on_hover(series_label, idx):
            row = share[share["Series"].astype(str)==series_label]
            if row.empty or idx>=len(months): return
            val = float(row[months[idx]].iloc[0]) if not pd.isna(row[months[idx]].iloc[0]) else np.nan
            push_detail(self.struct_detail, {"Series": series_label, "Month": months[idx], "Share %": val})
        self._attach_hover(self.ax_struct, lines, months, "%", cb=on_hover)

        for ln in lines:
            try: ln.set_picker(True); ln.set_pickradius(8)
            except Exception: pass

        self._struct_ctx = {"share": share, "months": months, "dim": dim, "gcol": gcol, "df": df}
        if not hasattr(self, "_mpl_pick_id_struct"):
            self._mpl_pick_id_struct = self.canvas_struct.mpl_connect('pick_event', self._on_pick_structure)

        latest = share.sort_values(by=months[-1], ascending=False)[["Series", months[-1]]].head(5) if len(months)>0 else pd.DataFrame(columns=["Series"])
        pieces = [f"{r.Series}: {r[months[-1]]:.1f}%" for _, r in latest.iterrows()] if len(latest)>0 else []
        self.kpi_struct.set("Latest composition — " + " | ".join(pieces) if pieces else "Latest composition — (no data)")
        self._last_series = share

    def _on_pick_structure(self, event):
        if self._struct_ctx is None: return
        label = event.artist.get_label()
        if not hasattr(event, "ind") or len(event.ind)==0: return
        idx = int(event.ind[0])
        share = self._struct_ctx["share"]; months = self._struct_ctx["months"]
        dim = self._struct_ctx["dim"]; gcol = self._struct_ctx["gcol"]; df = self._struct_ctx["df"]
        if idx<0 or idx>=len(months): return
        month = months[idx]

        row = share[share["Series"].astype(str)==label]
        if row.empty: return
        val = float(row[month].iloc[0]) if not pd.isna(row[month].iloc[0]) else np.nan

        grp_val = None; dim_val = label
        if gcol is not None and " — " in label:
            parts = label.split(" — ", 1); grp_val, dim_val = parts[0], parts[1]

        mom = np.nan; yoy = np.nan
        if idx-1 >= 0 and not pd.isna(row[months[idx-1]].iloc[0]):
            prev = float(row[months[idx-1]].iloc[0]); 
            if prev != 0: mom = (val/prev - 1.0) * 100.0
        if idx-12 >= 0 and not pd.isna(row[months[idx-12]].iloc[0]):
            prev12 = float(row[months[idx-12]].iloc[0]); 
            if prev12 != 0: yoy = (val/prev12 - 1.0) * 100.0

        svals = row[months].astype(float).values.flatten().tolist()
        r3 = float(np.nanmean(svals[max(0,idx-2):idx+1]))
        r12 = float(np.nanmean(svals[max(0,idx-11):idx+1]))

        info = {"Series": label, "Month": month, "Share %": val, "MoM %": mom, "YoY %": yoy, "3M Avg %": r3, "12M Avg %": r12}

        if gcol is None:
            peers = share[["Series", month]].sort_values(by=month, ascending=False).head(12)
            breakdown = peers.rename(columns={"Series":"Peer", month:"val"})[["Peer","val"]]
        else:
            df_peer = df.copy()
            df_peer = df_peer[df_peer[gcol].astype(str) == grp_val] if grp_val is not None else df_peer
            peer_num = df_peer.groupby(dim)[month].sum()
            peer_den = df_peer[month].sum()
            if peer_den == 0 or pd.isna(peer_den):
                breakdown = None
            else:
                peer_share = (peer_num / peer_den * 100.0).reset_index().sort_values(by=month, ascending=False).head(12)
                breakdown = peer_share[[dim, month]].rename(columns={dim:"Peer", month:"val"})

        push_detail(self.struct_detail, info, breakdown)

    # ---------- Tab 3: Competitor Compare (same as v2.8) ----------
    def _build_comp(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))

        pr = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        pr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        center = ttk.Frame(pr); right = ttk.Frame(pr, width=360)
        pr.add(center, weight=5); pr.add(right, weight=2)

        self.comp_level = tk.StringVar(value="Make")
        ttk.Label(left, text="Compare level:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.comp_level, state="readonly",
                     values=["Model","Make","Segment - Proton"]).pack(fill=tk.X)

        self.lb_comp = tk.Listbox(left, selectmode=tk.MULTIPLE, exportselection=False, height=22)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.lb_comp.yview)
        self.lb_comp.configure(yscrollcommand=sb.set)
        ttk.Label(left, text="Pick series to compare:").pack(anchor="w", pady=(8,0))
        self.lb_comp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Start month:").pack(anchor="w", pady=(8,0))
        self.comp_start = tk.StringVar()
        ttk.Label(left, text="End month:").pack(anchor="w")
        self.comp_end = tk.StringVar()
        self.comp_start_cb = ttk.Combobox(left, textvariable=self.comp_start, state="readonly"); self.comp_start_cb.pack(fill=tk.X)
        self.comp_end_cb   = ttk.Combobox(left, textvariable=self.comp_end,   state="readonly"); self.comp_end_cb.pack(fill=tk.X)

        ttk.Button(left, text="Plot Compare", command=self.on_plot_comp).pack(pady=10, fill=tk.X)

        self.fig_comp = Figure(figsize=(8,5), dpi=100)
        self.ax_comp = self.fig_comp.add_subplot(111)
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=center)
        self.canvas_comp.draw(); self.canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.kpi_comp = tk.StringVar(value="Latest month top —"); ttk.Label(center, textvariable=self.kpi_comp).pack(anchor="w", pady=(8,0))

        self.comp_detail = build_detail_panel(right)
        self._comp_ctx = None

        def on_level_change(*_):
            self._refresh_comp_series()
        self.comp_level.trace_add("write", on_level_change)

    def _refresh_comp_series(self):
        if self.df is None: return
        level = self.comp_level.get()
        if level not in self.df.columns:
            self.lb_comp.delete(0, tk.END); self.lb_comp.insert(tk.END, "(N/A)"); self.lb_comp.configure(state="disabled"); return
        self.lb_comp.configure(state="normal")
        self.lb_comp.delete(0, tk.END)
        items = sorted(self.df[level].astype(str).dropna().unique().tolist(), key=lambda x: str(x))
        for v in items: self.lb_comp.insert(tk.END, v)

    def on_plot_comp(self):
        if self.df is None or not self.months: return
        level = self.comp_level.get()
        if level not in self.df.columns:
            messagebox.showwarning("Missing column", f"'{level}' not found."); return
        picks = [self.lb_comp.get(i) for i in self.lb_comp.curselection()]
        if not picks:
            messagebox.showinfo("Pick items", "Select 1~8 items to compare."); return
        months = self._slice_months(self.comp_start.get() or self.months[0], self.comp_end.get() or self.months[-1])

        sub = self.df[self.df[level].astype(str).isin(picks)]
        agg = sub.groupby(level)[months].sum().reset_index().rename(columns={level:"Series"})

        self.ax_comp.clear()
        x = np.arange(len(months))
        lines=[]
        for _, row in agg.iterrows():
            y = row[months].astype(float).values
            ln, = self.ax_comp.plot(x, y, marker="o", linewidth=1.6, markersize=3, label=str(row["Series"]))
            lines.append(ln)
        self.ax_comp.set_title(f"Competitor Compare — {level}")
        self.ax_comp.set_xlabel("Month"); self.ax_comp.set_ylabel("Units")
        step = self._safe_xtick_step(len(x))
        self.ax_comp.set_xticks(x[::step])
        self.ax_comp.set_xticklabels([months[i] for i in range(0,len(months),step)], rotation=45, ha="right")
        self.ax_comp.grid(True, linestyle="--", alpha=0.4); self.ax_comp.legend(fontsize=8)
        self.fig_comp.tight_layout(); self.canvas_comp.draw()

        last_rank = agg[["Series", months[-1]]].sort_values(by=months[-1], ascending=False).head(5) if len(months)>0 else pd.DataFrame(columns=["Series"])
        parts = [f"{r.Series}: {r[months[-1]]:,.0f}" for _, r in last_rank.iterrows()] if len(last_rank)>0 else []
        self.kpi_comp.set("Latest month top — " + " | ".join(parts) if parts else "Latest month top — (no data)")
        self._last_series = agg

        def on_hover(series_label, idx):
            row = agg[agg["Series"].astype(str)==series_label]
            if row.empty or idx>=len(months): return
            info = {"Series": series_label, "Month": months[idx], "Value": float(row[months[idx]].iloc[0])}
            push_detail(self.comp_detail, info)
        self._attach_hover(self.ax_comp, lines, months, "Units", cb=on_hover)

        for ln in lines:
            try: ln.set_picker(True); ln.set_pickradius(8)
            except Exception: pass

        self._comp_ctx = {"agg": agg, "months": months, "level": level, "sub": sub}
        if not hasattr(self, "_mpl_pick_id_comp"):
            self._mpl_pick_id_comp = self.canvas_comp.mpl_connect('pick_event', self._on_pick_comp)

    def _on_pick_comp(self, event):
        if self._comp_ctx is None: return
        artist = event.artist
        label = artist.get_label()
        if not hasattr(event, "ind") or len(event.ind)==0: return
        idx = int(event.ind[0])
        months = self._comp_ctx["months"]
        agg = self._comp_ctx["agg"]
        sub = self._comp_ctx["sub"]
        level = self._comp_ctx["level"]
        if idx < 0 or idx >= len(months): return
        month = months[idx]

        row = agg[agg["Series"].astype(str)==label]
        if row.empty: return
        val = float(row[month].iloc[0])

        mom = np.nan; yoy = np.nan
        if idx-1 >= 0:
            prev = float(row[months[idx-1]].iloc[0])
            if prev != 0: mom = (val/prev - 1.0) * 100.0
        if idx-12 >= 0:
            prev12 = float(row[months[idx-12]].iloc[0])
            if prev12 != 0: yoy = (val/prev12 - 1.0) * 100.0

        info = {"Series": label, "Month": month, "Value": val, "MoM %": mom, "YoY %": yoy,
                "3M Sum": float(row[months[max(0,idx-2):idx+1]].sum(axis=1).iloc[0]),
                "12M Sum": float(row[months[max(0,idx-11):idx+1]].sum(axis=1).iloc[0])}

        child_col = None
        if level == "Make" and "Model" in sub.columns:
            child_col = "Model"
        elif level == "Segment - Proton" and "Make" in sub.columns:
            child_col = "Make"
        if child_col is not None and month in sub.columns:
            br = sub[sub[level].astype(str)==label].groupby(child_col)[month].sum().reset_index().sort_values(by=month, ascending=False).head(12)
            breakdown = br[[child_col, month]]
        else:
            breakdown = None

        push_detail(self.comp_detail, info, breakdown)

    # ---------- File Ops ----------
    def on_open(self):
        path = filedialog.askopenfilename(title="Open Excel", filetypes=[("Excel files","*.xlsx *.xlsm *.xls")])
        if not path: return
        self.load_file(path)

    def load_file(self, path: str):
        try:
            xls = pd.ExcelFile(path)
            if "Sales Data" not in xls.sheet_names:
                raise ValueError("The Excel file must contain a sheet named 'Sales Data'.")
            df = pd.read_excel(path, sheet_name="Sales Data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file:\\n{e}"); return

        self.path_var.set(path)
        self.df = df
        auto_months = detect_month_columns(df)
        if auto_months:
            self.months = auto_months
        else:
            self.months = []
            messagebox.showwarning("No monthly columns", "未自动识别到月份列，请手动选择。Tools → Select month columns…")
            self.after(100, self.on_pick_months)

        # populate filters
        for col_attr in ["Country","Data type","Vehicle type","Segment - Proton","Segment - Luxury","Data source","Make","Model","Body style (Proton)","Seats","Drivetrain type","Pure electric range (kms)","Price band"]:
            lb = self.lb_trend.get(col_attr)
            frame = self._trend_widgets.get(col_attr)
            if lb is None: 
                continue
            if col_attr in df.columns:
                lb.configure(state="normal")
                vals = df[col_attr].astype(str).unique().tolist()
                lb.delete(0, tk.END)
                lb.insert(tk.END, "(All)")
                for v in sorted([x for x in vals if str(x).strip() != "" and x != "nan"], key=lambda x: str(x)):
                    lb.insert(tk.END, v)
                lb.selection_set(0)
            else:
                lb.delete(0, tk.END); lb.insert(tk.END, "(N/A)"); lb.configure(state="disabled")
                if frame:
                    try: frame.configure(text=frame.cget("text") + " — not found")
                    except Exception: pass

        if "Country" in self.lb_trend:
            self.lb_trend["Country"].bind("<<ListboxSelect>>", lambda e: self._on_country_select())

        if self.months:
            self.trend_start_cb["values"]=self.months; self.trend_end_cb["values"]=self.months
            self.struct_start_cb["values"]=self.months; self.struct_end_cb["values"]=self.months
            self.comp_start_cb["values"]=self.months; self.comp_end_cb["values"]=self.months
            self.trend_start.set(self.months[0]); self.trend_end.set(self.months[-1])
            self.struct_start.set(self.months[0]); self.struct_end.set(self.months[-1])
            self.comp_start.set(self.months[0]); self.comp_end.set(self.months[-1])

        self._refresh_comp_series()
        if hasattr(self, "_refresh_struct_filter"):
            self._refresh_struct_filter()

        if hasattr(self, "_on_country_select"):
            self._on_country_select()

        # initial cascade refresh so downstream options respect upstream "(All)"
        if self.CASCADE_ORDER and self.CASCADE_ORDER[0] in self.lb_trend:
            self._refresh_downstream(self.CASCADE_ORDER[0])

        try:
            if self.months:
                self.on_plot_trend(); self.on_plot_structure(); self.on_plot_comp()
        except Exception as e:
            messagebox.showwarning("Plot error", f"Plotting failed:\\n{e}")

    # ---------- Exports/Tools/About ----------
    def on_export_png(self):
        cur = self.tabs.index(self.tabs.select())
        fig = self.fig_trend if cur==0 else (self.fig_struct if cur==1 else self.fig_comp)
        path = filedialog.asksaveasfilename(title="Save chart as PNG", defaultextension=".png", filetypes=[("PNG","*.png")])
        if not path: return
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Chart saved to:\\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PNG:\\n{e}")

    def on_export_csv(self):
        if self._last_series is None:
            messagebox.showinfo("Info", "Please plot first."); return
        path = filedialog.asksaveasfilename(title="Save series as CSV", defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            self._last_series.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Series saved to:\\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\\n{e}")

    def on_show_headers(self):
        if self.df is None:
            messagebox.showinfo("Info", "Open a file first."); return
        cols = [str(c) for c in self.df.columns]
        preview = "\\n".join(cols[:200])
        messagebox.showinfo("Column headers (first 200)", preview)

    def on_about(self):
        tips = [
            "Market Trend Analyzer — v2.9",
            "• Sales Trend: cascading selectors + Country VS + details panel",
            "• Market Structure: secondary filter + details panel",
            "• Competitor Compare: details panel with drilldown",
            "• Hover tooltips require: pip install mplcursors",
        ]
        messagebox.showinfo("About", "\\n".join(tips))

if __name__ == "__main__":
    App().mainloop()
