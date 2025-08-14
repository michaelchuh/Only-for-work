
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Trend Analyzer — v2 + Hover
• Base: MarketTrendGUI_v2 (Sales Data only, robust month detection, 3 tabs)
• New: Mouse hover tooltips on all charts (uses mplcursors)
Run:
  pip install -U pandas matplotlib openpyxl mplcursors
  python MarketTrendGUI_v2_hover.py your_file.xlsx
"""
import os, re, sys
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import mplcursors
except Exception:
    mplcursors = None

# ---------- Month detection (robust) ----------
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

    m = re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:uary|ch|il|e|y|ust|tember|ober|ember)?[ \-_/\.]*'?(\d{2}|\d{4})$", s0)
    if m:
        mon = MONTH_NAMES[m.group(1)]
        yr  = int(m.group(2))
        if yr < 100: yr += 2000 if yr <= 79 else 1900
        return (yr, mon)

    m = re.match(r"^(\d{4})[ \-_/\.]*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:uary|ch|il|e|y|ust|tember|ober|ember)?$", s0)
    if m:
        yr = int(m.group(1)); mon = MONTH_NAMES[m.group(2)]
        return (yr, mon)

    m = re.match(r"^(\d{4})[ \-_/\.](\d{1,2})$", s0)
    if m:
        yr = int(m.group(1)); mon = int(m.group(2))
        if 1 <= mon <= 12: return (yr, mon)

    m = re.match(r"^(\d{1,2})[ \-_/\.](\d{4})$", s0)
    if m:
        mon = int(m.group(1)); yr = int(m.group(2))
        if 1 <= mon <= 12: return (yr, mon)

    m = re.match(r"^(\d{4})\s*(\d{2})$", s0)
    if m:
        yr = int(m.group(1)); mon = int(m.group(2))
        if 1 <= mon <= 12: return (yr, mon)

    m = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月$", s0)
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
    months = []
    for c in df.columns:
        if _parse_month_from_header(str(c)) is not None:
            months.append(c)
    return months

# -------------- GUI App --------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Market Trend Analyzer — v2 + Hover")
        self.geometry("1400x840")
        self.minsize(1200, 760)

        self.df: Optional[pd.DataFrame] = None
        self.months: List[str] = []
        self._last_series = None

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

        # Tab 1 — Sales Trend
        self.tab_trend = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_trend, text="Sales Trend")
        self._build_trend(self.tab_trend)

        # Tab 2 — Market Structure
        self.tab_struct = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_struct, text="Market Structure")
        self._build_structure(self.tab_struct)

        # Tab 3 — Competitor Compare
        self.tab_comp = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_comp, text="Competitor Compare")
        self._build_comp(self.tab_comp)

    # ---------- Helpers ----------
    def _slice_months(self, start: str, end: str) -> List[str]:
        if not self.months: return []
        s = self.months.index(start) if start in self.months else 0
        e = self.months.index(end) if end in self.months else len(self.months)-1
        if s>e: s,e = e,s
        return self.months[s:e+1]

    def _apply_filters(self, df: pd.DataFrame, filter_map: Dict[str, tk.Listbox]) -> pd.DataFrame:
        for col, lb in filter_map.items():
            if col not in df.columns or str(lb['state']) != 'normal': continue
            vals = [lb.get(i) for i in lb.curselection()]
            if not vals or "(All)" in vals: continue
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

    def _metric_transform(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        months = df.columns[1:]
        vals = df[months].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        out = vals.copy()
        if metric == "YoY %":
            for i in range(len(months)):
                if i-12>=0:
                    prev = vals.iloc[:,i-12]
                    out.iloc[:,i] = np.where(prev==0, np.nan, (vals.iloc[:,i]/prev - 1.0) * 100.0)
                else:
                    out.iloc[:,i] = np.nan
        elif metric == "MoM %":
            for i in range(len(months)):
                if i-1>=0:
                    prev = vals.iloc[:,i-1]
                    out.iloc[:,i] = np.where(prev==0, np.nan, (vals.iloc[:,i]/prev - 1.0) * 100.0)
                else:
                    out.iloc[:,i] = np.nan
        return pd.concat([df[["Series"]], out], axis=1)

    def _attach_hover(self, ax, lines, months, unit_label):
        if mplcursors is None:
            return
        try:
            cursor = mplcursors.cursor(lines, hover=True)
            @cursor.connect("add")
            def _(sel):
                x, y = sel.target
                idx = int(round(x))
                if idx < 0: idx = 0
                if idx >= len(months): idx = len(months)-1
                line = sel.artist
                label = line.get_label()
                month = str(months[idx])
                if unit_label == "%":
                    val_txt = f"{y:.2f}%"
                else:
                    val_txt = f"{y:,.0f}"
                sel.annotation.set(text=f"{label}\n{month}\n{val_txt}")
        except Exception:
            pass

    # ---------- Tab 1 ----------
    def _build_trend(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))
        right = ttk.Frame(root); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def mk_box(label):
            frame = ttk.LabelFrame(left, text=label, padding=6)
            lb = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=False, height=6)
            sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            lb.grid(row=0, column=0, sticky="nsew"); sb.grid(row=0, column=1, sticky="ns")
            frame.columnconfigure(0, weight=1); frame.rowconfigure(0, weight=1)
            return frame, lb

        self.lb_trend: Dict[str, tk.Listbox] = {}
        sections = [
            ("Country", "Country"),
            ("Make", "Make"),
            ("Model", "Model"),
            ("Vehicle type", "Vehicle type"),
            ("Segment - Proton", "Segment - Proton"),
            ("Segment - Luxury", "Segment - Luxury"),
            ("Data type", "Data type"),
            ("Drivetrain type", "Drivetrain type"),
            ("Price band", "Price band"),
        ]
        for i,(lab,col) in enumerate(sections):
            f, lb = mk_box(lab); f.grid(row=i//2, column=i%2, sticky="nsew", padx=4, pady=4)
            self.lb_trend[col]=lb
        for r in range((len(sections)+1)//2): left.rowconfigure(r, weight=1)
        for c in range(2): left.columnconfigure(c, weight=1)

        opts = ttk.LabelFrame(left, text="Options", padding=6)
        opts.grid(row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
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
        for i in range(4): opts.columnconfigure(i, weight=1)

        b = ttk.Frame(left); b.grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        ttk.Button(b, text="Plot", command=self.on_plot_trend).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="Reset", command=self.on_reset_trend).pack(side=tk.LEFT, padx=4)

        self.fig_trend = Figure(figsize=(8,5), dpi=100)
        self.ax_trend = self.fig_trend.add_subplot(111)
        self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, master=right)
        self.canvas_trend.draw(); self.canvas_trend.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        kpi = ttk.Frame(right); kpi.pack(fill=tk.X, pady=(8,0))
        self.kpi_trend = tk.StringVar(value="KPIs —")
        ttk.Label(kpi, textvariable=self.kpi_trend, anchor="w").pack(fill=tk.X)

    def on_plot_trend(self):
        if self.df is None or not self.months: return
        months = self._slice_months(self.trend_start.get() or self.months[0], self.trend_end.get() or self.months[-1])
        fdf = self._apply_filters(self.df.copy(), self.lb_trend)
        group_col = None if self.trend_agg.get()=="Total" else self.trend_agg.get()
        agg = self._aggregate(fdf, group_col, months)
        ser = self._metric_transform(agg, self.trend_metric.get())

        self.ax_trend.clear()
        x = np.arange(len(months))
        lines=[]
        for _, row in ser.iterrows():
            y = row[months].astype(float).values
            ln, = self.ax_trend.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=str(row["Series"]))
            lines.append(ln)
        ylabel = "Units" if self.trend_metric.get()=="Trend" else "%"
        self.ax_trend.set_title(f"Sales {self.trend_metric.get()} — {self.trend_agg.get()}")
        self.ax_trend.set_xlabel("Month"); self.ax_trend.set_ylabel(ylabel)
        self.ax_trend.set_xticks(x[::max(1,len(x)//12)])
        self.ax_trend.set_xticklabels([months[i] for i in range(0,len(months),max(1,len(x)//12))], rotation=45, ha="right")
        self.ax_trend.grid(True, linestyle="--", alpha=0.4); self.ax_trend.legend(fontsize=8)
        self.fig_trend.tight_layout(); self.canvas_trend.draw()
        self._attach_hover(self.ax_trend, lines, months, ylabel)

        s = ser.iloc[0,1:].astype(float)
        total = float(np.nansum(s.values))
        latest = next((float(v) for v in reversed(s.values) if not np.isnan(v)), np.nan)
        self.kpi_trend.set(f"KPIs — Total in range: {total:,.0f} | Latest month: {latest:,.0f}")
        self._last_series = ser

    def on_reset_trend(self):
        if self.df is None: return
        for lb in self.lb_trend.values():
            if str(lb['state']) == 'normal':
                lb.selection_clear(0, tk.END); lb.selection_set(0)
        if self.months:
            self.trend_start.set(self.months[0]); self.trend_end.set(self.months[-1])
        self.trend_agg.set("Make"); self.trend_metric.set("Trend")
        self.on_plot_trend()

    # ---------- Tab 2 ----------
    def _build_structure(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))
        right = ttk.Frame(root); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.struct_dim = tk.StringVar(value="Drivetrain type")
        ttk.Label(left, text="Share dimension:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.struct_dim, state="readonly",
                     values=["Drivetrain type","Body style (Proton)","Price band"]).pack(fill=tk.X)

        self.struct_agg = tk.StringVar(value="Total")
        ttk.Label(left, text="Aggregate by (optional drilldown):").pack(anchor="w", pady=(8,0))
        ttk.Combobox(left, textvariable=self.struct_agg, state="readonly",
                     values=["Total","Country","Make","Segment - Proton","Vehicle type"]).pack(fill=tk.X)

        ttk.Label(left, text="Start month:").pack(anchor="w", pady=(8,0))
        self.struct_start = tk.StringVar()
        ttk.Label(left, text="End month:").pack(anchor="w")
        self.struct_end = tk.StringVar()
        self.struct_start_cb = ttk.Combobox(left, textvariable=self.struct_start, state="readonly"); self.struct_start_cb.pack(fill=tk.X)
        self.struct_end_cb   = ttk.Combobox(left, textvariable=self.struct_end,   state="readonly"); self.struct_end_cb.pack(fill=tk.X)

        ttk.Button(left, text="Plot Share Trend", command=self.on_plot_structure).pack(pady=10, fill=tk.X)

        self.fig_struct = Figure(figsize=(8,5), dpi=100)
        self.ax_struct = self.fig_struct.add_subplot(111)
        self.canvas_struct = FigureCanvasTkAgg(self.fig_struct, master=right)
        self.canvas_struct.draw(); self.canvas_struct.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.kpi_struct = tk.StringVar(value="Structure —"); ttk.Label(right, textvariable=self.kpi_struct).pack(anchor="w", pady=(8,0))

    def on_plot_structure(self):
        if self.df is None or not self.months: return
        dim = self.struct_dim.get()
        if dim not in self.df.columns:
            messagebox.showwarning("Missing column", f"'{dim}' is not in this worksheet."); return
        months = self._slice_months(self.struct_start.get() or self.months[0], self.struct_end.get() or self.months[-1])

        gcol = None if self.struct_agg.get()=="Total" else self.struct_agg.get()
        df = self.df.copy()

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
        self.ax_struct.set_title(f"Market Share by {dim} — {self.struct_agg.get()}")
        self.ax_struct.set_xlabel("Month"); self.ax_struct.set_ylabel("%")
        self.ax_struct.set_xticks(x[::max(1,len(x)//12)])
        self.ax_struct.set_xticklabels([months[i] for i in range(0,len(months),max(1,len(x)//12))], rotation=45, ha="right")
        self.ax_struct.grid(True, linestyle="--", alpha=0.4); self.ax_struct.legend(fontsize=8)
        self.fig_struct.tight_layout(); self.canvas_struct.draw()
        self._attach_hover(self.ax_struct, lines, months, "%")

        latest = share.sort_values(by=months[-1], ascending=False)[["Series", months[-1]]].head(5)
        pieces = [f"{r.Series}: {r[months[-1]]:.1f}%" for _, r in latest.iterrows()]
        self.kpi_struct.set("Latest composition — " + " | ".join(pieces))
        self._last_series = share

    # ---------- Tab 3 ----------
    def _build_comp(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))
        right = ttk.Frame(root); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.comp_level = tk.StringVar(value="Model")
        ttk.Label(left, text="Compare level:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.comp_level, state="readonly",
                     values=["Model","Make","Segment - Proton"]).pack(fill=tk.X)

        self.lb_comp = tk.Listbox(left, selectmode=tk.MULTIPLE, exportselection=False, height=14)
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
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=right)
        self.canvas_comp.draw(); self.canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.kpi_comp = tk.StringVar(value="Compare —"); ttk.Label(right, textvariable=self.kpi_comp).pack(anchor="w", pady=(8,0))

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
        self.ax_comp.set_xticks(x[::max(1,len(x)//12)])
        self.ax_comp.set_xticklabels([months[i] for i in range(0,len(months),max(1,len(x)//12))], rotation=45, ha="right")
        self.ax_comp.grid(True, linestyle="--", alpha=0.4); self.ax_comp.legend(fontsize=8)
        self.fig_comp.tight_layout(); self.canvas_comp.draw()
        self._attach_hover(self.ax_comp, lines, months, "Units")

        last_rank = agg[["Series", months[-1]]].sort_values(by=months[-1], ascending=False).head(5)
        parts = [f"{r.Series}: {r[months[-1]]:,.0f}" for _, r in last_rank.iterrows()]
        self.kpi_comp.set("Latest month top — " + " | ".join(parts))
        self._last_series = agg

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
            messagebox.showerror("Error", f"Failed to open file:\n{e}"); return

        self.path_var.set(path)
        self.df = df
        self.months = detect_month_columns(df)
        if not self.months:
            messagebox.showwarning("Warning", "No monthly columns identified. Try Tools → Show column headers to inspect.")
            return

        # populate trend filters
        for col, lb in self.lb_trend.items():
            if col in df.columns:
                lb.configure(state="normal")
                vals = df[col].astype(str).unique().tolist()
                lb.delete(0, tk.END)
                lb.insert(tk.END, "(All)")
                for v in sorted([x for x in vals if str(x).strip() != "" and x != "nan"], key=lambda x: str(x)):
                    lb.insert(tk.END, v)
                lb.selection_set(0)
            else:
                lb.delete(0, tk.END); lb.insert(tk.END, "(N/A)"); lb.configure(state="disabled")
        self.trend_start_cb["values"]=self.months; self.trend_end_cb["values"]=self.months
        self.trend_start.set(self.months[0]); self.trend_end.set(self.months[-1])

        # structure months pickers
        self.struct_start_cb["values"]=self.months; self.struct_end_cb["values"]=self.months
        self.struct_start.set(self.months[0]); self.struct_end.set(self.months[-1])

        # comparator months pickers
        self.comp_start_cb["values"]=self.months; self.comp_end_cb["values"]=self.months
        self.comp_start.set(self.months[0]); self.comp_end.set(self.months[-1])

        self._refresh_comp_series()

        # initial plots
        self.on_plot_trend()
        self.on_plot_structure()
        self.on_plot_comp()

    # ---------- Export ----------
    def on_export_png(self):
        cur = self.tabs.index(self.tabs.select())
        fig = self.fig_trend if cur==0 else (self.fig_struct if cur==1 else self.fig_comp)
        path = filedialog.asksaveasfilename(title="Save chart as PNG", defaultextension=".png", filetypes=[("PNG","*.png")])
        if not path: return
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Chart saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PNG:\n{e}")

    def on_export_csv(self):
        if self._last_series is None:
            messagebox.showinfo("Info", "Please plot first."); return
        path = filedialog.asksaveasfilename(title="Save series as CSV", defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            self._last_series.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Series saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

    # ---------- Tools ----------
    def on_show_headers(self):
        if self.df is None:
            messagebox.showinfo("Info", "Open a file first."); return
        cols = [str(c) for c in self.df.columns]
        preview = "\n".join(cols[:80])
        messagebox.showinfo("Column headers (first 80)", preview)

    def on_about(self):
        tips = [
            "Market Trend Analyzer — v2 + Hover",
            "• Sales Data only, robust month detection",
            "• Hover tooltips require: pip install mplcursors",
            "• Tabs: Sales Trend / Market Structure / Competitor Compare"
        ]
        messagebox.showinfo("About", "\n".join(tips))

if __name__ == "__main__":
    App().mainloop()
