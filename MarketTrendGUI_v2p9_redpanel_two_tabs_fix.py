
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Trend Analyzer — v2.9 + RedPanel (Two Tabs: Sales Trend, Market Structure)
- 完全移除 Competitor Compare 页签与相关代码。
- 保留并稳定红区（智能钻取）：维度/度量/周期/Top-N/表格或水平条形图/导出CSV。
- Structure 默认 Share %；其它默认 Units。
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
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct":10, "october":10, "nov":11, "november":11, "dec":12, "december":12,
}

def _normalize_colname(x: Union[str, object]) -> str:
    if isinstance(x, pd.Timestamp):
        return x.strftime("%b '%y")
    s = str(x).strip().replace("’","'").replace("‘","'")
    s = re.sub(r"\s+", " ", s)
    return s

def _is_total_or_ytd(s: str) -> bool:
    s_low = s.lower()
    return ("tot" in s_low) or ("ytd" in s_low) or ("year to date" in s_low)

def _parse_month_from_header(s: str) -> Optional[Tuple[int,int]]:
    if not s or _is_total_or_ytd(s): return None
    s0 = _normalize_colname(s).lower()
    m = re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:uary|ch|il|e|y|ust|tember|ober|ember)?[ \-_/\.]*'?([0-9]{2}|[0-9]{4})$", s0)
    if m:
        mon = MONTH_NAMES[m.group(1)]; yr = int(m.group(2))
        if yr < 100: yr += 2000 if yr <= 79 else 1900
        return (yr, mon)
    m = re.match(r"^([0-9]{4})[ \-_/\.]([01]?\d)$", s0)
    if m:
        yr = int(m.group(1)); mon = int(m.group(2))
        if 1<=mon<=12: return (yr, mon)
    m = re.match(r"^([01]?\d)[ \-_/\.]([12]\d{3})$", s0)
    if m:
        mon = int(m.group(1)); yr = int(m.group(2))
        if 1<=mon<=12: return (yr, mon)
    m = re.match(r"^([0-9]{4})\s*年\s*([0-9]{1,2})\s*月$", s0)
    if m:
        yr = int(m.group(1)); mon = int(m.group(2))
        if 1<=mon<=12: return (yr, mon)
    try:
        dt = pd.to_datetime(s0, errors="raise", dayfirst=False)
        return (int(dt.year), int(dt.month))
    except Exception:
        return None

def detect_month_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if _parse_month_from_header(str(c)) is not None]

# ---------- Period aggregation & metrics ----------
def _period_key(colname: str, granularity: str) -> Optional[Tuple[str, Tuple[int,int]]]:
    ym = _parse_month_from_header(colname)
    if ym is None: return None
    y, m = ym
    if granularity == "Monthly":
        return (f"{y}-{m:02d}", (y, m))
    if granularity == "Quarterly":
        q = (m-1)//3 + 1; return (f"{y} Q{q}", (y, q))
    return (f"{y}", (y,1))

def aggregate_time(df_vals: pd.DataFrame, months: List[str], granularity: str) -> Tuple[pd.DataFrame, List[str]]:
    if granularity == "Monthly":
        return df_vals[months].copy(), months
    groups: Dict[str, List[str]] = {}; order: Dict[str, Tuple[int,int]] = {}
    for c in months:
        out = _period_key(c, granularity)
        if out is None: continue
        lab, key = out; groups.setdefault(lab, []).append(c); order[lab] = key
    labels = sorted(groups.keys(), key=lambda k: order[k])
    agg = pd.DataFrame(index=df_vals.index)
    for lab in labels:
        cols = [c for c in groups[lab] if c in df_vals.columns]
        if cols: agg[lab] = df_vals[cols].sum(axis=1, numeric_only=True)
    return agg, labels

def metric_transform_period(df_ser: pd.DataFrame, metric: str, granularity: str) -> pd.DataFrame:
    periods = df_ser.columns[1:]
    vals = df_ser[periods].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out = vals.copy()
    if metric in ("YoY %", "MoM %"):
        if granularity == "Monthly":   yoy_lag, prev_lag = 12, 1
        elif granularity == "Quarterly": yoy_lag, prev_lag = 4, 1
        else: yoy_lag, prev_lag = 1, 1
        lag = yoy_lag if metric == "YoY %" else prev_lag
        out[:] = np.nan
        for i in range(len(periods)):
            if i-lag >= 0:
                prev = vals.iloc[:, i-lag]
                out.iloc[:, i] = np.where(prev==0, np.nan, (vals.iloc[:, i]/prev - 1.0) * 100.0)
    return pd.concat([df_ser[["Series"]], out], axis=1)

def safe_xtick_step(n: int) -> int:
    if n <= 0: return 1
    if n <= 12: return 1
    if n <= 24: return 2
    if n <= 48: return 3
    return max(4, n // 16)

# ---------- Red-panel (shared) ----------
DIM_PRIORITY = [
    "Country", "Make", "Model", "Price band", "Body style (Proton)",
    "Seats", "Data source", "Segment - Proton"
]

def _available_dims(df: pd.DataFrame, locks: Dict[str, object], exclude: List[str]) -> List[str]:
    sub = df.copy()
    for k,v in locks.items():
        if k in sub.columns:
            sub = sub[sub[k].astype(str)==str(v)]
    out = []
    for d in DIM_PRIORITY:
        if d in exclude: continue
        if d in sub.columns:
            vals = sub[d].dropna().astype(str).unique()
            if len(vals) > 1:
                out.append(d)
    return out

def build_detail_panel(parent) -> Dict[str, object]:
    card = ttk.LabelFrame(parent, text="Details", padding=8)
    card.pack(fill=tk.BOTH, expand=True)
    kv = ttk.Treeview(card, columns=("k","v"), show="headings", height=7)
    kv.heading("k", text="Field"); kv.heading("v", text="Value")
    kv.column("k", width=140, anchor="w"); kv.column("v", width=180, anchor="w")
    kv.pack(fill=tk.X, expand=False, pady=(0,8))

    br = ttk.Treeview(card, columns=("name","val"), show="headings", height=12)
    br.heading("name", text="Breakdown"); br.heading("val", text="Units / %")
    br.column("name", width=200, anchor="w"); br.column("val", width=100, anchor="e")
    br.pack(fill=tk.BOTH, expand=True)

    # ---- red interactive region
    red = tk.Frame(card, bg="#d7263d", height=230); red.pack(fill=tk.BOTH, expand=True, pady=(8,0))
    topbar = tk.Frame(red, bg="#d7263d"); topbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

    tk.Label(topbar, text="维度:", bg="#d7263d", fg="white").pack(side=tk.LEFT)
    dim_var = tk.StringVar(value=""); dim_cb = ttk.Combobox(topbar, textvariable=dim_var, state="readonly", width=22)
    dim_cb.pack(side=tk.LEFT, padx=6)

    tk.Label(topbar, text="度量:", bg="#d7263d", fg="white").pack(side=tk.LEFT, padx=(12,0))
    metric_var = tk.StringVar(value="Units")
    metric_cb = ttk.Combobox(topbar, textvariable=metric_var, state="readonly", values=["Units","Share %"], width=10)
    metric_cb.pack(side=tk.LEFT, padx=6)

    tk.Label(topbar, text="周期:", bg="#d7263d", fg="white").pack(side=tk.LEFT, padx=(12,0))
    period_var = tk.StringVar(value="(auto)"); period_cb = ttk.Combobox(topbar, textvariable=period_var, state="readonly", width=12)
    period_cb.pack(side=tk.LEFT, padx=6)

    tk.Label(topbar, text="Top-N:", bg="#d7263d", fg="white").pack(side=tk.LEFT, padx=(12,0))
    topn_var = tk.IntVar(value=10); topn_sp = tk.Spinbox(topbar, from_=1, to=50, textvariable=topn_var, width=5)
    topn_sp.pack(side=tk.LEFT, padx=6)

    view_var = tk.StringVar(value="排名表")
    view_cb = ttk.Combobox(topbar, textvariable=view_var, state="readonly", values=["排名表","水平条形图"], width=12)
    view_cb.pack(side=tk.LEFT, padx=(12,6))

    export_btn = ttk.Button(topbar, text="导出CSV"); export_btn.pack(side=tk.RIGHT, padx=6)

    middle = tk.Frame(red, bg="#d7263d"); middle.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))
    table = ttk.Treeview(middle, columns=("item","val"), show="headings")
    table.heading("item", text="项"); table.heading("val", text="值")
    table.column("item", width=200, anchor="w"); table.column("val", width=80, anchor="e")
    table.pack(fill=tk.BOTH, expand=True)

    fig = Figure(figsize=(3.6, 2.0), dpi=110); ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=middle); canvas.draw()

    status = tk.StringVar(value="Hover or click a point…"); ttk.Label(card, textvariable=status, anchor="w").pack(fill=tk.X, pady=(6,0))

    panel = {
        "kv": kv, "br": br, "status": status,
        "dim_var": dim_var, "dim_cb": dim_cb, "metric_var": metric_var, "metric_cb": metric_cb,
        "period_var": period_var, "period_cb": period_cb, "topn_var": topn_var, "view_var": view_var,
        "table": table, "fig": fig, "ax": ax, "canvas": canvas,
        "ctx": None, "child_dim": None, "child_value": None
    }

    def _clear_table(): table.delete(*table.get_children())
    def _show_table(): canvas.get_tk_widget().pack_forget(); table.pack(fill=tk.BOTH, expand=True)
    def _show_chart(): table.pack_forget(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _render(data: pd.DataFrame, label_col: str, val_col: str):
        topn = max(1, min(int(panel["topn_var"].get() or 10), 50))
        d = data[[label_col, val_col]].copy().sort_values(by=val_col, ascending=False).head(topn)
        _clear_table()
        for _, r in d.iterrows():
            v = r[val_col]
            if isinstance(v,(int,float,np.floating)): v = f"{float(v):,.2f}"
            table.insert("", "end", values=(str(r[label_col]), v))
        ax.clear(); y = np.arange(len(d))[::-1]
        ax.barh(y, d[val_col].values); ax.set_yticks(y); ax.set_yticklabels(d[label_col].astype(str).tolist()[::-1])
        ax.grid(True, axis="x", linestyle="--", alpha=0.4); ax.set_xlabel(panel["metric_var"].get())
        fig.tight_layout(); canvas.draw_idle()
        _show_table() if panel["view_var"].get()=="排名表" else _show_chart()

    def _compute_and_render(*_):
        ctx = panel["ctx"]
        if not ctx: return
        df = ctx["df"].copy()
        locks: Dict[str, object] = dict(ctx.get("locks", {}))
        if panel["child_dim"] and panel["child_value"] is not None:
            locks[panel["child_dim"]] = panel["child_value"]
        for k,v in locks.items():
            if k in df.columns: df = df[df[k].astype(str)==str(v)]
        cols = ctx.get("period_cols") or ([ctx.get("period")] if ctx.get("period") in df.columns else [])
        if not cols:
            months = ctx.get("months", []); cols = [months[-1]] if months else []
        if not cols: _clear_table(); return
        dim = panel["dim_var"].get()
        if not dim or dim not in df.columns: _clear_table(); return
        g = df.groupby(dim)[cols].sum(numeric_only=True)
        g["val_units"] = g.sum(axis=1)
        metric = panel["metric_var"].get()
        if metric == "Share %":
            denom = g["val_units"].sum(); g["val"] = 0.0 if denom in (0,np.nan) else g["val_units"]/denom*100.0
        else:
            g["val"] = g["val_units"]
        out = g.reset_index().rename(columns={dim:"item"}); _render(out, "item", "val")

    def _refresh_dim_options():
        ctx = panel["ctx"]; 
        if not ctx: return
        df = ctx["df"]; locks = dict(ctx.get("locks", {}))
        if panel["child_dim"] and panel["child_value"] is not None:
            locks[panel["child_dim"]] = panel["child_value"]
        exclude = list(locks.keys())
        candidates = _available_dims(df, locks, exclude)
        dim_cb["values"] = tuple(candidates)
        if candidates:
            if panel["dim_var"].get() not in candidates:
                panel["dim_var"].set(candidates[0])
        else:
            panel["dim_var"].set("")
        _compute_and_render()

    def set_ctx(ctx: Dict[str, object]):
        panel["ctx"] = ctx
        panel["child_dim"] = ctx.get("child_dim")
        panel["child_value"] = None
        m0 = ctx.get("default_metric","Units")
        panel["metric_var"].set("Units" if m0 not in ("Units","Share %") else m0)
        months = ctx.get("months", [])
        period_cb["values"] = tuple(months)
        if ctx.get("period"): panel["period_var"].set(ctx["period"])
        elif months: panel["period_var"].set(months[-1])
        else: panel["period_var"].set("(auto)")
        panel["view_var"].set("排名表"); panel["topn_var"].set(10)
        _refresh_dim_options()

    def on_change(*_): _compute_and_render()
    dim_cb.bind("<<ComboboxSelected>>", on_change); metric_cb.bind("<<ComboboxSelected>>", on_change)
    period_cb.bind("<<ComboboxSelected>>", on_change); view_cb.bind("<<ComboboxSelected>>", on_change)
    topn_sp.configure(command=_compute_and_render)

    def on_export():
        rows = [table.item(i)["values"] for i in table.get_children()]
        if not rows: messagebox.showinfo("导出", "无数据可导出。"); return
        path = filedialog.asksaveasfilename(title="保存红区结果CSV", defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["Item","Value"]); w.writerows(rows)
        messagebox.showinfo("导出", f"已保存到:\n{path}")
    export_btn.configure(command=on_export)

    # 当点击 Breakdown 行时，将该行作为 child_value 锁定
    def on_select_breakdown(event=None):
        sel = br.selection()
        if not sel:
            panel["child_value"] = None; _refresh_dim_options(); return
        name = br.item(sel[0]).get("values", ["",""])[0]
        if not name:
            panel["child_value"] = None; _refresh_dim_options(); return
        panel["child_value"] = name; _refresh_dim_options()
    br.bind("<<TreeviewSelect>>", on_select_breakdown)

    return {"kv":kv, "br":br, "status":status, "set_ctx":set_ctx,
            "dim_var":dim_var, "metric_var":metric_var, "period_var":period_var,
            "topn_var":topn_var, "view_var":view_var,
            "table":table, "fig":fig, "ax":ax, "canvas":canvas,
            "child_setter": lambda dim,val: ({"child_dim":dim, "child_value":val}, _refresh_dim_options())}

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

# -------------- App (two tabs only) --------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Market Trend Analyzer — v2.9 + RedPanel (2 tabs)")
        self.geometry("1600x960"); self.minsize(1280, 760)
        self.df: Optional[pd.DataFrame] = None; self.months: List[str] = []
        self._last_series = None
        self.CASCADE_ORDER = ["Make","Model","Body style (Proton)","Seats","Drivetrain type","Pure electric range (kms)"]
        self._build_ui()
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]): self.load_file(sys.argv[1])

    def _build_ui(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Excel…", command=self.on_open)
        filemenu.add_separator()
        filemenu.add_command(label="Export Chart as PNG…", command=self.on_export_png)
        filemenu.add_command(label="Export Series as CSV…", command=self.on_export_csv)
        filemenu.add_separator(); filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        toolsmenu = tk.Menu(menubar, tearoff=0)
        toolsmenu.add_command(label="Show column headers", command=self.on_show_headers)
        toolsmenu.add_command(label="Select month columns…", command=self.on_pick_months)
        menubar.add_cascade(label="Tools", menu=toolsmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=lambda: messagebox.showinfo("About","v2.9 + RedPanel (2 tabs)"))
        menubar.add_cascade(label="Help", menu=helpmenu); self.config(menu=menubar)

        top = ttk.Frame(self, padding=8); top.pack(side=tk.TOP, fill=tk.X)
        self.path_var = tk.StringVar(value="(No file loaded)")
        ttk.Label(top, text="Data file:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.path_var, width=120, anchor="w").pack(side=tk.LEFT, padx=(6,10))
        ttk.Button(top, text="Open Excel…", command=self.on_open).pack(side=tk.LEFT)

        self.tabs = ttk.Notebook(self); self.tabs.pack(fill=tk.BOTH, expand=True)
        self.tab_trend = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_trend, text="Sales Trend")
        self.tab_struct = ttk.Frame(self.tabs, padding=8); self.tabs.add(self.tab_struct, text="Market Structure")

        self._build_trend(self.tab_trend)
        self._build_structure(self.tab_struct)

    # utility
    def _slice_months(self, start: str, end: str) -> List[str]:
        if not self.months: return []
        s = self.months.index(start) if start in self.months else 0
        e = self.months.index(end) if end in self.months else len(self.months)-1
        if s>e: s,e = e,s
        return self.months[s:e+1]

    def _get_selected(self, lb: tk.Listbox) -> Optional[List[str]]:
        if lb is None or str(lb['state']) != 'normal': return None
        vals = [lb.get(i) for i in lb.curselection()]
        if not vals or "(All)" in vals: return None
        return vals

    def _apply_filters(self, df: pd.DataFrame, filter_map: Dict[str, tk.Listbox], skip_cols: Optional[List[str]]=None) -> pd.DataFrame:
        skip = set(skip_cols or [])
        for col, lb in filter_map.items():
            if col in skip: continue
            if col not in df.columns or str(lb['state']) != 'normal': continue
            vals = self._get_selected(lb)
            if vals is None: continue
            df = df[df[col].astype(str).isin(vals)]
        return df

    def _aggregate(self, df: pd.DataFrame, group_col: Optional[str], months: List[str]) -> pd.DataFrame:
        if df.empty: return pd.DataFrame(columns=["Series"]+months)
        if group_col is None or group_col == "Total":
            series = df[months].sum(axis=0).to_frame().T; series.insert(0, "Series", "Total"); return series
        g = df.groupby(group_col)[months].sum().reset_index().rename(columns={group_col:"Series"}); return g

    def _attach_hover(self, ax, lines, labels, unit_label, cb=None):
        if mplcursors is None: return
        try:
            cursor = mplcursors.cursor(lines, hover=True)
            @cursor.connect("add")
            def _(sel):
                x, y = sel.target; idx = int(round(x)); idx = max(0, min(idx, len(labels)-1))
                label = sel.artist.get_label()
                txt = f"{label}\n{labels[idx]}\n{y:.2f}%" if unit_label=="%" else f"{label}\n{labels[idx]}\n{y:,.0f}"
                sel.annotation.set(text=txt)
                if cb:
                    try: cb(label, idx)
                    except Exception: pass
        except Exception: pass

    # ---------- month picker ----------
    def on_pick_months(self):
        if self.df is None:
            messagebox.showinfo("Info", "Open a file first."); return
        top = tk.Toplevel(self); top.title("Select month columns"); top.geometry("520x520")
        ttk.Label(top, text="请选择月份列（多选）：").pack(anchor="w", padx=8, pady=(8,2))
        frm = ttk.Frame(top); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        lb = tk.Listbox(frm, selectmode=tk.MULTIPLE, exportselection=False)
        sb = ttk.Scrollbar(frm, orient="vertical", command=lb.yview); lb.configure(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.LEFT, fill=tk.Y)
        cols = [str(c) for c in self.df.columns]
        for i, c in enumerate(cols):
            lb.insert(tk.END, c)
            if c in self.months: lb.selection_set(i)
        btns = ttk.Frame(top); btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btns, text="全选", command=lambda: lb.selection_set(0, tk.END)).pack(side=tk.LEFT)
        ttk.Button(btns, text="清空", command=lambda: lb.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=6)
        def apply_and_close():
            picks = [lb.get(i) for i in lb.curselection()]
            if not picks: messagebox.showwarning("No selection", "请至少选择一列作为月份列"); return
            self.months = picks
            for cb in (self.trend_start_cb, self.trend_end_cb, self.struct_start_cb, self.struct_end_cb):
                cb["values"] = self.months
            self.trend_start.set(self.months[0]); self.trend_end.set(self.months[-1])
            self.struct_start.set(self.months[0]); self.struct_end.set(self.months[-1])
            top.destroy()
            try: self.on_plot_trend(); self.on_plot_structure()
            except Exception as e: messagebox.showwarning("Plot error", f"Plotting failed:\n{e}")
        ttk.Button(btns, text="确定", command=apply_and_close).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=top.destroy).pack(side=tk.RIGHT, padx=6)

    # ---------- Tab 1: Sales Trend ----------
    def _build_trend(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))
        pr = ttk.PanedWindow(root, orient=tk.HORIZONTAL); pr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        center = ttk.Frame(pr); right = ttk.Frame(pr, width=360); pr.add(center, weight=5); pr.add(right, weight=2)

        def mk_box(label):
            frame = ttk.LabelFrame(left, text=label, padding=6)
            lb = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=False, height=6)
            sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview); lb.configure(yscrollcommand=sb.set)
            lb.grid(row=0, column=0, sticky="nsew"); sb.grid(row=0, column=1, sticky="ns")
            frame.columnconfigure(0, weight=1); frame.rowconfigure(0, weight=1); return frame, lb

        self.lb_trend: Dict[str, tk.Listbox] = {}; self._trend_widgets: Dict[str, ttk.LabelFrame] = {}
        sections = [("Country","Country"),("Data type","Data type"),("Vehicle type","Vehicle type"),
                    ("Segment","Segment - Proton"),("Segment - Luxury (IDN only)","Segment - Luxury"),
                    ("Data source","Data source"),("Make","Make"),("Model","Model"),
                    ("Body style","Body style (Proton)"),("Seats","Seats"),("Drivetrain type","Drivetrain type"),
                    ("Pure electric range (kms)","Pure electric range (kms)"),("Price band","Price band")]
        for i,(lab,col) in enumerate(sections):
            f, lb = mk_box(lab); f.grid(row=i//2, column=i%2, sticky="nsew", padx=4, pady=4)
            self.lb_trend[col]=lb; self._trend_widgets[col]=f
        for r in range((len(sections)+1)//2): left.rowconfigure(r, weight=1)
        for c in range(2): left.columnconfigure(c, weight=1)

        opts = ttk.LabelFrame(left, text="Options", padding=6); opts.grid(row=7, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        self.trend_agg = tk.StringVar(value="Make")
        ttk.Label(opts, text="Aggregate by:").grid(row=0,column=0,sticky="w")
        ttk.Combobox(opts, textvariable=self.trend_agg, state="readonly",
                     values=["Total","Make","Segment - Proton","Segment - Luxury","Model","Country"]).grid(row=0,column=1,sticky="ew",padx=6)
        self.trend_metric = tk.StringVar(value="Trend")
        ttk.Label(opts, text="Metric:").grid(row=0,column=2,sticky="w")
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

        self.fig_trend = Figure(figsize=(8,5), dpi=100); self.ax_trend = self.fig_trend.add_subplot(111)
        self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, master=center); self.canvas_trend.draw(); self.canvas_trend.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        kpi = ttk.Frame(center); kpi.pack(fill=tk.X, pady=(8,0))
        self.kpi_trend = tk.StringVar(value="KPIs —"); ttk.Label(kpi, textvariable=self.kpi_trend, anchor="w").pack(fill=tk.X)

        self.trend_detail = build_detail_panel(right)
        self._trend_ctx = None

    
    def on_plot_trend(self):
        if self.df is None or not self.months: return
        months = self._slice_months(self.trend_start.get() or self.months[0], self.trend_end.get() or self.months[-1])
        skip_cols = ["Country"] if self.country_vs.get() else []
        fdf = self._apply_filters(self.df.copy(), self.lb_trend, skip_cols=skip_cols)
        agg_col = "Country" if self.country_vs.get() and ("Country" in fdf.columns) else (None if self.trend_agg.get()=="Total" else self.trend_agg.get())
        base = self._aggregate(fdf, agg_col, months)
        per_vals, per_labels = aggregate_time(base.iloc[:, 1:], months, self.trend_freq.get())
        base_period = pd.concat([base[["Series"]], per_vals], axis=1)
        ser = metric_transform_period(base_period, self.trend_metric.get(), self.trend_freq.get())

        self.ax_trend.clear(); x = np.arange(len(per_labels)); lines=[]
        for _, row in ser.iterrows():
            y = row[per_labels].astype(float).values
            ln, = self.ax_trend.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=str(row["Series"])); lines.append(ln)
        ylabel = "Units" if self.trend_metric.get()=="Trend" else "%"
        title_agg = "Country" if (agg_col=="Country") else self.trend_agg.get()
        self.ax_trend.set_title(f"Sales {self.trend_metric.get()} — {title_agg} ({self.trend_freq.get()})")
        self.ax_trend.set_xlabel("Period"); self.ax_trend.set_ylabel(ylabel)
        step = safe_xtick_step(len(x)); self.ax_trend.set_xticks(x[::step]); self.ax_trend.set_xticklabels([per_labels[i] for i in range(0,len(per_labels),step)], rotation=45, ha="right")
        self.ax_trend.grid(True, linestyle="--", alpha=0.4); self.ax_trend.legend(fontsize=8)
        self.fig_trend.tight_layout(); self.canvas_trend.draw()

        def on_hover(series_label, idx):
            row = ser[ser["Series"].astype(str)==series_label]
            if row.empty or idx>=len(per_labels): return
            val = float(row[per_labels[idx]].iloc[0]) if not pd.isna(row[per_labels[idx]].iloc[0]) else np.nan
            push_detail(self.trend_detail, {"Series": series_label, "Period": per_labels[idx], "Value": val})
        self._attach_hover(self.ax_trend, lines, per_labels, ylabel, cb=on_hover)
        for ln in lines:
            try: ln.set_picker(True); ln.set_pickradius(8)
            except Exception: pass

        # ------ important: red-panel context default (until a point + breakdown clicked) ------
        self.trend_detail["set_ctx"]({"df": fdf, "months": months, "locks": {},
                                      "period": per_labels[-1] if per_labels else None,
                                      "period_cols": [months[-1]] if months else [],
                                      "default_metric": "Units", "child_dim": None})

        # ------ NEW: click a point -> compute Breakdown and seed child_dim for drill ------
        def on_pick(event):
            if not hasattr(event, "ind") or len(event.ind)==0: return
            idx = int(event.ind[0])
            if idx<0 or idx>=len(per_labels): return
            series_label = event.artist.get_label()
            period_label = per_labels[idx]

            # decide child dimension for breakdown based on current aggregation
            if agg_col == "Country":
                child_col = "Make" if "Make" in fdf.columns else None
            elif agg_col == "Make":
                child_col = "Model" if "Model" in fdf.columns else None
            elif agg_col in ("Segment - Proton","Segment - Luxury"):
                child_col = "Make" if "Make" in fdf.columns else None
            elif agg_col == "Model":
                # 已经到 model，则尝试进一步拆分：Data source 或 Country
                child_col = "Data source" if "Data source" in fdf.columns else ("Country" if "Country" in fdf.columns else None)
            else:
                child_col = "Make" if "Make" in fdf.columns else None

            # determine which month columns compose the chosen period
            period_cols = []
            if self.trend_freq.get() == "Monthly":
                if period_label in months: period_cols = [period_label]
            else:
                for m in months:
                    out = _period_key(m, self.trend_freq.get())
                    if out and out[0] == period_label:
                        period_cols.append(m)

            # filter to the clicked series if aggregating
            dfsub = fdf.copy()
            locks = {}
            if agg_col is not None and agg_col in dfsub.columns:
                dfsub = dfsub[dfsub[agg_col].astype(str)==str(series_label)]
                locks[agg_col] = series_label

            breakdown = None
            if child_col is not None and period_cols:
                br = dfsub.groupby(child_col)[period_cols].sum(numeric_only=True)
                br["val"] = br.sum(axis=1)
                breakdown = br[["val"]].reset_index().sort_values(by="val", ascending=False).head(50)
                breakdown = breakdown[[child_col, "val"]]

            push_detail(self.trend_detail, {"Series": series_label, "Period": period_label}, breakdown)

            # update red-panel ctx so that when user clicks a breakdown row it locks correctly
            self.trend_detail["set_ctx"]({"df": fdf, "months": months, "locks": locks,
                                          "period": period_label, "period_cols": period_cols,
                                          "default_metric": "Units", "child_dim": child_col})

        if hasattr(self, "_mpl_pick_id_trend"):
            self.canvas_trend.mpl_disconnect(self._mpl_pick_id_trend)
        self._mpl_pick_id_trend = self.canvas_trend.mpl_connect('pick_event', on_pick)

        s = ser.iloc[0,1:].astype(float) if not ser.empty else pd.Series(dtype=float)
        total = float(np.nansum(s.values)) if len(s)>0 else 0.0
        latest = next((float(v) for v in reversed(s.values) if not (pd.isna(v))), np.nan) if len(s)>0 else np.nan
        self.kpi_trend.set(f"KPIs — Total in range: {total:,.0f} | Latest period: {latest:,.0f}")
        self._last_series = ser


    # ---------- Tab 2: Market Structure ----------
    def _build_structure(self, root):
        left = ttk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,8))
        pr = ttk.PanedWindow(root, orient=tk.HORIZONTAL); pr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        center = ttk.Frame(pr); right = ttk.Frame(pr, width=360); pr.add(center, weight=5); pr.add(right, weight=2)

        self.struct_dim = tk.StringVar(value="Drivetrain type")
        ttk.Label(left, text="Share dimension:").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.struct_dim, state="readonly",
                     values=["Drivetrain type","Body style (Proton)","Price band","Segment - Proton"]).pack(fill=tk.X)

        self.struct_start = tk.StringVar(); self.struct_end = tk.StringVar()
        self.struct_start_cb = ttk.Combobox(left, textvariable=self.struct_start, state="readonly"); self.struct_end_cb = ttk.Combobox(left, textvariable=self.struct_end, state="readonly")
        ttk.Label(left, text="Start").pack(anchor="w"); self.struct_start_cb.pack(fill=tk.X)
        ttk.Label(left, text="End").pack(anchor="w"); self.struct_end_cb.pack(fill=tk.X)
        ttk.Button(left, text="Plot Share Trend", command=self.on_plot_structure).pack(fill=tk.X, pady=8)

        self.fig_struct = Figure(figsize=(8,5), dpi=100); self.ax_struct = self.fig_struct.add_subplot(111)
        self.canvas_struct = FigureCanvasTkAgg(self.fig_struct, master=center); self.canvas_struct.draw(); self.canvas_struct.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.kpi_struct = tk.StringVar(value="Structure —"); ttk.Label(center, textvariable=self.kpi_struct).pack(anchor="w", pady=(6,0))

        self.struct_detail = build_detail_panel(right)

    def on_plot_structure(self):
        if self.df is None or not self.months: return
        months = self._slice_months(self.struct_start.get() or self.months[0], self.struct_end.get() or self.months[-1])
        dim = self.struct_dim.get(); df = self.df

        num = df.groupby(dim)[months].sum(numeric_only=True)
        denom = df[months].sum(axis=0, numeric_only=True).replace(0, np.nan)
        share = (num.div(denom, axis=1) * 100.0).reset_index().rename(columns={dim:"Series"})

        self.ax_struct.clear(); x=np.arange(len(months)); lines=[]
        for _, r in share.iterrows():
            ln, = self.ax_struct.plot(x, pd.to_numeric(r[months], errors="coerce").to_numpy(), marker="o", linewidth=1.6, markersize=3, label=str(r["Series"]))
            lines.append(ln)
        self.ax_struct.set_title(f"Market Share by {dim}"); self.ax_struct.grid(True, linestyle="--", alpha=0.4); self.ax_struct.legend(fontsize=8)
        step=safe_xtick_step(len(x)); self.ax_struct.set_xticks(x[::step]); self.ax_struct.set_xticklabels([months[i] for i in range(0,len(months),step)], rotation=45, ha="right")
        self.canvas_struct.draw_idle()

        # 默认红区（Structure 默认 Share %）
        self.struct_detail["set_ctx"]({"df": df, "months": months, "locks": {}, "period": months[-1] if months else None,
                                       "period_cols": [months[-1]] if months else [], "default_metric": "Share %", "child_dim": dim})

        def on_pick(event):
            if not hasattr(event, "ind") or len(event.ind)==0: return
            idx = int(event.ind[0]); 
            if idx<0 or idx>=len(months): return
            series_label = event.artist.get_label()
            month = months[idx]
            br = df.groupby(dim)[[month]].sum(numeric_only=True).reset_index().sort_values(by=month, ascending=False).head(50)
            breakdown = br[[dim, month]].rename(columns={dim:dim, month:"val"})
            push_detail(self.struct_detail, {"Series": series_label, "Month": month}, breakdown)
            self.struct_detail["set_ctx"]({"df": df, "months": months, "locks": {}, "period": month, "period_cols": [month],
                                           "default_metric": "Share %", "child_dim": dim})
        if hasattr(self, "_mpl_pick_id_struct"):
            self.canvas_struct.mpl_disconnect(self._mpl_pick_id_struct)
        self._mpl_pick_id_struct = self.canvas_struct.mpl_connect('pick_event', on_pick)
        for ln in lines:
            try: ln.set_picker(True); ln.set_pickradius(8)
            except Exception: pass

        latest = share.sort_values(by=months[-1], ascending=False)[["Series", months[-1]]].head(5) if len(months)>0 else pd.DataFrame(columns=["Series"])
        pieces = [f"{r.Series}: {r[months[-1]]:.1f}%" for _, r in latest.iterrows()] if len(latest)>0 else []
        self.kpi_struct.set("Latest composition — " + " | ".join(pieces) if pieces else "Latest composition — (no data)")
        self._last_series = share

    # ---------- File Ops ----------
    def on_open(self):
        path = filedialog.asksaveasfilename if False else filedialog.askopenfilename  # keep explicit for consistency
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

        self.path_var.set(path); self.df = df
        auto_months = detect_month_columns(df); self.months = auto_months if auto_months else []

        vals = tuple(self.months) if self.months else ()
        for cb in (self.trend_start_cb, self.trend_end_cb, self.struct_start_cb, self.struct_end_cb):
            cb["values"] = vals
        if self.months:
            self.trend_start.set(self.months[0]); self.trend_end.set(self.months[-1])
            self.struct_start.set(self.months[0]); self.struct_end.set(self.months[-1])

        for col_attr, lb in getattr(self, "lb_trend", {}).items():
            if col_attr in df.columns:
                lb.configure(state="normal"); vals = df[col_attr].dropna().astype(str).unique().tolist()
                lb.delete(0, tk.END); lb.insert(tk.END, "(All)")
                for v in sorted(vals): lb.insert(tk.END, v)
                lb.selection_set(0)
            else:
                lb.delete(0, tk.END); lb.insert(tk.END, "(N/A)"); lb.configure(state="disabled")

        if self.months:
            self.on_plot_trend(); self.on_plot_structure()

    def on_export_png(self):
        cur = self.tabs.index(self.tabs.select())
        fig = self.fig_trend if cur==0 else self.fig_struct
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
            self._last_series.to_csv(path, index=False); messagebox.showinfo("Saved", f"Series saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

    def on_show_headers(self):
        if self.df is None:
            messagebox.showinfo("Info", "Open a file first."); return
        cols = [str(c) for c in self.df.columns]; preview = "\n".join(cols[:200])
        messagebox.showinfo("Column headers (first 200)", preview)

if __name__ == "__main__":
    App().mainloop()
