# MarketTrendGUI (v2 with Hover)

A desktop GUI for **market trend analysis** using only the **Sales Data** worksheet of your Excel source.

## Features
- Robust month detection (`Jan '22`, `2022-01`, `202201`, `2022年01月`, etc.).
- Filters: Country, Make, Model, Vehicle type, Segment (Proton), Segment - Luxury, Data type, Drivetrain type, Price band (if any).
- Tabs: Sales Trend / Market Structure / Competitor Compare.
- Metrics: Trend / YoY% / MoM%.
- **Hover tooltips** on all charts (series, month, value).
- Export chart to PNG, current series to CSV.

## Quickstart
```bash
pip install -U -r requirements.txt
python MarketTrendGUI_v2_hover.py YOUR_DATA.xlsx
```

> If you prefer the plain v2 (no hover), just run `MarketTrendGUI_v2.py` instead.

## File list
- `MarketTrendGUI_v2_hover.py` – v2 + hover tooltips.
- `MarketTrendGUI_v2.py` – v2 baseline without hover.
- `requirements.txt` – Python dependencies.
- `.gitignore` – Python project ignores.