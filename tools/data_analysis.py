"""
Operon Data Analysis Tool.

Powerful data manipulation and analysis using pandas + numpy.
Works on CSV, JSON, Excel, Parquet, SQLite, and in-memory data.

Capabilities:
  - Load/save/convert between formats (CSV, JSON, Excel, Parquet, TSV)
  - Statistical summaries (describe, value_counts, correlations)
  - Data cleaning (null handling, deduplication, type coercion)
  - Filtering, sorting, grouping, pivoting, merging
  - Column transforms, string ops, datetime parsing
  - Chart generation (matplotlib/seaborn → PNG)
  - SQL-like querying on DataFrames
  - Anomaly detection (Z-score, IQR)

Install:
    pip install pandas numpy matplotlib seaborn openpyxl

All functions return {success, output, error}
"""

from __future__ import annotations

import io
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

def _ok(output: Any)  -> dict: return {"success": True,  "output": output, "error": None}
def _err(msg: str)    -> dict: return {"success": False, "output": None,   "error": msg}

def _pd():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError("pandas not installed. Run: pip install pandas openpyxl")

def _np():
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy not installed. Run: pip install numpy")

_CHART_DIR = Path.home() / ".operon" / "charts"
_CHART_DIR.mkdir(parents=True, exist_ok=True)


# ── Loading / Saving ──────────────────────────────────────────────────────────

def data_load(
    path: str,
    format: Optional[str] = None,
    sheet: Optional[str] = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> dict:
    """
    Load tabular data from a file into a summary.

    Supported formats: csv, tsv, json, jsonl, excel (xlsx/xls), parquet, sqlite.

    Args:
        path:      File path.
        format:    Force format ("csv", "json", "excel", "parquet"). Auto-detected if None.
        sheet:     Excel sheet name (default = first sheet).
        encoding:  Text encoding (default utf-8).
        delimiter: CSV delimiter (default comma).

    Returns:
        {success, output: {rows, columns, dtypes, head, shape, null_counts, format}, error}
    """
    pd = _pd()
    try:
        fmt = format or _infer_format(path)
        df  = _load_df(pd, path, fmt, sheet, encoding, delimiter)
        return _ok(_df_summary(df, fmt))
    except Exception as e:
        return _err(f"data_load failed: {e}")


def data_save(
    path: str,
    data: Union[str, List[dict]],
    format: Optional[str] = None,
    index: bool = False,
) -> dict:
    """
    Save data to a file.

    Args:
        path:   Output file path.
        data:   JSON string or list of dicts representing rows.
        format: "csv", "json", "excel", "parquet", "tsv". Auto-detected from extension.
        index:  Include row index (default False).

    Returns:
        {success, output: str (saved path), error}
    """
    pd = _pd()
    try:
        rows = json.loads(data) if isinstance(data, str) else data
        df   = pd.DataFrame(rows)
        fmt  = format or _infer_format(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _save_df(df, path, fmt, index)
        return _ok(f"Saved {len(df)} rows × {len(df.columns)} cols → {path}")
    except Exception as e:
        return _err(f"data_save failed: {e}")


def data_convert(
    input_path: str,
    output_path: str,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
) -> dict:
    """
    Convert between data formats (e.g. CSV → Parquet, JSON → Excel).

    Args:
        input_path:    Source file.
        output_path:   Destination file.
        input_format:  Force input format (auto-detected if None).
        output_format: Force output format (auto-detected if None).

    Returns:
        {success, output: str, error}
    """
    pd = _pd()
    try:
        ifmt = input_format  or _infer_format(input_path)
        ofmt = output_format or _infer_format(output_path)
        df   = _load_df(pd, input_path, ifmt)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _save_df(df, output_path, ofmt)
        return _ok(f"Converted {input_path} ({ifmt}) → {output_path} ({ofmt})  [{len(df)} rows]")
    except Exception as e:
        return _err(f"data_convert failed: {e}")


# ── Analysis ──────────────────────────────────────────────────────────────────

def data_describe(
    path: str,
    columns: Optional[List[str]] = None,
    format: Optional[str] = None,
) -> dict:
    """
    Statistical summary of a dataset.

    Args:
        path:    File path.
        columns: Specific columns to describe (None = all numeric columns).
        format:  File format (auto-detected if None).

    Returns:
        {success, output: {describe, value_counts_top, null_pct, dtypes, shape}, error}
    """
    pd = _pd()
    try:
        fmt = format or _infer_format(path)
        df  = _load_df(pd, path, fmt)
        if columns:
            df = df[columns]

        desc    = df.describe(include="all").to_dict()
        null_pct = {col: round(df[col].isna().mean() * 100, 1) for col in df.columns}

        # Top value counts for categorical cols
        vc: Dict[str, Any] = {}
        for col in df.select_dtypes(include="object").columns:
            vc[col] = df[col].value_counts().head(5).to_dict()

        return _ok({
            "shape":       list(df.shape),
            "dtypes":      {c: str(t) for c, t in df.dtypes.items()},
            "null_pct":    null_pct,
            "describe":    {k: {str(ki): str(vi) for ki, vi in v.items()} for k, v in desc.items()},
            "top_values":  vc,
        })
    except Exception as e:
        return _err(f"data_describe failed: {e}")


def data_query(
    path: str,
    query: str,
    format: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """
    Filter rows using a pandas query expression.

    Args:
        path:   File path.
        query:  Pandas query string, e.g. "age > 30 and salary < 100000".
        format: File format (auto-detected).
        limit:  Max rows to return (default 100).

    Returns:
        {success, output: {rows: list[dict], count, total, query}, error}
    """
    pd = _pd()
    try:
        fmt = format or _infer_format(path)
        df  = _load_df(pd, path, fmt)
        filtered = df.query(query)
        total = len(filtered)
        sample = filtered.head(limit)
        return _ok({
            "query":  query,
            "count":  min(total, limit),
            "total":  total,
            "rows":   sample.to_dict(orient="records"),
        })
    except Exception as e:
        return _err(f"data_query failed: {e}")


def data_groupby(
    path: str,
    group_cols: List[str],
    agg: Dict[str, str],
    format: Optional[str] = None,
) -> dict:
    """
    Group data and compute aggregations.

    Args:
        path:       File path.
        group_cols: Columns to group by, e.g. ["region", "product"].
        agg:        Dict mapping column → aggregation, e.g. {"sales": "sum", "qty": "mean"}.
        format:     File format.

    Returns:
        {success, output: list of aggregated rows, error}
    """
    pd = _pd()
    try:
        fmt    = format or _infer_format(path)
        df     = _load_df(pd, path, fmt)
        result = df.groupby(group_cols).agg(agg).reset_index()
        return _ok(result.to_dict(orient="records"))
    except Exception as e:
        return _err(f"data_groupby failed: {e}")


def data_clean(
    path: str,
    output: str,
    drop_nulls: bool = False,
    fill_nulls: Optional[Any] = None,
    drop_duplicates: bool = True,
    strip_whitespace: bool = True,
    normalize_column_names: bool = True,
    format: Optional[str] = None,
) -> dict:
    """
    Clean a dataset: handle nulls, deduplication, whitespace, column names.

    Args:
        path:                   Input file path.
        output:                 Output file path.
        drop_nulls:             Drop rows with any null value.
        fill_nulls:             Value to fill nulls (e.g. 0, "unknown", "ffill").
                                "ffill"/"bfill" = forward/backward fill.
        drop_duplicates:        Remove duplicate rows (default True).
        strip_whitespace:       Strip leading/trailing whitespace from string cols.
        normalize_column_names: Lowercase + underscore column names.
        format:                 File format.

    Returns:
        {success, output: {rows_before, rows_after, cols, path}, error}
    """
    pd = _pd()
    try:
        fmt          = format or _infer_format(path)
        df           = _load_df(pd, path, fmt)
        rows_before  = len(df)

        if normalize_column_names:
            import re
            df.columns = [
                re.sub(r"[^a-z0-9]+", "_", c.lower().strip()).strip("_")
                for c in df.columns
            ]

        if strip_whitespace:
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()

        if drop_nulls:
            df = df.dropna()
        elif fill_nulls is not None:
            if fill_nulls in ("ffill", "bfill"):
                df = df.fillna(method=fill_nulls)
            else:
                df = df.fillna(fill_nulls)

        if drop_duplicates:
            df = df.drop_duplicates()

        out_fmt = _infer_format(output)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _save_df(df, output, out_fmt)

        return _ok({
            "rows_before": rows_before,
            "rows_after":  len(df),
            "rows_removed": rows_before - len(df),
            "columns":     list(df.columns),
            "path":        output,
        })
    except Exception as e:
        return _err(f"data_clean failed: {e}")


def data_merge(
    left_path: str,
    right_path: str,
    output: str,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: str = "inner",
) -> dict:
    """
    Merge/join two datasets.

    Args:
        left_path:  Path to left dataset.
        right_path: Path to right dataset.
        output:     Output path.
        on:         Column(s) to join on (same name in both).
        left_on:    Left join column (if different names).
        right_on:   Right join column (if different names).
        how:        "inner", "left", "right", "outer" (default "inner").

    Returns:
        {success, output: {rows, path}, error}
    """
    pd = _pd()
    try:
        left  = _load_df(pd, left_path,  _infer_format(left_path))
        right = _load_df(pd, right_path, _infer_format(right_path))

        merged = left.merge(right, on=on, left_on=left_on, right_on=right_on, how=how)
        out_fmt = _infer_format(output)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _save_df(merged, output, out_fmt)
        return _ok({"rows": len(merged), "cols": list(merged.columns), "path": output})
    except Exception as e:
        return _err(f"data_merge failed: {e}")


def data_pivot(
    path: str,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "sum",
    output: Optional[str] = None,
    format: Optional[str] = None,
) -> dict:
    """
    Create a pivot table.

    Args:
        path:     Input file.
        index:    Row grouping column.
        columns:  Column grouping column.
        values:   Values column to aggregate.
        aggfunc:  "sum", "mean", "count", "min", "max" (default "sum").
        output:   Save pivot to file (optional).
        format:   Input format (auto-detected).

    Returns:
        {success, output: list of rows (or saved path), error}
    """
    pd = _pd()
    try:
        fmt  = format or _infer_format(path)
        df   = _load_df(pd, path, fmt)
        pivot = pd.pivot_table(df, index=index, columns=columns, values=values,
                               aggfunc=aggfunc, fill_value=0)
        pivot = pivot.reset_index()

        if output:
            out_fmt = _infer_format(output)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            _save_df(pivot, output, out_fmt)
            return _ok(f"Pivot table saved → {output}")
        return _ok(pivot.to_dict(orient="records"))
    except Exception as e:
        return _err(f"data_pivot failed: {e}")


def data_anomalies(
    path: str,
    columns: Optional[List[str]] = None,
    method: str = "zscore",
    threshold: float = 3.0,
    format: Optional[str] = None,
) -> dict:
    """
    Detect anomalies/outliers in numeric columns.

    Args:
        path:      File path.
        columns:   Columns to check (None = all numeric).
        method:    "zscore" (default) or "iqr".
        threshold: Z-score threshold (default 3.0) or IQR multiplier.
        format:    File format.

    Returns:
        {success, output: {anomalies: {col: [row_indices]}, counts, summary}, error}
    """
    pd = _pd()
    np = _np()
    try:
        fmt  = format or _infer_format(path)
        df   = _load_df(pd, path, fmt)
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        anomaly_map: Dict[str, List[int]] = {}
        for col in cols:
            series = df[col].dropna()
            if method == "zscore":
                z = np.abs((series - series.mean()) / series.std())
                bad = z[z > threshold].index.tolist()
            elif method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr    = q3 - q1
                lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
                bad    = series[(series < lo) | (series > hi)].index.tolist()
            else:
                bad = []
            if bad:
                anomaly_map[col] = bad

        total = sum(len(v) for v in anomaly_map.values())
        return _ok({
            "method":   method,
            "threshold": threshold,
            "anomalies": anomaly_map,
            "total_anomalies": total,
            "affected_columns": list(anomaly_map.keys()),
        })
    except Exception as e:
        return _err(f"data_anomalies failed: {e}")


# ── Chart Generation ──────────────────────────────────────────────────────────

def data_chart(
    path: str,
    chart_type: str,
    x: str,
    y: Optional[Union[str, List[str]]] = None,
    title: str = "",
    output: Optional[str] = None,
    color_by: Optional[str] = None,
    figsize: str = "10x6",
    format: Optional[str] = None,
) -> dict:
    """
    Generate a chart from tabular data.

    Chart types: bar, barh, line, scatter, hist, box, pie, area, heatmap.

    Args:
        path:       Data file path.
        chart_type: One of: bar, barh, line, scatter, hist, box, pie, area, heatmap.
        x:          X-axis column (or column to use for hist/box/pie).
        y:          Y-axis column(s). Can be a list for multi-line charts.
        title:      Chart title.
        output:     Output PNG path. Auto-generated if None.
        color_by:   Column to use for colour grouping.
        figsize:    Width x Height in inches, e.g. "10x6".
        format:     Data file format.

    Returns:
        {success, output: {path, chart_type, title}, error}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return _err("matplotlib/seaborn not installed. Run: pip install matplotlib seaborn")

    pd = _pd()
    try:
        fmt = format or _infer_format(path)
        df  = _load_df(pd, path, fmt)

        # Parse figsize
        try:
            fw, fh = [float(v) for v in figsize.split("x")]
        except Exception:
            fw, fh = 10.0, 6.0

        fig, ax = plt.subplots(figsize=(fw, fh))
        sns.set_theme(style="darkgrid")

        ct = chart_type.lower()
        if ct == "bar":
            if color_by:
                sns.barplot(data=df, x=x, y=y, hue=color_by, ax=ax)
            else:
                df.groupby(x)[y].sum().plot(kind="bar", ax=ax)
        elif ct == "barh":
            df.groupby(x)[y].sum().sort_values().plot(kind="barh", ax=ax)
        elif ct == "line":
            ycols = y if isinstance(y, list) else [y]
            for yc in ycols:
                ax.plot(df[x], df[yc], label=yc, marker="o", markersize=3)
            ax.legend()
        elif ct == "scatter":
            if color_by and color_by in df.columns:
                groups = df[color_by].unique()
                for g in groups:
                    sub = df[df[color_by] == g]
                    ax.scatter(sub[x], sub[y], label=str(g), alpha=0.7)
                ax.legend()
            else:
                ax.scatter(df[x], df[y], alpha=0.7)
        elif ct == "hist":
            df[x].dropna().plot(kind="hist", bins=30, ax=ax, edgecolor="white")
        elif ct == "box":
            ycols = y if isinstance(y, list) else ([y] if y else None)
            target_df = df[ycols] if ycols else df.select_dtypes(include="number")
            target_df.plot(kind="box", ax=ax)
        elif ct == "pie":
            counts = df[x].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        elif ct == "area":
            ycols = y if isinstance(y, list) else [y]
            df.set_index(x)[ycols].plot(kind="area", ax=ax, alpha=0.7)
        elif ct == "heatmap":
            num_df = df.select_dtypes(include="number")
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
        else:
            return _err(f"Unknown chart type: {chart_type}. "
                        "Use: bar, barh, line, scatter, hist, box, pie, area, heatmap")

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x)
        if y and ct not in ("hist", "box", "pie", "heatmap"):
            ylabel = y if isinstance(y, str) else ", ".join(y)
            ax.set_ylabel(ylabel)

        plt.tight_layout()

        if not output:
            import time, hashlib
            slug = hashlib.md5(f"{path}{ct}{x}".encode()).hexdigest()[:6]
            output = str(_CHART_DIR / f"chart_{int(time.time())}_{slug}.png")

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return _ok({"path": output, "chart_type": ct, "title": title})
    except Exception as e:
        return _err(f"data_chart failed: {e}")


# ── Correlation / Statistics ──────────────────────────────────────────────────

def data_correlations(
    path: str,
    method: str = "pearson",
    threshold: float = 0.7,
    format: Optional[str] = None,
) -> dict:
    """
    Compute pairwise correlations and flag strong ones.

    Args:
        path:      File path.
        method:    "pearson", "spearman", or "kendall".
        threshold: Flag correlations above this absolute value.
        format:    File format.

    Returns:
        {success, output: {strong_pairs, matrix}, error}
    """
    pd = _pd()
    try:
        fmt    = format or _infer_format(path)
        df     = _load_df(pd, path, fmt)
        corr   = df.select_dtypes(include="number").corr(method=method)
        matrix = {c: {k: round(v, 4) for k, v in row.items()} for c, row in corr.to_dict().items()}

        strong: List[dict] = []
        seen = set()
        for col_a in corr.columns:
            for col_b in corr.columns:
                if col_a == col_b:
                    continue
                key = tuple(sorted([col_a, col_b]))
                if key in seen:
                    continue
                seen.add(key)
                val = corr.loc[col_a, col_b]
                if abs(val) >= threshold:
                    strong.append({"col_a": col_a, "col_b": col_b,
                                   "correlation": round(float(val), 4)})

        strong.sort(key=lambda r: abs(r["correlation"]), reverse=True)
        return _ok({"method": method, "strong_pairs": strong, "matrix": matrix})
    except Exception as e:
        return _err(f"data_correlations failed: {e}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _infer_format(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {
        "csv": "csv", "tsv": "tsv", "txt": "csv",
        "json": "json", "jsonl": "jsonl",
        "xlsx": "excel", "xls": "excel",
        "parquet": "parquet", "pq": "parquet",
        "db": "sqlite", "sqlite": "sqlite", "sqlite3": "sqlite",
    }.get(ext, "csv")


def _load_df(pd: Any, path: str, fmt: str,
             sheet: Optional[str] = None,
             encoding: str = "utf-8",
             delimiter: str = ",") -> Any:
    if fmt == "csv":
        return pd.read_csv(path, encoding=encoding, sep=delimiter, on_bad_lines="warn")
    elif fmt == "tsv":
        return pd.read_csv(path, encoding=encoding, sep="\t", on_bad_lines="warn")
    elif fmt == "json":
        return pd.read_json(path, encoding=encoding)
    elif fmt == "jsonl":
        return pd.read_json(path, lines=True, encoding=encoding)
    elif fmt == "excel":
        return pd.read_excel(path, sheet_name=sheet or 0)
    elif fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "sqlite":
        import sqlite3
        con = sqlite3.connect(path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
        tbl = tables.iloc[0]["name"] if len(tables) > 0 else None
        if not tbl:
            raise ValueError("No tables found in SQLite database")
        return pd.read_sql_query(f"SELECT * FROM {tbl}", con)
    else:
        return pd.read_csv(path, encoding=encoding)


def _save_df(df: Any, path: str, fmt: str, index: bool = False) -> None:
    if fmt == "csv":
        df.to_csv(path, index=index)
    elif fmt == "tsv":
        df.to_csv(path, index=index, sep="\t")
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True)
    elif fmt == "excel":
        df.to_excel(path, index=index)
    elif fmt == "parquet":
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)


def _df_summary(df: Any, fmt: str) -> dict:
    rows, cols = df.shape
    null_counts = df.isnull().sum().to_dict()
    return {
        "shape":       [rows, cols],
        "columns":     list(df.columns),
        "dtypes":      {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {k: int(v) for k, v in null_counts.items()},
        "head":        df.head(5).to_dict(orient="records"),
        "format":      fmt,
    }
