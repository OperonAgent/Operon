"""Tests for tools/data_analysis.py"""
import json
import pytest
import tempfile
from pathlib import Path

pytest.importorskip("pandas",     reason="pandas not installed")
pytest.importorskip("numpy",      reason="numpy not installed")

from tools.data_analysis import (
    data_load, data_save, data_convert, data_describe,
    data_query, data_groupby, data_clean, data_merge,
    data_anomalies, data_correlations, _infer_format,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ROWS = [
    {"name": "Alice", "dept": "Eng",   "salary": 95000, "age": 30},
    {"name": "Bob",   "dept": "Eng",   "salary": 85000, "age": 25},
    {"name": "Carol", "dept": "Sales", "salary": 72000, "age": 35},
    {"name": "Dave",  "dept": "Sales", "salary": 68000, "age": 28},
    {"name": "Eve",   "dept": "Eng",   "salary": 110000, "age": 40},
    {"name": "Frank", "dept": "HR",    "salary": 65000, "age": 33},
]

SAMPLE_JSON = json.dumps(SAMPLE_ROWS)


@pytest.fixture
def csv_file(tmp_path):
    path = str(tmp_path / "data.csv")
    data_save(path, SAMPLE_JSON)
    return path


@pytest.fixture
def json_file(tmp_path):
    path = str(tmp_path / "data.json")
    data_save(path, SAMPLE_JSON, format="json")
    return path


# ── Format inference ──────────────────────────────────────────────────────────

class TestInferFormat:
    def test_csv(self):  assert _infer_format("data.csv") == "csv"
    def test_tsv(self):  assert _infer_format("data.tsv") == "tsv"
    def test_json(self): assert _infer_format("data.json") == "json"
    def test_xlsx(self): assert _infer_format("data.xlsx") == "excel"
    def test_parquet(self): assert _infer_format("data.parquet") == "parquet"
    def test_unknown_defaults_csv(self): assert _infer_format("data.txt") == "csv"


# ── Save / Load round-trip ────────────────────────────────────────────────────

class TestSaveLoad:
    def test_csv_round_trip(self, tmp_path):
        path = str(tmp_path / "out.csv")
        r_save = data_save(path, SAMPLE_JSON)
        assert r_save["success"], r_save["error"]
        assert Path(path).exists()

        r_load = data_load(path)
        assert r_load["success"], r_load["error"]
        assert r_load["output"]["shape"][0] == len(SAMPLE_ROWS)
        assert r_load["output"]["shape"][1] == 4

    def test_json_round_trip(self, tmp_path):
        path = str(tmp_path / "out.json")
        data_save(path, SAMPLE_JSON, format="json")
        r = data_load(path)
        assert r["success"]
        assert r["output"]["shape"][0] == len(SAMPLE_ROWS)

    def test_parquet_round_trip(self, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow not installed for parquet")
        path = str(tmp_path / "out.parquet")
        data_save(path, SAMPLE_JSON, format="parquet")
        r = data_load(path)
        assert r["success"]

    def test_load_missing_file(self):
        r = data_load("/no/such/file.csv")
        assert not r["success"]

    def test_columns_in_output(self, csv_file):
        r = data_load(csv_file)
        assert set(r["output"]["columns"]) == {"name", "dept", "salary", "age"}

    def test_head_rows(self, csv_file):
        r = data_load(csv_file)
        assert len(r["output"]["head"]) == min(5, len(SAMPLE_ROWS))


# ── Convert ───────────────────────────────────────────────────────────────────

class TestConvert:
    def test_csv_to_json(self, csv_file, tmp_path):
        out = str(tmp_path / "out.json")
        r = data_convert(csv_file, out)
        assert r["success"]
        assert Path(out).exists()
        import json as _json
        rows = _json.loads(Path(out).read_text())
        assert len(rows) == len(SAMPLE_ROWS)

    def test_json_to_csv(self, json_file, tmp_path):
        out = str(tmp_path / "out.csv")
        r = data_convert(json_file, out)
        assert r["success"]
        assert Path(out).exists()


# ── Describe ─────────────────────────────────────────────────────────────────

class TestDescribe:
    def test_shape(self, csv_file):
        r = data_describe(csv_file)
        assert r["success"]
        assert r["output"]["shape"] == [len(SAMPLE_ROWS), 4]

    def test_null_pct_all_zero(self, csv_file):
        r = data_describe(csv_file)
        for col, pct in r["output"]["null_pct"].items():
            assert pct == 0.0

    def test_specific_columns(self, csv_file):
        r = data_describe(csv_file, columns=["salary", "age"])
        assert r["success"]


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_numeric_filter(self, csv_file):
        r = data_query(csv_file, "salary > 80000")
        assert r["success"]
        # Alice (95k), Bob (85k), Eve (110k) → 3 rows
        assert r["output"]["total"] == 3

    def test_string_filter(self, csv_file):
        r = data_query(csv_file, "dept == 'Eng'")
        assert r["success"]
        assert r["output"]["total"] == 3

    def test_empty_result(self, csv_file):
        r = data_query(csv_file, "salary > 999999")
        assert r["success"]
        assert r["output"]["total"] == 0

    def test_limit_respected(self, csv_file):
        r = data_query(csv_file, "salary > 0", limit=2)
        assert r["success"]
        assert len(r["output"]["rows"]) <= 2

    def test_invalid_query(self, csv_file):
        r = data_query(csv_file, "not_a_column == 'x'")
        assert not r["success"]


# ── Groupby ───────────────────────────────────────────────────────────────────

class TestGroupby:
    def test_sum_by_dept(self, csv_file):
        r = data_groupby(csv_file, group_cols=["dept"], agg={"salary": "sum"})
        assert r["success"]
        rows = r["output"]
        assert isinstance(rows, list)
        depts = {row["dept"] for row in rows}
        assert "Eng" in depts and "Sales" in depts

    def test_count_by_dept(self, csv_file):
        r = data_groupby(csv_file, group_cols=["dept"], agg={"name": "count"})
        assert r["success"]


# ── Clean ─────────────────────────────────────────────────────────────────────

class TestClean:
    def test_normalize_column_names(self, tmp_path):
        import pandas as pd
        raw = tmp_path / "messy.csv"
        pd.DataFrame([{"First Name": "A", "Last  Name": "B"}]).to_csv(raw, index=False)
        out = str(tmp_path / "clean.csv")
        r = data_clean(str(raw), out, normalize_column_names=True)
        assert r["success"]
        r2 = data_load(out)
        assert "first_name" in r2["output"]["columns"]

    def test_drop_duplicates(self, tmp_path):
        import pandas as pd
        raw = tmp_path / "dups.csv"
        pd.DataFrame([{"x": 1}, {"x": 1}, {"x": 2}]).to_csv(raw, index=False)
        out = str(tmp_path / "clean.csv")
        r = data_clean(str(raw), out, drop_duplicates=True)
        assert r["success"]
        assert r["output"]["rows_after"] == 2

    def test_fill_nulls(self, tmp_path):
        import pandas as pd
        raw = tmp_path / "nulls.csv"
        pd.DataFrame([{"a": 1, "b": None}, {"a": 2, "b": 3}]).to_csv(raw, index=False)
        out = str(tmp_path / "filled.csv")
        r = data_clean(str(raw), out, fill_nulls=0)
        assert r["success"]


# ── Merge ─────────────────────────────────────────────────────────────────────

class TestMerge:
    def test_inner_join(self, tmp_path):
        import pandas as pd
        left  = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        out   = str(tmp_path / "merged.csv")
        pd.DataFrame([{"id": 1, "a": "x"}, {"id": 2, "a": "y"}]).to_csv(left, index=False)
        pd.DataFrame([{"id": 1, "b": 10},  {"id": 3, "b": 30}]).to_csv(right, index=False)
        r = data_merge(str(left), str(right), out, on="id", how="inner")
        assert r["success"]
        assert r["output"]["rows"] == 1  # only id=1 matches

    def test_left_join(self, tmp_path):
        import pandas as pd
        left  = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        out   = str(tmp_path / "merged.csv")
        pd.DataFrame([{"id": 1, "a": "x"}, {"id": 2, "a": "y"}]).to_csv(left, index=False)
        pd.DataFrame([{"id": 1, "b": 10}]).to_csv(right, index=False)
        r = data_merge(str(left), str(right), out, on="id", how="left")
        assert r["success"]
        assert r["output"]["rows"] == 2  # both left rows kept


# ── Anomaly detection ─────────────────────────────────────────────────────────

class TestAnomalies:
    def test_finds_outlier(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"value": [10, 11, 12, 10, 9, 11, 1000]})  # 1000 is outlier
        path = str(tmp_path / "data.csv")
        df.to_csv(path, index=False)
        r = data_anomalies(path, columns=["value"], method="zscore", threshold=2.0)
        assert r["success"]
        assert "value" in r["output"]["anomalies"]
        assert r["output"]["total_anomalies"] >= 1

    def test_iqr_method(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 100]})
        path = str(tmp_path / "data.csv")
        df.to_csv(path, index=False)
        r = data_anomalies(path, columns=["v"], method="iqr", threshold=1.5)
        assert r["success"]

    def test_no_outliers(self, csv_file):
        r = data_anomalies(csv_file, columns=["salary"], threshold=10.0)
        assert r["success"]
        # With very high threshold, nothing is an outlier
        assert r["output"]["total_anomalies"] == 0


# ── Correlations ─────────────────────────────────────────────────────────────

class TestCorrelations:
    def test_returns_matrix(self, csv_file):
        r = data_correlations(csv_file, threshold=0.5)
        assert r["success"]
        assert "matrix" in r["output"]
        assert "strong_pairs" in r["output"]

    def test_spearman(self, csv_file):
        r = data_correlations(csv_file, method="spearman")
        assert r["success"]
