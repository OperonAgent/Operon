"""tests/test_example_plugins.py — validate the seed example plugins load + work."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.plugin_sdk import PluginManager

EXAMPLES = ROOT / "plugins" / "examples"


@pytest.fixture(scope="module")
def pm():
    mgr = PluginManager(plugins_dir=EXAMPLES)
    mgr.load_all()
    return mgr


def _tool(pm, plugin, fn):
    return pm._plugins[plugin].tool_fns[fn]


class TestLoading:
    def test_examples_dir_exists(self):
        assert EXAMPLES.is_dir()

    def test_five_plugins_present(self):
        dirs = [d for d in EXAMPLES.iterdir() if d.is_dir()]
        assert len(dirs) >= 5

    def test_all_have_manifest_and_tools(self):
        for d in EXAMPLES.iterdir():
            if d.is_dir():
                assert (d / "plugin.json").exists(), f"{d.name} missing manifest"
                assert (d / "tools.py").exists(), f"{d.name} missing tools.py"

    def test_all_load(self, pm):
        assert len(pm._plugins) >= 5

    def test_no_load_errors(self, pm):
        for name, p in pm._plugins.items():
            assert not p.error, f"{name}: {p.error}"


class TestTextStats:
    def test_word_count(self, pm):
        r = _tool(pm, "text_stats", "text_stats")(text="hello world foo")
        assert r["words"] == 3

    def test_reading_time(self, pm):
        r = _tool(pm, "text_stats", "text_stats")(text="word " * 200, wpm=200)
        assert r["reading_time_min"] == pytest.approx(1.0, abs=0.1)

    def test_empty(self, pm):
        r = _tool(pm, "text_stats", "text_stats")(text="")
        assert r["words"] == 0 and r["success"]


class TestUuidGen:
    def test_uuid_format(self, pm):
        r = _tool(pm, "uuid_gen", "uuid_generate")(count=1)
        assert len(r["uuids"][0]) == 36 and r["uuids"][0].count("-") == 4

    def test_count_clamped(self, pm):
        r = _tool(pm, "uuid_gen", "uuid_generate")(count=999)
        assert r["count"] == 100

    def test_token_hex(self, pm):
        r = _tool(pm, "uuid_gen", "random_token")(nbytes=8, fmt="hex")
        assert len(r["token"]) == 16

    def test_token_urlsafe(self, pm):
        r = _tool(pm, "uuid_gen", "random_token")(nbytes=8, fmt="urlsafe")
        assert r["format"] == "urlsafe" and r["token"]


class TestCodec:
    def test_b64_roundtrip(self, pm):
        enc = _tool(pm, "codec", "base64_encode")(text="hello")
        dec = _tool(pm, "codec", "base64_decode")(data=enc["result"])
        assert dec["result"] == "hello"

    def test_url_roundtrip(self, pm):
        enc = _tool(pm, "codec", "url_encode")(text="a b&c")
        dec = _tool(pm, "codec", "url_decode")(text=enc["result"])
        assert dec["result"] == "a b&c"

    def test_b64_decode_bad(self, pm):
        r = _tool(pm, "codec", "base64_decode")(data="!!!not base64!!!")
        # tolerant decode returns success with replacement, or a clean error
        assert "success" in r


class TestHashing:
    def test_sha256_known(self, pm):
        r = _tool(pm, "hashing", "hash_text")(text="abc", algo="sha256")
        assert r["hexdigest"] == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_md5(self, pm):
        r = _tool(pm, "hashing", "hash_text")(text="abc", algo="md5")
        assert r["hexdigest"] == "900150983cd24fb0d6963f7d28e17f72"

    def test_bad_algo(self, pm):
        r = _tool(pm, "hashing", "hash_text")(text="x", algo="rot13")
        assert r["success"] is False


class TestJsonTools:
    def test_validate_ok(self, pm):
        r = _tool(pm, "json_tools", "json_validate")(text='{"a":1}')
        assert r["valid"] is True

    def test_validate_bad(self, pm):
        r = _tool(pm, "json_tools", "json_validate")(text="{not json}")
        assert r["valid"] is False

    def test_minify(self, pm):
        r = _tool(pm, "json_tools", "json_minify")(text='{ "a": 1 }')
        assert r["result"] == '{"a":1}'

    def test_pretty(self, pm):
        r = _tool(pm, "json_tools", "json_pretty")(text='{"a":1}', indent=2)
        assert "\n" in r["result"]
