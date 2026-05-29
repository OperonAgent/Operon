# Operon Example Plugins

Five small, dependency-free plugins that demonstrate the Operon plugin SDK.
Each is a directory containing a `plugin.json` manifest and a `tools.py` module.

| Plugin | Tools | What it does |
|--------|-------|--------------|
| `text_stats` | `text_stats` | Word/char/line counts + reading-time estimate |
| `uuid_gen` | `uuid_generate`, `random_token` | UUID v4 + secure random tokens |
| `codec` | `base64_encode/decode`, `url_encode/decode` | Base64 & URL codecs |
| `hashing` | `hash_text` | MD5 / SHA-1 / SHA-256 / SHA-512 digests |
| `json_tools` | `json_validate`, `json_pretty`, `json_minify` | JSON validate/format/minify |

## Install one

```bash
operon
/plugins install plugins/examples/text_stats
/plugins reload
```

Or copy into your plugin directory directly:

```bash
cp -r plugins/examples/text_stats ~/.operon/plugins/
```

Then restart Operon — the new tools are auto-registered and become callable by
the model (and appear in `/tools`).

## Anatomy of a plugin

```
my_plugin/
├── plugin.json     # manifest: name, version, tools[], requires[]
└── tools.py        # one Python function per name listed in manifest.tools
```

**`plugin.json`**
```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "What it does",
  "tools": ["my_tool"],
  "requires": []
}
```

**`tools.py`**
```python
def my_tool(arg: str = "") -> dict:
    """Docstring becomes the tool description shown to the model."""
    return {"success": True, "result": arg.upper()}
```

### Conventions
- Each tool returns a `dict` (a `{"success": bool, ...}` shape is recommended).
- Keep tools side-effect-light and fast; gate heavy/optional deps behind
  `manifest.requires` so the installer can fetch them.
- The function's docstring is surfaced to the model as the tool description.
