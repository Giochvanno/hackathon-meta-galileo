import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_static_site_has_everything_the_page_asks_for(tmp_path):
    spec = importlib.util.spec_from_file_location("build_static", ROOT / "scripts" / "build_static.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "site"
    files = mod.build(out)
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "const STATIC_SITE = true;" in html and 'content="https://' in html
    for name in ("replay", "hero", "scorecard", "examples", "status"):
        body = (out / "api" / f"{name}.json").read_text(encoding="utf-8")
        assert '"/evidence/' not in body  # relative, so it works under /<repo>/ on GitHub Pages
        json.loads(body)
    assert json.loads((out / "api" / "status.json").read_text())["live"] is False
    assert "p3_auto/overlay_web.mp4" in files
    for rel in files:
        assert (out / "evidence" / rel).is_file()
    assert not list(out.rglob(".env"))
