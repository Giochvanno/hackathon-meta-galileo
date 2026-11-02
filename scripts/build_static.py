"""Build the static copy of the site for GitHub Pages: no server, no API calls, no key.

Everything the page shows (the replay of a real run, the SAM 3.1 video, the predictions,
the measured runs) is precomputed in docs/evidence, so the page can be served as plain
files. The live "measure your video" part is replaced by instructions to run the same
site locally with your own Meta Model API key.

    python scripts/build_static.py [--out site] [--url https://<user>.github.io/<repo>/]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_URL = "https://giochvanno.github.io/hackathon-meta-galileo/"


def build(out: Path, url: str = PAGES_URL) -> list[str]:
    from galileo import web

    if out.exists():
        shutil.rmtree(out)
    (out / "api").mkdir(parents=True)
    data = {
        "replay": web._replay(),
        "hero": web._hero_trace(),
        "scorecard": {"predictions": web._scorecard()},
        "examples": {"examples": web._examples()},
        "status": {"live": False, "static": True,
                   "reason": "this is the static copy of the site; run it locally with your key"},
    }
    text = json.dumps(data)
    files = sorted(set(re.findall(r'"/evidence/([^"]+)"', text)))
    for name, payload in data.items():
        body = json.dumps(payload, separators=(",", ":")).replace('"/evidence/', '"evidence/')
        (out / "api" / f"{name}.json").write_text(body, encoding="utf-8")
    for rel in files:
        src = (web.EVIDENCE / rel).resolve()
        if web.EVIDENCE.resolve() not in src.parents or not src.is_file():
            raise FileNotFoundError(f"the page refers to a missing evidence file: {rel}")
        dst = out / "evidence" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    html = (web.STATIC / "index.html").read_text(encoding="utf-8")
    marker = "const STATIC_SITE = false;"
    if marker not in html:
        raise RuntimeError("index.html no longer has the STATIC_SITE switch")
    html = html.replace(marker, "const STATIC_SITE = true;")
    # link previews need an absolute image address
    html = html.replace('content="static/og.png"', f'content="{url}static/og.png"')
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "static").mkdir()
    for f in ("logo.svg", "og.png"):
        shutil.copy2(web.STATIC / f, out / "static" / f)
    shutil.copy2(web.STATIC / "logo.svg", out / "favicon.ico")
    (out / ".nojekyll").write_text("")  # serve files as they are
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--url", default=PAGES_URL, help="where the site will live (for link previews)")
    a = ap.parse_args()
    files = build(Path(a.out), a.url)
    size = sum(p.stat().st_size for p in Path(a.out).rglob("*") if p.is_file())
    print(f"{a.out}: {len(files)} evidence files, {size / 1e6:.1f} MB in total")


if __name__ == "__main__":
    main()
