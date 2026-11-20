"""Bundle the whole Streamlit app into one self-contained HTML file.

The page uses stlite (Streamlit on Pyodide/WebAssembly), so it runs entirely in
the browser: double-click the file, no Python install. The first load downloads
the Python runtime from the jsDelivr CDN (needs internet, ~30 s); the app code
and the data CSVs are embedded in the HTML itself.

Usage:  python tools/build_html.py            -> arlanda-saf-app.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STLITE = "1.12.1"
OUT = ROOT / "arlanda-saf-app.html"

INCLUDE = (
    ["app.py", ".streamlit/config.toml"]
    + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "saf_app").glob("*.py"))
    + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data" / "arlanda").glob("*.csv"))
    + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data" / "arlanda" / "tracks").glob("*.csv"))
)

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SAF Allocation Arlanda</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.css" />
<style>
  :root { --navy:#0b2545; --teal:#12867a; --bg:#ffffff; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#ffffff; } }
  body { margin:0; background:var(--bg); color:var(--navy); font-family:system-ui,-apple-system,Segoe UI,sans-serif; }
  #loading { max-width:640px; margin:12vh auto; padding:0 16px; text-align:center; }
  #loading h1 { font-size:1.4rem; }
  #loading p { line-height:1.5; }
  .bar { height:6px; background:#e6f2f0; border-radius:3px; overflow:hidden; margin:20px auto; max-width:320px; }
  .bar span { display:block; height:100%; width:35%; background:var(--teal); animation:slide 1.4s ease-in-out infinite; }
  @keyframes slide { 0%{margin-left:-35%} 100%{margin-left:100%} }
</style>
</head>
<body>
<div id="loading">
  <h1>Climate-Optimal SAF Allocation · Stockholm Arlanda</h1>
  <div class="bar"><span></span></div>
  <p>Starting Python in your browser. The first load downloads the runtime
  (about 30–60 seconds, needs internet); the app and its data are inside this file.</p>
</div>
<div id="root"></div>
<script type="module">
import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.js";
const files = __FILES__;
const obs = new MutationObserver(() => {
  if (document.querySelector('#root [data-testid="stApp"], #root .stApp')) {
    document.getElementById("loading").style.display = "none"; obs.disconnect();
  }
});
obs.observe(document.getElementById("root"), { childList: true, subtree: true });
mount({
  requirements: ["plotly", "scipy", "pandas", "numpy"],
  entrypoint: "app.py",
  files,
  streamlitConfig: {
    "theme.base": "light", "theme.primaryColor": "#12867a", "theme.backgroundColor": "#ffffff",
    "theme.secondaryBackgroundColor": "#f2f6f8", "theme.textColor": "#0b2545",
    "client.toolbarMode": "viewer"
  }
}, document.getElementById("root"));
</script>
</body>
</html>
"""


def build(out: Path = OUT) -> Path:
    files = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in INCLUDE}
    blob = json.dumps(files, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__V__", STLITE).replace("__FILES__", blob)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB, {len(INCLUDE)} files embedded)")
