"""build_html.py — assemble a standalone, shareable index.html from the reviewed PAPER.md,
the 5 charts (base64-inlined), and the Lottie hero (inlined). Renders markdown client-side via
marked.js; lottie-web plays the hero. Figure refs in the md are rewritten to data URIs so the
file is fully self-contained."""
import base64, json, os, re

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")

def datauri(png):
    b = base64.b64encode(open(os.path.join(FIG, png), "rb").read()).decode()
    return f"data:image/png;base64,{b}"

md = open(os.path.join(HERE, "PAPER.md")).read()
# rewrite figures/figN_*.png -> data URI so the html is standalone
for png in os.listdir(FIG):
    if png.endswith(".png"):
        md = md.replace(f"figures/{png}", datauri(png))
        md = md.replace(f"paper/figures/{png}", datauri(png))

lottie = json.load(open(os.path.join(FIG, "flywheel_hero.json")))
md_js = json.dumps(md)
lottie_js = json.dumps(lottie)

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Flywheel That Spun But Didn't Climb</title>
<script src="https://cdn.jsdelivr.net/npm/lottie-web@5.12.2/build/player/lottie.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<style>
:root{--ink:#15202b;--mute:#5b6b7b;--blue:#2f6df0;--red:#e0563f;--line:#e6edf3;--bg:#fbfdff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:820px;margin:0 auto;padding:0 22px 90px;}
header{max-width:980px;margin:0 auto;padding:40px 22px 8px;text-align:center;}
#hero{width:100%;max-width:760px;height:430px;margin:0 auto;}
.eyebrow{letter-spacing:.14em;text-transform:uppercase;font-size:12px;color:var(--mute);font-weight:600;}
h1.title{font-size:34px;line-height:1.15;margin:.25em 0 .1em;font-weight:800;letter-spacing:-.01em;}
.sub{color:var(--mute);font-size:17px;margin:0 auto 6px;max-width:640px;}
.links{font-size:14px;color:var(--mute);margin:10px 0 0;}
.links a{color:var(--blue);text-decoration:none;font-weight:600;}
.tldr{background:#fff;border:1px solid var(--line);border-left:5px solid var(--blue);
 border-radius:12px;padding:18px 22px;margin:28px 0;box-shadow:0 1px 3px rgba(20,40,80,.05);}
.tldr h2{margin:.1em 0 .5em;font-size:15px;letter-spacing:.06em;text-transform:uppercase;color:var(--blue);}
.paper h1{font-size:27px;margin:1.6em 0 .4em;font-weight:800;letter-spacing:-.01em;}
.paper h2{font-size:21px;margin:1.7em 0 .4em;padding-top:.3em;border-top:1px solid var(--line);font-weight:700;}
.paper h3{font-size:17px;margin:1.3em 0 .3em;font-weight:700;}
.paper p{margin:.7em 0;} .paper li{margin:.3em 0;}
.paper img{display:block;max-width:100%;margin:22px auto;border:1px solid var(--line);border-radius:10px;
 background:#fff;box-shadow:0 1px 4px rgba(20,40,80,.06);}
.paper table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14.5px;}
.paper th,.paper td{border:1px solid var(--line);padding:7px 11px;text-align:left;}
.paper th{background:#f3f7fc;font-weight:700;} .paper tr:nth-child(even) td{background:#fafcff;}
.paper code{background:#f1f5fa;padding:.12em .4em;border-radius:5px;font-size:.9em;}
.paper pre{background:#0f1722;color:#dce6f0;padding:14px 16px;border-radius:10px;overflow:auto;font-size:13px;}
.paper pre code{background:none;color:inherit;padding:0;}
.paper blockquote{margin:1em 0;padding:.4em 16px;border-left:4px solid #d6e0ea;color:var(--mute);background:#f7fafd;border-radius:0 8px 8px 0;}
.paper a{color:var(--blue);} hr{border:none;border-top:1px solid var(--line);margin:2em 0;}
.foot{max-width:820px;margin:30px auto 0;padding:18px 22px;color:var(--mute);font-size:13px;text-align:center;border-top:1px solid var(--line);}
.badge{display:inline-block;background:#fdecea;color:var(--red);border:1px solid #f6cfc8;border-radius:999px;
 padding:3px 12px;font-size:12.5px;font-weight:700;letter-spacing:.02em;margin-top:10px;}
</style></head><body>
<header>
  <div class="eyebrow">Reversible-circuit synthesis · process report</div>
  <h1 class="title">The flywheel that spun but didn't climb</h1>
  <p class="sub">Can a small open model learn verifier-backed reversible-circuit synthesis — and improve itself? One result held up. One didn't. The honest write-up.</p>
  <div class="badge">HONEST NEGATIVE RESULT</div>
  <div id="hero"></div>
  <p class="links">
    Model: <a href="https://huggingface.co/dennisonb/reversible-circuit-8b-tool">🤗 dennisonb/reversible-circuit-8b-tool</a> ·
    Code &amp; full log: <a href="https://github.com/dennisonbertram/reversible-circuit-llm">GitHub</a>
  </p>
</header>
<div class="wrap"><div id="paper" class="paper"></div></div>
<div class="foot">Generated from the project's process log. Every number traces to the repo's eval data; the paper was drafted, adversarially reviewed, and citation-checked by separate agents.</div>
<script>
const PAPER_MD = __MD__;
const HERO = __LOTTIE__;
document.getElementById('paper').innerHTML = marked.parse(PAPER_MD);
try{ lottie.loadAnimation({container:document.getElementById('hero'),renderer:'svg',loop:true,autoplay:true,animationData:HERO}); }
catch(e){ document.getElementById('hero').style.display='none'; }
</script>
</body></html>"""

HTML = HTML.replace("__MD__", md_js).replace("__LOTTIE__", lottie_js)
out = os.path.join(HERE, "index.html")
open(out, "w").write(HTML)
print("wrote", out, "(", round(os.path.getsize(out)/1024), "KB )")
