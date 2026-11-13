#!/usr/bin/env python3
"""Build or unpack the grouped encrypted hub (pages/encrypted/index.html).

The menu (group names and page titles) is plain text; every page's content is
encrypted separately with AES-256-GCM, all under one PBKDF2-SHA256 key.

Build:  PAGE_PASSWORD=... python3 scripts/encrypt_hub.py manifest.json pages/encrypted/index.html
Unpack: PAGE_PASSWORD=... python3 scripts/encrypt_hub.py --decrypt pages/encrypted/index.html outdir/
        (writes outdir/manifest.json and one .html per page, ready to edit and rebuild)

manifest.json:
  {"title": "Encrypted pages",
   "groups": [{"name": "...", "pages": [{"id": "slug", "title": "...", "file": "page.html"}]}]}
File paths are relative to the manifest. Never commit the manifest folder or the password.
"""
import base64, html, json, os, re, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITER = 600000
b64 = lambda b: base64.b64encode(b).decode()
unb64 = base64.b64decode


def derive(password, salt, it=ITER):
    return PBKDF2HMAC(SHA256(), 32, salt, it).derive(password.encode())


SHELL = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f5f2;--side:#fff;--fg:#1f2421;--fg2:#646d67;--line:#e0e3de;--accent:#2457c5;--accent-fg:#fff;--soft:#e8effa;--warn:#9c4e08}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111316;--side:#181b20;--fg:#e6e9ee;--fg2:#98a1ad;--line:#2a2f37;--accent:#7fa8ff;--accent-fg:#0d1320;--soft:#1e2a3d;--warn:#f0a44b}}
:root[data-theme=dark]{--bg:#111316;--side:#181b20;--fg:#e6e9ee;--fg2:#98a1ad;--line:#2a2f37;--accent:#7fa8ff;--accent-fg:#0d1320;--soft:#1e2a3d;--warn:#f0a44b}
:root[data-theme=light]{color-scheme:light}:root[data-theme=dark]{color-scheme:dark}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);display:grid;grid-template-columns:270px 1fr;grid-template-rows:100%}
aside{background:var(--side);border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
.brand{padding:16px 18px 10px;display:flex;align-items:center;gap:8px}
.brand a{color:var(--fg2);text-decoration:none;font-size:13px}
.brand a:hover{color:var(--accent)}
.brand h1{font-size:16px;margin:0;flex:1}
.status{font-size:12px;color:var(--fg2);padding:0 18px 10px;display:flex;align-items:center;gap:6px}
.status .dot{width:8px;height:8px;border-radius:50%;background:var(--warn)}
body.open .status .dot{background:#2ea043}
nav{overflow-y:auto;padding:4px 10px 16px;flex:1}
.grp{margin:12px 0 4px;padding:0 8px;font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--fg2);font-weight:600}
nav button{display:flex;align-items:center;gap:8px;width:100%;text-align:left;font:inherit;font-size:14px;color:var(--fg);background:none;border:0;border-radius:6px;padding:7px 8px;cursor:pointer}
nav button:hover{background:var(--soft)}
nav button.on{background:var(--accent);color:var(--accent-fg)}
nav button .ic{font-size:12px;opacity:.6;width:14px;flex-shrink:0}
body.open nav button .ic{display:none}
.foot{padding:10px 18px;border-top:1px solid var(--line);display:flex;gap:8px}
.foot button{font:inherit;font-size:13px;width:100%;padding:7px;border-radius:6px;border:1px solid var(--line);background:none;color:var(--fg);cursor:pointer}
main{position:relative;min-width:0;min-height:0}
iframe{position:absolute;inset:0;width:100%;height:100%;border:0;background:var(--bg);display:none}
iframe.on{display:block}
#gate{height:100%;display:grid;place-items:center;padding:20px}
#gate form{width:min(360px,100%);background:var(--side);border:1px solid var(--line);border-radius:12px;padding:24px}
#gate h2{margin:0 0 4px;font-size:19px}
#gate p{margin:0 0 16px;color:var(--fg2);font-size:14px}
#gate input,#gate button{width:100%;padding:10px 12px;border-radius:8px;font:inherit}
#gate input{border:1px solid var(--line);background:var(--bg);color:var(--fg);margin-bottom:8px}
#gate button{border:0;background:var(--accent);color:var(--accent-fg);font-weight:600;cursor:pointer}
#err{min-height:20px;margin-top:10px;font-size:14px;color:var(--warn)}
body.open #gate{display:none}
#menuBtn{display:none}
@media(max-width:760px){
  body{grid-template-columns:1fr;grid-template-rows:auto 1fr}
  aside{border-right:0;border-bottom:1px solid var(--line);max-height:100vh}
  #menuBtn{display:inline-block;font:inherit;font-size:13px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:none;color:var(--fg);cursor:pointer}
  aside.shut nav,aside.shut .foot,aside.shut .status{display:none}
  nav{max-height:60vh}
}
</style></head><body>
<aside id="side">
  <div class="brand"><h1>__TITLE__</h1><button id="menuBtn" type="button" aria-expanded="true">Menu</button></div>
  <div class="status"><span class="dot"></span><span id="st">Locked</span></div>
  <nav id="nav" aria-label="Encrypted pages"></nav>
  <div class="foot"><button id="theme" type="button" title="Switch light / dark mode">Theme: Auto</button><button id="lock" type="button">Lock</button></div>
</aside>
<main id="main">
  <div id="gate"><form id="f"><h2>Locked</h2><p id="hint">Enter the password to open these pages.</p>
    <input id="p" type="password" autocomplete="current-password" autofocus aria-label="Password">
    <button>Unlock</button><div id="err" role="status"></div></form></div>
</main>
<script>
const D=__DATA__;
const u=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const nav=document.getElementById('nav'),main=document.getElementById('main'),side=document.getElementById('side');
let key=null,want=null;const pages={};
D.groups.forEach(g=>{const h=document.createElement('div');h.className='grp';h.textContent=g.name;nav.appendChild(h);
  g.pages.forEach(p=>{pages[p.id]=p;const b=document.createElement('button');b.type='button';b.dataset.id=p.id;
    b.innerHTML='<span class="ic">&#128274;</span>';b.appendChild(document.createTextNode(p.title));
    b.onclick=()=>open_(p.id);nav.appendChild(b);});});
async function dec(p){return new TextDecoder().decode(await crypto.subtle.decrypt({name:'AES-GCM',iv:u(p.n)},key,u(p.c)));}
async function open_(id){
  if(!pages[id])return;
  if(!key){want=id;document.getElementById('hint').textContent='Enter the password to open “'+pages[id].title+'”.';document.getElementById('p').focus();return;}
  [...nav.querySelectorAll('button')].forEach(b=>b.classList.toggle('on',b.dataset.id===id));
  [...main.querySelectorAll('iframe')].forEach(f=>f.classList.remove('on'));
  let f=document.getElementById('f-'+id);
  if(!f){f=document.createElement('iframe');f.id='f-'+id;f.title=pages[id].title;f.setAttribute('allow','clipboard-write');f.addEventListener('load',()=>themeFrame(f));f.srcdoc=await dec(pages[id]);main.appendChild(f);}
  f.classList.add('on');document.title=pages[id].title+' · '+D.title;
  try{history.replaceState(null,'','#'+id)}catch(e){}
  if(matchMedia('(max-width:760px)').matches)setMenu(false);
}
// Theme: Auto follows the system; Light/Dark force it on the menu and, through the iframe's
// color-scheme, on every page's prefers-color-scheme rules.
let theme='auto';try{theme=localStorage.getItem('hub-theme')||'auto'}catch(e){}
// Pages are same-origin srcdoc frames: rewrite their prefers-color-scheme media rules in place.
function themeFrame(f){let d;try{d=f.contentDocument}catch(e){return}if(!d)return;
  d.documentElement.style.colorScheme=theme==='auto'?'':theme;
  const walk=rules=>{for(const r of rules){if(r.media&&r.conditionText!==undefined||r.media){const o=r.__o||(r.__o=r.media.mediaText);
      if(/prefers-color-scheme/.test(o)){const dark=/dark/.test(o);r.media.mediaText=theme==='auto'?o:((theme==='dark')===dark?'all':'not all');}}
    if(r.cssRules)walk(r.cssRules);}};
  for(const sh of d.styleSheets){try{walk(sh.cssRules)}catch(e){}}}
function applyTheme(){if(theme==='auto')document.documentElement.removeAttribute('data-theme');else document.documentElement.setAttribute('data-theme',theme);
  document.getElementById('theme').textContent='Theme: '+theme[0].toUpperCase()+theme.slice(1);
  [...main.querySelectorAll('iframe')].forEach(themeFrame);}
document.getElementById('theme').onclick=()=>{theme={auto:'light',light:'dark',dark:'auto'}[theme];try{localStorage.setItem('hub-theme',theme)}catch(e){}applyTheme();};
applyTheme();
function setMenu(o){side.classList.toggle('shut',!o);document.getElementById('menuBtn').setAttribute('aria-expanded',o);}
document.getElementById('menuBtn').onclick=()=>setMenu(side.classList.contains('shut'));
document.getElementById('lock').onclick=()=>location.reload();
document.getElementById('f').addEventListener('submit',async ev=>{ev.preventDefault();const e=document.getElementById('err');e.textContent='Unlocking…';
  try{const m=await crypto.subtle.importKey('raw',new TextEncoder().encode(document.getElementById('p').value),'PBKDF2',false,['deriveKey']);
    key=await crypto.subtle.deriveKey({name:'PBKDF2',hash:'SHA-256',salt:u(D.s),iterations:D.i},m,{name:'AES-GCM',length:256},false,['decrypt']);
    const first=D.groups[0].pages[0];await dec(first);
    document.body.classList.add('open');document.getElementById('st').textContent='Unlocked';e.textContent='';
    open_(want||first.id);
  }catch(x){key=null;e.textContent='Wrong password.';}});
const h=decodeURIComponent(location.hash.slice(1));if(pages[h]){want=h;document.getElementById('hint').textContent='Enter the password to open “'+pages[h].title+'”.';}
</script></body></html>
"""


def build(manifest_path, out):
    password = os.environ["PAGE_PASSWORD"]
    man = json.load(open(manifest_path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(manifest_path))
    salt = os.urandom(16)
    aes = AESGCM(derive(password, salt))
    groups, seen = [], set()
    for g in man["groups"]:
        ps = []
        for p in g["pages"]:
            assert p["id"] not in seen and re.fullmatch(r"[a-z0-9-]+", p["id"]), p["id"]
            seen.add(p["id"])
            nonce = os.urandom(12)
            data = aes.encrypt(nonce, open(os.path.join(base, p["file"]), "rb").read(), None)
            ps.append({"id": p["id"], "title": p["title"], "n": b64(nonce), "c": b64(data)})
        groups.append({"name": g["name"], "pages": ps})
    payload = json.dumps({"title": man["title"], "s": b64(salt), "i": ITER, "groups": groups},
                         ensure_ascii=False).replace("</", "<\\/")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    page = SHELL.replace("__TITLE__", html.escape(man["title"])).replace("__DATA__", payload)
    open(out, "w", encoding="utf-8").write(page)


def unpack(src, outdir):
    password = os.environ["PAGE_PASSWORD"]
    text = open(src, encoding="utf-8").read()
    start = text.index("const D=") + len("const D=")
    d, _ = json.JSONDecoder().raw_decode(text, start)
    aes = AESGCM(derive(password, unb64(d["s"]), d["i"]))
    os.makedirs(outdir, exist_ok=True)
    man = {"title": d["title"], "groups": []}
    for g in d["groups"]:
        mg = {"name": g["name"], "pages": []}
        for p in g["pages"]:
            fn = p["id"] + ".html"
            open(os.path.join(outdir, fn), "wb").write(aes.decrypt(unb64(p["n"]), unb64(p["c"]), None))
            mg["pages"].append({"id": p["id"], "title": p["title"], "file": fn})
        man["groups"].append(mg)
    json.dump(man, open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if not os.environ.get("PAGE_PASSWORD"):
        sys.exit("Set PAGE_PASSWORD")
    if sys.argv[1] == "--decrypt":
        unpack(sys.argv[2], sys.argv[3])
    else:
        build(sys.argv[1], sys.argv[2])
