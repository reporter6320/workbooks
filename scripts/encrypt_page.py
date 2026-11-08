#!/usr/bin/env python3
"""Encrypt or decrypt a self-contained HTML page for GitHub Pages.

Encrypt: PAGE_PASSWORD=... python3 scripts/encrypt_page.py source.html output.html "Title"
Decrypt: PAGE_PASSWORD=... python3 scripts/encrypt_page.py --decrypt page.html source.html

Uses AES-256-GCM with a PBKDF2-SHA256 key. Never commit the source or password.
"""
import base64, html, os, re, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITER = 600000
b64 = lambda b: base64.b64encode(b).decode()


def key(password, salt):
    return PBKDF2HMAC(SHA256(), 32, salt, ITER).derive(password.encode())


LOADER = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>__TITLE__</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#11141a;color:#e9ecf1;font:16px system-ui,sans-serif}form{width:min(340px,88vw);text-align:center}h1{font-size:22px;margin:0 0 6px}p{color:#a5aebc;margin:0 0 18px;font-size:14px}input,button{box-sizing:border-box;width:100%;padding:11px;border-radius:8px;font:inherit}input{border:1px solid #3a4252;background:#1a1f28;color:inherit;margin-bottom:8px}button{border:0;background:#0c6f79;color:#fff;font-weight:600;cursor:pointer}#e{color:#e3a862;min-height:22px;margin-top:10px;font-size:14px}</style></head>
<body><form id="f"><h1>Encrypted page</h1><p>Enter the password to decrypt.</p><input id="p" type="password" autocomplete="current-password" autofocus aria-label="Password"><button>Unlock</button><div id="e" role="status"></div></form>
<script>
const D=__DATA__,u=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
document.getElementById('f').addEventListener('submit',async ev=>{ev.preventDefault();const e=document.getElementById('e');e.textContent='Decrypting...';try{
const m=await crypto.subtle.importKey('raw',new TextEncoder().encode(document.getElementById('p').value),'PBKDF2',false,['deriveKey']);
const k=await crypto.subtle.deriveKey({name:'PBKDF2',hash:'SHA-256',salt:u(D.s),iterations:D.i},m,{name:'AES-GCM',length:256},false,['decrypt']);
const t=new TextDecoder().decode(await crypto.subtle.decrypt({name:'AES-GCM',iv:u(D.n)},k,u(D.c)));
document.open();document.write(t);document.close();}catch(x){e.textContent='Wrong password.';}});
</script></body></html>
"""


def main():
    password = os.environ.get("PAGE_PASSWORD")
    if not password:
        sys.exit("Set PAGE_PASSWORD")
    if sys.argv[1] == "--decrypt":
        text = open(sys.argv[2], encoding="utf-8").read()
        m = re.search(r'const D=\{"s":"([^"]+)","n":"([^"]+)","c":"([^"]+)","i":(\d+)\}', text)
        salt, nonce, data = (base64.b64decode(x) for x in m.groups()[:3])
        k = PBKDF2HMAC(SHA256(), 32, salt, int(m.group(4))).derive(password.encode())
        open(sys.argv[3], "wb").write(AESGCM(k).decrypt(nonce, data, None))
        return
    src, out, title = sys.argv[1:4]
    salt, nonce = os.urandom(16), os.urandom(12)
    data = AESGCM(key(password, salt)).encrypt(nonce, open(src, "rb").read(), None)
    payload = '{"s":"%s","n":"%s","c":"%s","i":%d}' % (b64(salt), b64(nonce), b64(data), ITER)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(LOADER.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload))


if __name__ == "__main__":
    main()
