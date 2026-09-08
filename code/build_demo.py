"""Inline the demo into one self-contained file: demo/dashboard.html.

The served version (demo/index.html) fetches data.json, which browsers block
over file://. This build embeds the CSS, the JSON and every script in load
order so the result opens by double-click, offline, with no server -- which is
what you want on a projector in a room with no wifi.
"""
import json
import os
import re

DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo")
SRC = os.path.join(DEMO, "index.html")
OUT = os.path.join(DEMO, "dashboard.html")


def read(*parts):
    with open(os.path.join(DEMO, *parts), encoding="utf-8") as f:
        return f.read()


def main():
    html = read("index.html")

    # 1. stylesheet -> <style>
    css = read("styles.css")
    html = re.sub(r'<link rel="stylesheet" href="styles\.css">',
                  "<style>\n" + css + "\n</style>", html, count=1)

    # 2. collect the scripts in the order index.html loads them
    srcs = re.findall(r'<script src="([^"]+)"></script>', html)
    if not srcs:
        raise SystemExit("[build_demo] no <script src> tags found in index.html")
    missing = [s for s in srcs if not os.path.exists(os.path.join(DEMO, *s.split("/")))]
    if missing:
        raise SystemExit(f"[build_demo] missing script(s): {missing}")

    # 3. data.json, embedded ahead of main.js so its boot() takes the fast path
    with open(os.path.join(DEMO, "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    payload = json.dumps(data, separators=(",", ":"))
    # </script> inside a JSON string would end the block early
    payload = payload.replace("</", "<\\/")

    bundle = ["<script>window.AT=window.AT||{};window.AT.DATA_INLINE=" + payload + ";</script>"]
    for s in srcs:
        body = read(*s.split("/"))
        bundle.append("<script>\n/* ---- " + s + " ---- */\n" + body + "\n</script>")

    # 4. replace the whole script block with the bundle
    first = html.index('<script src="' + srcs[0] + '"></script>')
    last = html.index('<script src="' + srcs[-1] + '"></script>') + len('<script src="' + srcs[-1] + '"></script>')
    html = html[:first] + "\n".join(bundle) + html[last:]

    html = html.replace("<title>", "<!-- self-contained build: no network, no server needed -->\n<title>", 1)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)

    size = os.path.getsize(OUT)
    print(f"[build_demo] wrote {os.path.abspath(OUT)}  ({size/1024:.0f} KB)")
    print(f"[build_demo] inlined {len(srcs)} scripts + {len(payload)/1024:.0f} KB of data + {len(css)/1024:.0f} KB css")
    for s in srcs:
        print("             ", s)

    # Artifact-shaped variant: the host supplies <!doctype>/<html>/<head>/<body>,
    # so ship the page content only, title and styles first.
    art = html
    art = art[art.index("<title>"):]
    art = art.replace("</head>", "", 1)
    art = re.sub(r"<body>|</body>|</html>", "", art)
    art = art.replace('<meta charset="utf-8">', "").replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1">', "")
    art_path = os.path.join(DEMO, "artifact.html")
    with open(art_path, "w", encoding="utf-8") as f:
        f.write(art.strip() + "\n")
    print(f"[build_demo] wrote {os.path.abspath(art_path)}  ({os.path.getsize(art_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
