import json, os, urllib.request

REPO = "Micro-Novelty/abstract-modules"
PACKAGE = "abstractintegratedmodule"
OUT_DIR = f"gh-pages/whl/{PACKAGE}"

# Fetch all releases from GitHub API
url = f"https://api.github.com/repos/{REPO}/releases"
req = urllib.request.Request(url, headers={
    "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
    "Accept": "application/vnd.github+json"
})
releases = json.loads(urllib.request.urlopen(req).read())

# Collect all .whl assets
links = []
for release in releases:
    for asset in release["assets"]:
        if asset["name"].endswith(".whl"):
            links.append(
                f'<a href="{asset["browser_download_url"]}">{asset["name"]}</a>'
            )

# Write package index
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/index.html", "w") as f:
    f.write(f"""<!DOCTYPE html>
<html>
  <head><title>Links for {PACKAGE}</title></head>
  <body>
    <h1>Links for {PACKAGE}</h1>
    {chr(10).join(links)}
  </body>
</html>""")

# Write root index
os.makedirs("gh-pages/whl", exist_ok=True)
with open("gh-pages/whl/index.html", "w") as f:
    f.write(f"""<!DOCTYPE html>
<html>
  <head><title>Simple Index</title></head>
  <body>
    <a href="{PACKAGE}/">{PACKAGE}</a>
  </body>
</html>""")

# Disable Jekyll so GitHub Pages serves files as-is
with open("gh-pages/.nojekyll", "w") as f:
    f.write("")

print(f"Generated index with {len(links)} wheels.")
