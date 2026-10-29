import os
import json
import urllib.request
import urllib.error
import time

TOKEN = os.getenv("VERCEL_TOKEN")
if not TOKEN:
    # Fallback to local config or input prompt
    TOKEN = input("Enter Vercel Token (or set VERCEL_TOKEN env var): ").strip()
TEAM_ID = "team_orarZagNMSnHXkij8attCXXC"
PROJECT_ID = "prj_oUhXPzqth18be3bYXSM2w12u9fey"
PROJECT_NAME = "object-detection"

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

def deploy():
    files = []
    for fname in ["index.html", "style.css", "app.js", "vercel.json"]:
        fpath = os.path.join(WEB_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            files.append({
                "file": fname,
                "data": content
            })
            print(f"Loaded {fname} ({len(content)} bytes)")

    payload = {
        "name": PROJECT_NAME,
        "project": PROJECT_ID,
        "target": "production",
        "projectSettings": {
            "framework": None
        },
        "files": files
    }

    url = f"https://api.vercel.com/v13/deployments?teamId={TEAM_ID}&skipAutoDetectionConfirmation=1"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json"
        }
    )

    print("Deploying to Vercel...")
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode())
            dep_id = res.get("id")
            dep_url = res.get("url")
            ready_state = res.get("readyState")
            print(f"Deployment initiated: {dep_id}")
            print(f"Preview URL: https://{dep_url}")
            print(f"Initial State: {ready_state}")
    except urllib.error.HTTPError as e:
        print("Deploy error:", e.code, e.read().decode())
        return

    # Poll status
    status_url = f"https://api.vercel.com/v13/deployments/{dep_id}?teamId={TEAM_ID}"
    status_req = urllib.request.Request(status_url, headers={"Authorization": f"Bearer {TOKEN}"})
    
    for _ in range(15):
        time.sleep(2)
        try:
            with urllib.request.urlopen(status_req) as resp:
                status_res = json.loads(resp.read().decode())
                state = status_res.get("readyState")
                print(f"Checking status... {state}")
                if state in ["READY", "ERROR", "CANCELED"]:
                    break
        except Exception as err:
            print("Status poll error:", err)

    print("\nProduction Domains Configured:")
    print("  -> https://object-detection-web.vercel.app")
    print("  -> https://object-detection-vision.vercel.app")
    print("  -> https://object-detection-brown-five.vercel.app")
    print(f"  -> https://{dep_url}")

if __name__ == "__main__":
    deploy()
