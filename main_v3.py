"""
WanVideo ComfyUI API (WebSocket version)
---------------------------------------
A **single‑file** FastAPI micro‑service that drives the WanVideo ComfyUI workflow
with **real‑time WebSocket progress** instead of HTTP polling.

**Key features**
1. **Node patching preserved** – updates video (node 173), prompts (node 168) and
   output prefix (node 229).
2. **WebSocket monitor** – prints `[WS] Progress`, `[WS] Executing`, and
   `[WS] Status` lines live in your console.
3. **Graceful error handling** – automatic timeout if the socket stalls; returns
   a clear 500 with details.
4. **Result download** – grabs the final video via `/history` → `/view` and
   streams it back as the HTTP response.

```bash
# install deps
pip install fastapi uvicorn websockets requests python-dotenv

# run locally
python main_ws.py  # or  uvicorn main_ws:app --reload
```

Environment variables (all optional):
```
COMFY_SERVER_ADDRESS=127.0.0.1:8188  # host:port where ComfyUI HTTP & WS live
WORKFLOW_PATH=wanvideo_workflow.json # path to saved workflow
OUTPUT_DIR=outputs                   # where uploaded vids & prefixes go
DOWNLOAD_DIR=downloads               # where generated vids are stored
COMFY_COOKIE=auth=...                # cookie if UI is behind auth
PORT=8000                            # FastAPI port
```
"""

import asyncio
import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import websockets
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

# SERVER_ADDRESS = os.getenv("COMFY_SERVER_ADDRESS", "127.0.0.1:8188")
# WORKFLOW_PATH = Path(os.getenv("WORKFLOW_PATH", "wanvideo_workflow.json"))
# OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
# DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "downloads"))

SERVER_ADDRESS = "cancel-theaters-crossing-tournament.trycloudflare.com"
WORKFLOW_PATH = Path("/workspace/Wan14B_t2v_VACE_WF-API.json")
OUTPUT_DIR = Path("/workspace/ComfyUI/output")
DOWNLOAD_DIR = Path("/workspace/downloads")

PORT = 8000

QUEUE_URL = f"http://{SERVER_ADDRESS}/prompt"
HISTORY_URL = f"http://{SERVER_ADDRESS}/history"
VIEW_URL = f"http://{SERVER_ADDRESS}/view"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Headers for requests
HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cookie": "C.20984754_auth_token=af1ad767732eaa664611af8bf29b2b2cd3598dbeca534b4324748d1a029d2adc",
    "Origin": "https://cancel-theaters-crossing-tournament.trycloudflare.com/",
    "Priority": "u=1, i",
    "Referer": "https://cancel-theaters-crossing-tournament.trycloudflare.com/",
    "Sec-Ch-Ua": '"Chromium";v="136", "Microsoft Edge";v="136", "Not.A/Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
}

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class ComfyUIError(RuntimeError):
    """Raised for any ComfyUI‑related failure."""


# -----------------------------------------------------------------------------
# Workflow helpers
# -----------------------------------------------------------------------------

def load_and_patch_workflow(
    video_path: str,
    output_prefix: str,
    positive_prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Load saved workflow JSON and patch three key nodes."""

    if not WORKFLOW_PATH.exists():
        raise ComfyUIError(f"Workflow file not found: {WORKFLOW_PATH}")

    with WORKFLOW_PATH.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    wf = copy.deepcopy(wf)  # safety – do not mutate original

    # 1️⃣ Video path (VHS_LoadVideo – node 173)
    try:
        wf["173"]["inputs"]["video"] = video_path
    except Exception:
        print("⚠ Node 173 missing – skipping video patch")

    # 2️⃣ Output filename prefix (VHS_VideoCombine – node 229)
    try:
        wf["229"]["inputs"]["filename_prefix"] = output_prefix
    except Exception:
        print("⚠ Node 229 missing – skipping prefix patch")

    # 3️⃣ Prompts (WanVideoTextEncode – node 168)
    try:
        if positive_prompt is not None:
            wf["168"]["inputs"]["positive_prompt"] = positive_prompt
        if negative_prompt is not None:
            wf["168"]["inputs"]["negative_prompt"] = negative_prompt
    except Exception:
        print("⚠ Node 168 missing – skipping prompt patch")

    return wf

def build_node_title_map(workflow: Dict[str, Any]) -> Dict[int, str]:
    """Return {node_id: 'Human-readable title'} extracted from _meta.title."""
    return {
        int(nid): node.get("_meta", {}).get("title", "")
        for nid, node in workflow.items()
    }


def queue_prompt(workflow: Dict[str, Any], client_id: str) -> str:
    """POST to /prompt and get back the prompt_id."""
    payload = {"prompt": workflow, "client_id": client_id}
    try:
        resp = requests.post(QUEUE_URL, json=payload, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ComfyUIError(f"Queueing prompt failed: {e}") from e

    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise ComfyUIError("/prompt did not return a prompt_id")
    return prompt_id


async def monitor_progress_ws(prompt_id: str, client_id: str, node_titles: Dict[int, str], idle_limit: int = 60) -> str:
    """Listen to WebSocket, print logs, and return the final status."""
    # build map of node titles
    # print(workflow.items())
    # node_titles = {nid: data.get("_meta", {}).get("title", "") for nid, data in workflow.items()}
    ws_url = f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}"
    last_progress = -1.0
    last_status: Optional[str] = None
    idle = 0

    async with websockets.connect(ws_url, additional_headers=HEADERS, ping_interval=None, max_size=None) as ws:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                idle = 0  # reset idle timer
            except asyncio.TimeoutError:
                idle += 1
                if idle > idle_limit:
                    raise ComfyUIError("WebSocket idle timeout exceeded")
                continue

            # parse message
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")
            data = msg.get("data", {})

            # filter for our prompt
            if data.get("prompt_id") not in (None, prompt_id):
                continue

            if mtype == "progress":
                v, m = data.get("value", 0), data.get("max", 1)
                if m:
                    pct = round(v / m * 100, 2)
                    if pct != last_progress:
                        print(f"[WS] Progress: {pct}%")
                        last_progress = pct
            elif mtype == "executing":
                node_label = data.get("node_label") or data.get("node")
                if node_label is None:
                    # Workflow is idle/finished - no active node
                    print(f"[WS] Executing: Idle (no active node)")
                else:
                    try:
                        node_id = int(node_label)
                        title = node_titles.get(node_id, "unknown")
                        print(f"[WS] Executing: Node {node_label} ({title})")
                    except (ValueError, TypeError):
                        print(f"[WS] Executing: Node {node_label} (invalid node ID)")
            elif mtype == "status":
                # coerce nested dict → string if necessary
                raw_status = data.get("status")
                status = (
                    raw_status.get("status")        # if raw is dict, pull inner
                    if isinstance(raw_status, dict)
                    else raw_status                  # else assume it’s already a str
                )
                if status != last_status:
                    print(f"[WS] Status: {status}")
                    last_status = status
                if status in {"completed", "finished", "error", "failed", "cancelled"}:
                    return status or "unknown"


def fetch_output_file(prompt_id: str) -> Path:
    """Locate and download the generated video to DOWNLOAD_DIR."""
    try:
        history = requests.get(f"{HISTORY_URL}/{prompt_id}", headers=HEADERS, timeout=30).json()
    except Exception as e:
        raise ComfyUIError(f"Fetching /history failed: {e}") from e

    outputs = history.get("outputs", {})
    if not outputs:
        raise ComfyUIError("No outputs present in history response")

    # First node, first output
    first_node_out = next(iter(outputs.values()))[0]
    filename = first_node_out.get("filename")
    subfolder = first_node_out.get("subfolder", "")

    if not filename:
        raise ComfyUIError("Filename missing in history output")

    url = f"{VIEW_URL}?filename={filename}&subfolder={subfolder}"
    target = DOWNLOAD_DIR / filename

    with requests.get(url, stream=True, headers=HEADERS) as r:
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return target

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="ComfyUI WanVideo WebSocket API", version="2.0.0")

@app.websocket("/wanvideo-ws")
async def ws_start(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_text()
        params = json.loads(msg)
        inp = params.get("video_url") or params.get("video_path")
        pos, neg, out = params.get("positive_prompt"), params.get("negative_prompt"), params.get("output_name")
        ts = time.strftime("%Y%m%d_%H%M%S"); prefix=out or f"wan_{ts}"
        if inp.startswith("http"): r=requests.get(inp,headers=HEADERS,stream=True,timeout=60); r.raise_for_status(); fv=OUTPUT_DIR/f"in_{ts}.mp4"; fv.write_bytes(r.content)
        else: fv=Path(inp)
        wf=load_and_patch(BASE_WF_JSON,str(fv),prefix,pos,neg); cid=str(uuid.uuid4()); pid=queue_prompt(wf,cid)
        await ws.send_text(json.dumps({"type":"queued","prompt_id":pid}))
        status=await proxy_comfy_ws(pid,cid,ws,NODE_TITLES)
        if status in {"completed","finished"}:
            outf=fetch_output_file(pid)
            await ws.send_bytes(outf.read_bytes())
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type":"error","detail":str(e)}))
    finally:
        await ws.close()


@app.post("/wanvideo-generate")
async def wanvideo_generate_ws(
    video: Optional[UploadFile] = File(None, description="Input video file (.mp4, .mov, etc.)"),
    video_url: Optional[str] = Form(None, description="URL to a video if not uploading"),
    positive_prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    output_name: Optional[str] = Form(None),
):
    """Generate a video and stream real‑time progress logs."""

    if not video and not video_url:
        raise HTTPException(status_code=400, detail="Provide either `video` or `video_url`.")

    # 1) Prepare input video path
    ts = time.strftime("%Y%m%d_%H%M%S")
    if video:
        in_fname = f"input_{ts}_{video.filename}"
        video_path = OUTPUT_DIR / in_fname
        with video_path.open("wb") as f:
            f.write(await video.read())
    else:
        # download from URL
        try:
            resp = requests.get(video_url, stream=True, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Downloading video failed: {e}")
        in_fname = f"input_{ts}.mp4"
        video_path = OUTPUT_DIR / in_fname
        with video_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    # 2) Patch workflow
    out_prefix = output_name or f"wanvideo_{ts}"
    workflow = load_and_patch_workflow(
        str(video_path.resolve()),
        out_prefix,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
    )
    node_titles = build_node_title_map(workflow)

    # 3) Queue prompt
    client_id = str(uuid.uuid4())
    try:
        prompt_id = queue_prompt(workflow, client_id)
    except ComfyUIError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 4) Monitor websocket
    try:
        status = await monitor_progress_ws(prompt_id, client_id, node_titles)
        print(f"▶ Workflow {prompt_id} finished with status: {status}")
    except ComfyUIError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if status not in {"completed", "finished"}:
        return JSONResponse(status_code=500, content={"status": status, "detail": "Workflow did not finish successfully."})

    # 5) Fetch output
    try:
        out_file = fetch_output_file(prompt_id)
    except ComfyUIError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(out_file, filename=out_file.name, media_type="video/mp4")


# -----------------------------------------------------------------------------
# Script entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"▶ Starting FastAPI on http://0.0.0.0:8000 (server → {SERVER_ADDRESS})")
    uvicorn.run("comfyui-websocket:app", host="0.0.0.0", port=8000, reload=False)
