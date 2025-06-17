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
from typing import Any, Dict, AsyncGenerator, Optional

import requests
import websockets
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

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

async def monitor_progress_unified(
    prompt_id: str, 
    client_id: str, 
    node_titles: Dict[int, str]
):
    """
    Monitor function that yields SSE events for client streaming.
    """
    ws_url = f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}"
    last_progress = -1.0
    last_status: Optional[str] = None
    idle = 0
    idle_limit = 60
    workflow_started = False
    workflow_completed = False

    try:
        async with websockets.connect(ws_url, additional_headers=HEADERS, ping_interval=None, max_size=None) as ws:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    idle = 0
                except asyncio.TimeoutError:
                    idle += 1
                    if idle > idle_limit:
                        error_msg = "WebSocket idle timeout exceeded"
                        print(f"[WS] ERROR: {error_msg}")
                        yield {
                            "event": "error",
                            "data": json.dumps({"detail": error_msg})
                        }
                        return
                    continue

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
                    workflow_started = True
                    v, m = data.get("value", 0), data.get("max", 1)
                    if m:
                        pct = round(v / m * 100, 2)
                        if pct != last_progress:
                            print(f"[WS] Progress: {pct}%")
                            yield {
                                "event": "progress",
                                "data": json.dumps({
                                    "percentage": pct,
                                    "current": v,
                                    "total": m,
                                    "message": f"Progress: {pct}%"
                                })
                            }
                            last_progress = pct
                            
                elif mtype == "executing":
                    node_label = data.get("node_label") or data.get("node")
                    if node_label is None:
                        # Workflow is idle - if it was started before, it means completion
                        if workflow_started and not workflow_completed:
                            workflow_completed = True
                            print(f"[WS] Terminal status reached: completed. Closing connection.")
                            yield {
                                "event": "workflow_status",
                                "data": json.dumps({
                                    "final_status": "completed",
                                    "message": "Workflow completed successfully. Connection closing."
                                })
                            }
                            return
                        else:
                            print("[WS] Executing: Idle (no active node)")
                            yield {
                                "event": "executing",
                                "data": json.dumps({
                                    "node": "idle",
                                    "message": "Workflow idle (no active node)"
                                })
                            }
                    else:
                        workflow_started = True
                        try:
                            node_id = int(node_label)
                            title = node_titles.get(node_id, "unknown")
                            print(f"[WS] Executing: Node {node_label} ({title})")
                            yield {
                                "event": "executing",
                                "data": json.dumps({
                                    "node": node_label,
                                    "title": title,
                                    "message": f"Executing Node {node_label}: {title}"
                                })
                            }
                        except (ValueError, TypeError):
                            print(f"[WS] Executing: Node {node_label} (invalid node ID)")
                            yield {
                                "event": "executing",
                                "data": json.dumps({
                                    "node": str(node_label),
                                    "message": f"Executing Node {node_label} (invalid ID)"
                                })
                            }
                            
                elif mtype == "status":
                    raw_status = data.get("status")
                    status = (
                        raw_status.get("status")
                        if isinstance(raw_status, dict)
                        else raw_status
                    )
                    if status != last_status:
                        print(f"[WS] Status: {status}")
                        yield {
                            "event": "status_update",
                            "data": json.dumps({"status": status})
                        }
                        last_status = status
                        
                    # Handle explicit terminal states (error, failed, cancelled)
                    if status in {"error", "failed", "cancelled"}:
                        workflow_completed = True
                        print(f"[WS] Terminal status reached: {status}. Closing connection.")
                        yield {
                            "event": "workflow_status",
                            "data": json.dumps({
                                "final_status": status,
                                "message": f"Workflow {status}. Connection closing."
                            })
                        }
                        return
                        
    except Exception as e:
        error_msg = f"WebSocket connection error: {str(e)}"
        print(f"[WS] ERROR: {error_msg}")
        yield {
            "event": "error", 
            "data": json.dumps({"detail": error_msg})
        }


def fetch_output_file(prompt_id: str) -> Path:
    """Locate and download the generated video to DOWNLOAD_DIR."""
    try:
        history_response = requests.get(f"{HISTORY_URL}/{prompt_id}", headers=HEADERS, timeout=30).json()
    except Exception as e:
        raise ComfyUIError(f"Fetching /history failed: {e}") from e

    # The history response structure is: {prompt_id: {outputs: {...}, status: {...}, meta: {...}}}
    # Get the first (and likely only) prompt entry
    if not history_response:
        raise ComfyUIError("Empty history response")
    
    # Get the prompt data (should be the prompt_id key)
    prompt_data = history_response.get(prompt_id)
    if not prompt_data:
        # Fallback: try to get the first entry if prompt_id key doesn't exist
        prompt_data = next(iter(history_response.values()), None)
    
    if not prompt_data:
        raise ComfyUIError("No prompt data found in history response")
    
    outputs = prompt_data.get("outputs", {})
    if not outputs:
        raise ComfyUIError("No outputs present in history response")

    # Look for the video output - typically in node 229 based on your workflow
    video_output = None
    
    # First try node 229 (VHS_VideoCombine)
    if "229" in outputs and outputs["229"].get("gifs"):
        video_output = outputs["229"]["gifs"][0]
    else:
        # Fallback: search all outputs for video files
        for node_id, node_outputs in outputs.items():
            for output_type, output_list in node_outputs.items():
                if output_list and isinstance(output_list, list):
                    for item in output_list:
                        if isinstance(item, dict) and item.get("format", "").startswith("video/"):
                            video_output = item
                            break
                    if video_output:
                        break
            if video_output:
                break
    
    if not video_output:
        raise ComfyUIError("No video output found in history response")

    filename = video_output.get("filename")
    subfolder = video_output.get("subfolder", "")

    if not filename:
        raise ComfyUIError("Filename missing in video output")

    # Construct the view URL
    params = {"filename": filename}
    if subfolder:
        params["subfolder"] = subfolder
    
    # Build URL with parameters
    url_params = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{VIEW_URL}?{url_params}"
    
    target = DOWNLOAD_DIR / filename

    try:
        with requests.get(url, stream=True, headers=HEADERS, timeout=60) as r:
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        raise ComfyUIError(f"Failed to download video file: {e}") from e
        
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
async def wanvideo_generate_sse(
    video: Optional[UploadFile] = File(None, description="Input video file (.mp4, .mov, etc.)"),
    video_url: Optional[str] = Form(None, description="URL to a video if not uploading"),
    positive_prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    output_name: Optional[str] = Form(None),
):
    """Generate a video with real-time progress via Server-Sent Events."""
    
    # Pre-process the video file BEFORE creating the async generator
    if not video and not video_url:
        return EventSourceResponse(
            lambda: async_error_generator("Provide either `video` or `video_url`.")
        )
    
    # Handle video file upload/download outside the generator
    ts = time.strftime("%Y%m%d_%H%M%S")
    video_path = None
    
    try:
        if video:
            # Read the video content immediately while the file is still open
            video_content = await video.read()
            in_fname = f"input_{ts}_{video.filename}"
            video_path = OUTPUT_DIR / in_fname
            with video_path.open("wb") as f:
                f.write(video_content)
        else:
            # Download from URL
            resp = requests.get(video_url, stream=True, timeout=60, headers=HEADERS)
            resp.raise_for_status()
            in_fname = f"input_{ts}.mp4"
            video_path = OUTPUT_DIR / in_fname
            with video_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        return EventSourceResponse(
            lambda: async_error_generator(f"Video processing failed: {str(e)}")
        )
    
    async def generate_progress_stream():
        try:
            # Setup phase
            yield {"event": "status", "data": json.dumps({"message": "Setting up video processing..."})}
            
            # Workflow preparation
            yield {"event": "status", "data": json.dumps({"message": "Preparing ComfyUI workflow..."})}
            
            out_prefix = output_name or f"wanvideo_{ts}"
            workflow = load_and_patch_workflow(
                str(video_path.resolve()), out_prefix, positive_prompt, negative_prompt
            )
            node_titles = build_node_title_map(workflow)

            # Queue workflow
            yield {"event": "status", "data": json.dumps({"message": "Queueing workflow..."})}
            
            client_id = str(uuid.uuid4())
            try:
                prompt_id = queue_prompt(workflow, client_id)
                yield {
                    "event": "queued",
                    "data": json.dumps({"prompt_id": prompt_id, "message": "Workflow queued successfully!"})
                }
            except ComfyUIError as e:
                yield {"event": "error", "data": json.dumps({"detail": str(e)})}
                return

            # Stream progress updates
            final_status = None
            async for progress_event in monitor_progress_unified(prompt_id, client_id, node_titles):
                if progress_event.get("event") == "workflow_status":
                    event_data = json.loads(progress_event["data"])
                    final_status = event_data.get("final_status")
                    break

            # Handle completion
            if final_status in {"completed", "finished"}:
                yield {"event": "status", "data": json.dumps({"message": "Retrieving generated video..."})}
                try:
                    out_file = fetch_output_file(prompt_id)
                    yield {
                        "event": "completed",
                        "data": json.dumps({
                            "message": "🎉 Video generation completed successfully!",
                            "filename": out_file.name,
                            "download_url": f"/download/{out_file.name}",
                            "prompt_id": prompt_id
                        })
                    }
                except ComfyUIError as e:
                    yield {"event": "error", "data": json.dumps({"detail": str(e)})}
            else:
                yield {
                    "event": "error",
                    "data": json.dumps({"detail": f"Workflow failed with status: {final_status}"})
                }

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"detail": f"Unexpected error: {str(e)}"})}

    return EventSourceResponse(generate_progress_stream())


# Helper function for error responses
async def async_error_generator(error_message: str):
    """Helper to generate a single error event."""
    yield {"event": "error", "data": json.dumps({"detail": error_message})}


# -----------------------------------------------------------------------------
# Script entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"▶ Starting FastAPI on http://0.0.0.0:8000 (server → {SERVER_ADDRESS})")
    uvicorn.run("comfyui-websocket:app", host="0.0.0.0", port=8000, reload=False)
