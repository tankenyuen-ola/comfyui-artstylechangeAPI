import asyncio
import json
import uuid
import time
import copy
from pathlib import Path
from typing import Optional, List
import re
import os

import requests
import requests.exceptions
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import uvicorn
from dotenv import load_dotenv

# Aliyun OSS imports
import oss2

load_dotenv()

# Configuration
SERVER_ADDRESS = "collection-aj-baths-vegetables.trycloudflare.com"
QUEUE_URL = f"https://{SERVER_ADDRESS}/prompt"
VIDEO_URL = f"https://{SERVER_ADDRESS}/view"
UPLOAD_IMAGE_URL = f"https://{SERVER_ADDRESS}/upload/image"
WORKFLOW_PATH = "/workspace/Wan14B_t2v_VACE_WF-API.json"
OUTPUT_DIR = Path("/workspace/ComfyUI/output")
DOWNLOAD_DIR = Path("/workspace/downloads")

# Aliyun OSS Configuration
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "your_access_key_id")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "your_access_key_secret")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")  # Change to your region
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "your_bucket_name")
OSS_FOLDER_PREFIX = os.getenv("OSS_FOLDER_PREFIX", "wanvideo/")  # Folder in OSS bucket

# Polling settings for 3-8 minute workflows
POLL_INTERVAL_SECONDS = 15  # Check every 15 seconds
MAX_POLL_ATTEMPTS = 40      # 40 attempts = 10 minutes max
INITIAL_WAIT_SECONDS = 60   # Wait 1 minute before first poll

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(title="ComfyUI WanVideo API", version="1.0.0")

# Headers for requests
HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cookie": "C.20984754_auth_token=af1ad767732eaa664611af8bf29b2b2cd3598dbeca534b4324748d1a029d2adc",
    "Origin": "https://collection-aj-baths-vegetables.trycloudflare.com/",
    "Priority": "u=1, i",
    "Referer": "https://collection-aj-baths-vegetables.trycloudflare.com/",
    "Sec-Ch-Ua": '"Chromium";v="136", "Microsoft Edge";v="136", "Not.A/Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
}

# Initialize Aliyun OSS client
def init_oss_client():
    """Initialize Aliyun OSS client."""
    try:
        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
        return bucket
    except Exception as e:
        print(f"Failed to initialize OSS client: {e}")
        return None

oss_bucket = init_oss_client()

def validate_workflow_file():
    """Validate that the workflow file exists and is valid JSON."""
    if not Path(WORKFLOW_PATH).exists():
        raise FileNotFoundError(f"Workflow file not found: {WORKFLOW_PATH}")
    
    try:
        with open(WORKFLOW_PATH, "r") as f:
            json.load(f)
        print("✓ Workflow file validated")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in workflow file: {e}")

def download_file_from_url(url: str) -> bytes:
    """Download file content from URL."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from URL: {str(e)}")

def upload_file_to_comfyui(file_content: bytes, uploaded_filename: str) -> bool:
    """Upload a file to ComfyUI."""
    payload = {
        "overwrite": "true",
        "type": "input",
        "subfolder": "",
    }
    
    files = [("image", (uploaded_filename, file_content, "application/octet-stream"))]
    response = requests.post(UPLOAD_IMAGE_URL, data=payload, files=files, headers=HEADERS)
    response.raise_for_status()
    return True

def get_available_files() -> List[str]:
    """Get list of available output files from ComfyUI."""
    try:
        # Fallback: scan local output directory if available
        if OUTPUT_DIR.exists():
            return [f.name for f in OUTPUT_DIR.glob("*.mp4")]
        
        return []
    except Exception as e:
        print(f"Warning: Could not get file list: {e}")
        return []

def find_latest_output_file(output_prefix: str) -> Optional[str]:
    """Find the latest output file with the given prefix."""
    available_files = get_available_files()
    
    # Pattern to match files like: WanVideoWrapper_VACE_startendframe_00001.mp4, etc.
    pattern = rf"^{re.escape(output_prefix)}_(\d+)\.mp4$"
    
    matching_files = []
    for filename in available_files:
        match = re.match(pattern, filename)
        if match:
            sequence_num = int(match.group(1))
            matching_files.append((filename, sequence_num))
    
    if not matching_files:
        return None
    
    # Return the file with the highest sequence number
    latest_file = max(matching_files, key=lambda x: x[1])
    return latest_file[0]

def load_and_modify_workflow(video_filename: str, output_prefix: str, 
                           positive_prompt: str = None, negative_prompt: str = None) -> dict:
    """Load and modify the ComfyUI WanVideo workflow."""
    with open(WORKFLOW_PATH, "r") as f:
        workflow_data = json.load(f)
    
    modified_workflow = copy.deepcopy(workflow_data)
    
    # Node 173 - Load Video (UpLoad Control Video)
    if "173" in modified_workflow:
        modified_workflow["173"]["inputs"]["video"] = video_filename
        print(f"✓ Set video input to: {video_filename}")
    else:
        print("⚠ Warning: Node 173 (Video Load) not found in workflow")
    
    # Node 229 - Video Output (Video Combine)
    if "229" in modified_workflow:
        modified_workflow["229"]["inputs"]["filename_prefix"] = output_prefix
        print(f"✓ Set output prefix to: {output_prefix}")
    else:
        print("⚠ Warning: Node 229 (Video Output) not found in workflow")
    
    # Node 168 - WanVideo TextEncode (Text prompts)
    if "168" in modified_workflow:
        if positive_prompt is not None:
            modified_workflow["168"]["inputs"]["positive_prompt"] = positive_prompt
            print(f"✓ Set positive prompt: {positive_prompt[:100]}{'...' if len(positive_prompt) > 100 else ''}")
        
        if negative_prompt is not None:
            modified_workflow["168"]["inputs"]["negative_prompt"] = negative_prompt
            print(f"✓ Set negative prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
    else:
        print("⚠ Warning: Node 168 (WanVideo TextEncode) not found in workflow")
    
    return modified_workflow

def queue_prompt(workflow: dict) -> dict:
    """Queue workflow to ComfyUI."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    
    response = requests.post(QUEUE_URL, json=payload, timeout=30, headers=HEADERS)
    response.raise_for_status()
    
    result = response.json()
    result["client_id"] = client_id
    return result

def download_video(filename: str, output_path: Path) -> bool:
    """Download video from ComfyUI."""
    params = {"filename": filename, "type": "output"}
    
    try:
        response = requests.get(VIDEO_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {output_path}")
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return False  # Still processing
        else:
            raise e

async def poll_for_video(output_prefix: str, output_path: Path) -> Optional[str]:
    """Poll for video completion and download."""
    print(f"Waiting {INITIAL_WAIT_SECONDS} seconds before polling...")
    await asyncio.sleep(INITIAL_WAIT_SECONDS)
    
    for attempt in range(MAX_POLL_ATTEMPTS):
        print(f"Poll attempt {attempt + 1}/{MAX_POLL_ATTEMPTS} for prefix '{output_prefix}'")
        
        # Find the latest output file with the given prefix
        latest_filename = find_latest_output_file(output_prefix)
        
        if latest_filename:
            print(f"Found output file: {latest_filename}")
            
            # Try to download it
            if download_video(latest_filename, output_path):
                return latest_filename
        
        if attempt < MAX_POLL_ATTEMPTS - 1:
            print(f"Not ready, waiting {POLL_INTERVAL_SECONDS} seconds...")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    
    print(f"Timeout after {MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS // 60} minutes")
    return None

def upload_to_oss(file_path: Path, oss_key: str) -> Optional[str]:
    """Upload file to Aliyun OSS."""
    if not oss_bucket:
        print("OSS client not initialized")
        return None
    
    try:
        # Upload file
        oss_bucket.put_object_from_file(oss_key, str(file_path))
        
        # Generate URL (you might want to use signed URL for private buckets)
        oss_url = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{oss_key}"
        print(f"Uploaded to OSS: {oss_url}")
        return oss_url
        
    except Exception as e:
        print(f"Failed to upload to OSS: {e}")
        return None

@app.post("/wanvideo-generate")
async def wanvideo_generate(
    video: Optional[UploadFile] = File(None, description="Input video file"),
    video_url: Optional[str] = Form(None, description="Video URL (alternative to file upload)"),
    positive_prompt: Optional[str] = Form(None, description="Positive prompt for video generation"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt for video generation"),
    output_name: Optional[str] = Form(None, description="Custom output filename prefix"),
    upload_to_oss: bool = Form(False, description="Upload result to Aliyun OSS")
):
    """
    Generate video using WanVideo ComfyUI workflow.
    
    This endpoint processes video through the WanVideo pipeline using node 168 for text prompts.
    
    Args:
        video: Input video file (upload)
        video_url: Video URL (alternative to upload)  
        positive_prompt: Positive prompt for generation (applied to node 168)
        negative_prompt: Negative prompt for generation (applied to node 168)
        output_name: Optional output filename prefix
        upload_to_oss: Whether to upload result to OSS
        
    Returns:
        Success response with output file information
    """
    try:
        # Validate inputs
        if not (video or video_url):
            raise HTTPException(
                status_code=400, 
                detail="Must provide either video file or video_url"
            )
        
        # Validate workflow file
        validate_workflow_file()
        
        # Generate unique filenames to avoid conflicts
        video_suffix = '.mp4'
        
        if video:
            video_suffix = Path(video.filename).suffix or '.mp4'
        
        video_filename = f"input_video_{uuid.uuid4()}{video_suffix}"
        output_prefix = output_name or "WanVideoWrapper_VACE_startendframe"
        
        # Get video content
        if video:
            video_content = await video.read()
            print(f"Using uploaded video file: {video.filename}")
        else:
            print(f"Downloading video from URL: {video_url}")
            video_content = download_file_from_url(video_url)
        
        print(f"Processing WanVideo generation with prompts")
        print(f"Positive prompt: {positive_prompt[:100] + '...' if positive_prompt and len(positive_prompt) > 100 else positive_prompt}")
        print(f"Negative prompt: {negative_prompt[:100] + '...' if negative_prompt and len(negative_prompt) > 100 else negative_prompt}")
        
        # Upload video file
        print("Uploading video...")
        upload_file_to_comfyui(video_content, video_filename)
        
        # Load and modify workflow (node 168 for prompts, node 173 for video, node 229 for output)
        workflow = load_and_modify_workflow(
            video_filename, 
            output_prefix, 
            positive_prompt, 
            negative_prompt
        )
        
        # Queue workflow
        print("Queueing workflow...")
        queue_result = queue_prompt(workflow)
        print(f"Queued: {queue_result.get('prompt_id')}")
        
        # Wait for completion
        timestamp = int(time.time())
        output_filename = f"{output_prefix}_{timestamp}.mp4"
        output_path = DOWNLOAD_DIR / output_filename
        
        print("Waiting for processing...")
        success_filename = await poll_for_video(output_prefix, output_path)
        
        if success_filename:
            print(f"Success! Output: {output_path}")
            
            result = {
                "message": "WanVideo generation completed successfully",
                "output_file": str(output_path),
                "download_url": f"/download/{output_filename}",
                "queue_id": queue_result.get("prompt_id"),
                "client_id": queue_result.get("client_id"),
                "processing_time": f"{INITIAL_WAIT_SECONDS + MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS} seconds max",
                "prompts_used": {
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt
                }
            }
            
            # Upload to OSS if requested
            if upload_to_oss and oss_bucket:
                oss_key = f"{OSS_FOLDER_PREFIX}{output_filename}"
                oss_url = upload_to_oss(output_path, oss_key)
                if oss_url:
                    result["oss_url"] = oss_url
                    result["oss_key"] = oss_key
                else:
                    result["oss_error"] = "Failed to upload to OSS"
            elif upload_to_oss:
                result["oss_error"] = "OSS client not configured"
            
            return result
        else:
            max_wait_time = INITIAL_WAIT_SECONDS + (MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS)
            raise HTTPException(
                status_code=408,
                detail=f"Workflow timeout after {max_wait_time} seconds ({max_wait_time//60} minutes)"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during WanVideo generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-to-oss")
async def upload_file_to_oss(
    filename: str = Form(..., description="Filename in download directory"),
    oss_key: Optional[str] = Form(None, description="Custom OSS key (path in bucket)")
):
    """Upload a file from download directory to Aliyun OSS."""
    try:
        file_path = DOWNLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found in download directory")
        
        if not oss_bucket:
            raise HTTPException(status_code=500, detail="OSS client not configured")
        
        # Use custom key or default with folder prefix
        if not oss_key:
            oss_key = f"{OSS_FOLDER_PREFIX}{filename}"
        
        oss_url = upload_to_oss(file_path, oss_key)
        
        if oss_url:
            return {
                "message": "File uploaded to OSS successfully",
                "filename": filename,
                "oss_url": oss_url,
                "oss_key": oss_key
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to upload to OSS")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading to OSS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed video file."""
    file_path = DOWNLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test connection to ComfyUI
        response = requests.get(f"https://{SERVER_ADDRESS}/", timeout=5, headers=HEADERS)
        comfyui_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        comfyui_status = "unreachable"
    
    # Check workflow file
    try:
        validate_workflow_file()
        workflow_status = "valid"
    except:
        workflow_status = "invalid"
    
    # Check OSS connection
    oss_status = "configured" if oss_bucket else "not_configured"
    
    return {
        "status": "healthy",
        "comfyui_status": comfyui_status,
        "workflow_status": workflow_status,
        "oss_status": oss_status,
        "server_address": SERVER_ADDRESS,
        "workflow_path": WORKFLOW_PATH,
        "max_processing_time": f"{INITIAL_WAIT_SECONDS + MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS} seconds",
        "output_directory": str(OUTPUT_DIR),
        "download_directory": str(DOWNLOAD_DIR),
        "workflow_nodes": {
            "video_input": "Node 173 (VHS_LoadVideo)",
            "text_prompts": "Node 168 (WanVideoTextEncode)", 
            "video_output": "Node 229 (VHS_VideoCombine)"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ComfyUI WanVideo API",
        "version": "1.0.0",
        "description": "Video generation service using ComfyUI WanVideo workflow",
        "endpoints": {
            "wanvideo_generate": "POST /wanvideo-generate",
            "upload_to_oss": "POST /upload-to-oss",
            "download": "GET /download/{filename}",
            "health": "GET /health"
        },
        "usage": {
            "file_upload": "Use 'video' form field",
            "url_input": "Use 'video_url' form field",
            "prompts": "Use 'positive_prompt' and 'negative_prompt' for generation (applied to Node 168)",
            "output_name": "Optional custom output filename prefix",
            "oss_upload": "Set 'upload_to_oss=true' to upload result to Aliyun OSS"
        },
        "workflow_info": {
            "video_input_node": "173 (VHS_LoadVideo)",
            "text_encode_node": "168 (WanVideoTextEncode)",
            "output_node": "229 (VHS_VideoCombine)",
            "default_positive_prompt": "Photorealistic, Cinematic, Gritty Historical Action...",
            "default_negative_prompt": "bad quality, blurry, messy, chaotic"
        },
        "settings": {
            "initial_wait": f"{INITIAL_WAIT_SECONDS} seconds",
            "poll_interval": f"{POLL_INTERVAL_SECONDS} seconds",
            "max_attempts": MAX_POLL_ATTEMPTS,
            "timeout": f"{(INITIAL_WAIT_SECONDS + MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS) // 60} minutes"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)