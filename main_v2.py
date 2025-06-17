import asyncio
import json
import uuid
import time
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any
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
SERVER_ADDRESS = "cancel-theaters-crossing-tournament.trycloudflare.com"
QUEUE_URL = f"https://{SERVER_ADDRESS}/prompt"
VIDEO_URL = f"https://{SERVER_ADDRESS}/view"
UPLOAD_IMAGE_URL = f"https://{SERVER_ADDRESS}/upload/image"
HISTORY_URL = f"https://{SERVER_ADDRESS}/history"
QUEUE_STATUS_URL = f"https://{SERVER_ADDRESS}/queue"
WORKFLOW_PATH = "/workspace/Wan14B_t2v_VACE_WF_OOM.json"
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

class ComfyUIError(Exception):
    """Custom exception for ComfyUI errors."""
    pass

class WorkflowStatus:
    """Workflow execution status constants."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"

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

def check_comfyui_connection() -> bool:
    """Check if ComfyUI server is accessible."""
    try:
        response = requests.get(f"https://{SERVER_ADDRESS}/", timeout=10, headers=HEADERS)
        return response.status_code == 200
    except:
        return False

def get_queue_status() -> Dict[str, Any]:
    """Get current queue status from ComfyUI."""
    try:
        response = requests.get(QUEUE_STATUS_URL, timeout=10, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get queue status: {e}")
        return {}

def get_workflow_history(prompt_id: str) -> Dict[str, Any]:
    """Get workflow execution history for a specific prompt ID."""
    try:
        response = requests.get(f"{HISTORY_URL}/{prompt_id}", timeout=10, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get workflow history for {prompt_id}: {e}")
        return {}

def check_workflow_status(prompt_id: str) -> tuple[str, Optional[str], Optional[Dict]]:
    """
    Check the status of a workflow by prompt ID.
    
    Returns:
        tuple: (status, error_message, execution_details)
        - status: WorkflowStatus enum value
        - error_message: Error message if failed, None otherwise
        - execution_details: Detailed execution info if available
    """
    try:
        # First check if it's still in queue
        queue_status = get_queue_status()
        
        # Check running queue
        if "queue_running" in queue_status:
            for item in queue_status["queue_running"]:
                if len(item) >= 2 and item[1] == prompt_id:
                    return WorkflowStatus.RUNNING, None, {"position": "running"}
        
        # Check pending queue
        if "queue_pending" in queue_status:
            for i, item in enumerate(queue_status["queue_pending"]):
                if len(item) >= 2 and item[1] == prompt_id:
                    return WorkflowStatus.QUEUED, None, {"position": i + 1}
        
        history = get_workflow_history(prompt_id)
    
        if not history:
            return WorkflowStatus.UNKNOWN, "No history found for this workflow", None
        
        # Parse history to check for completion or errors
        if prompt_id in history:
            execution_data = history[prompt_id]
            
            # Check if there are any error messages in the execution
            if "status" in execution_data:
                status_info = execution_data["status"]
                
                # Check status_str first for quick error detection
                if "status_str" in status_info and status_info["status_str"] == "error":
                    # Look for specific error details in messages
                    if "messages" in status_info:
                        messages = status_info["messages"]
                        error_messages = []
                        
                        for msg in messages:
                            if isinstance(msg, list) and len(msg) >= 2:
                                msg_type, msg_content = msg[0], msg[1]
                                
                                # Check for execution_error messages
                                if msg_type == "execution_error" and isinstance(msg_content, dict):
                                    exception_type = msg_content.get("exception_type", "")
                                    exception_message = msg_content.get("exception_message", "")
                                    node_id = msg_content.get("node_id", "")
                                    node_type = msg_content.get("node_type", "")
                                    
                                    if "OutOfMemoryError" in exception_type:
                                        error_messages.append(f"Out of Memory Error in Node {node_id} ({node_type}): {exception_message}")
                                    else:
                                        error_messages.append(f"Execution Error in Node {node_id} ({node_type}): {exception_type} - {exception_message}")
                                
                                # Also check for other error types
                                elif msg_type in ["error", "exception"]:
                                    error_messages.append(str(msg_content))
                        
                        if error_messages:
                            return WorkflowStatus.FAILED, "; ".join(error_messages), execution_data
                        else:
                            return WorkflowStatus.FAILED, "Workflow failed with unknown error", execution_data
                    else:
                        return WorkflowStatus.FAILED, "Workflow marked as error but no error details found", execution_data
                
                # Check for successful completion
                elif "completed" in status_info and status_info["completed"]:
                    return WorkflowStatus.COMPLETED, None, execution_data
                
                # Check if still running (status_str might be 'running' or similar)
                elif "status_str" in status_info and status_info["status_str"] in ["running", "executing"]:
                    return WorkflowStatus.RUNNING, None, execution_data
            
            # If we have execution data but no clear status, might still be running
            return WorkflowStatus.RUNNING, None, execution_data
        
        return WorkflowStatus.UNKNOWN, "Workflow not found in history", None
            
    except Exception as e:
        return WorkflowStatus.UNKNOWN, f"Error checking workflow status: {str(e)}", None

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
    # Check ComfyUI connection first
    if not check_comfyui_connection():
        raise ComfyUIError("ComfyUI server is not accessible. Please ensure ComfyUI is running.")
    
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    
    try:
        response = requests.post(QUEUE_URL, json=payload, timeout=30, headers=HEADERS)
        response.raise_for_status()
        
        result = response.json()
        print("ComfyUI Response:", json.dumps(result, indent=2))
        
        # Check if there's an error in the response
        if "error" in result:
            error_detail = result["error"]
            raise ComfyUIError(f"Workflow validation failed: {error_detail}")
        
        if "node_errors" in result and len(result["node_errors"]) > 0:
            node_errors = result["node_errors"]
            error_details = []
            for node_id, errors in node_errors.items():
                error_details.append(f"Node {node_id}: {errors}")
            raise ComfyUIError(f"Node validation errors: {'; '.join(error_details)}")
        
        result["client_id"] = client_id
        return result
        
    except requests.exceptions.RequestException as e:
        raise ComfyUIError(f"Failed to queue workflow: {str(e)}")

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

async def poll_for_video_with_status_check(prompt_id: str, output_prefix: str, output_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Poll for video completion with comprehensive status checking.
    
    Returns:
        tuple: (filename, error_message)
        - filename: Output filename if successful, None if failed
        - error_message: Error message if failed, None if successful
    """
    print(f"Waiting {INITIAL_WAIT_SECONDS} seconds before polling...")
    await asyncio.sleep(INITIAL_WAIT_SECONDS)
    
    for attempt in range(MAX_POLL_ATTEMPTS):
        print(f"Poll attempt {attempt + 1}/{MAX_POLL_ATTEMPTS} for prompt ID: {prompt_id}")
        
        # Check workflow status first
        status, error_msg, details = check_workflow_status(prompt_id)
        
        print(f"Workflow status: {status}")
        if details and "position" in details:
            print(f"Queue position: {details['position']}")
        
        # Handle different statuses
        if status == WorkflowStatus.FAILED:
            return None, f"Workflow execution failed: {error_msg}"
        
        elif status == WorkflowStatus.COMPLETED:
            print("Workflow completed successfully, checking for output files...")
        
            # Find the latest output file with the given prefix
            latest_filename = find_latest_output_file(output_prefix)
            
            if latest_filename:
                print(f"Found output file: {latest_filename}")
                
                # Try to download it
                if download_video(latest_filename, output_path):
                    # Validate the downloaded file
                    if output_path.exists() and output_path.stat().st_size > 1024:  # File exists and > 1KB
                        return latest_filename, None
                    else:
                        return None, "Downloaded file is invalid or corrupted (likely due to OOM or processing error)"
                else:
                    print("Failed to download output file, continuing to poll...")
            else:
                return None, "Workflow completed but no valid output file was generated (likely failed due to OOM or other error)"
        
        elif status == WorkflowStatus.UNKNOWN:
            print(f"Unknown workflow status: {error_msg}")
            # Continue polling in case it's just a temporary issue
        
        # For QUEUED and RUNNING statuses, continue polling
        
        if attempt < MAX_POLL_ATTEMPTS - 1:
            print(f"Waiting {POLL_INTERVAL_SECONDS} seconds before next check...")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    
    # Final check for files in case status check failed but files were generated
    latest_filename = find_latest_output_file(output_prefix)
    if latest_filename and download_video(latest_filename, output_path):
        return latest_filename, None
    
    max_wait_time = INITIAL_WAIT_SECONDS + (MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS)
    return None, f"Workflow timeout after {max_wait_time} seconds ({max_wait_time//60} minutes). Last status: {status}"

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
    Generate video using WanVideo ComfyUI workflow with comprehensive error detection.
    
    This endpoint processes video through the WanVideo pipeline using node 168 for text prompts.
    It includes robust error detection and status monitoring for the ComfyUI workflow execution.
    
    Args:
        video: Input video file (upload)
        video_url: Video URL (alternative to upload)  
        positive_prompt: Positive prompt for generation (applied to node 168)
        negative_prompt: Negative prompt for generation (applied to node 168)
        output_name: Optional output filename prefix
        upload_to_oss: Whether to upload result to OSS
        
    Returns:
        Success response with output file information or detailed error message
    """
    try:
        # Validate inputs
        if not (video or video_url):
            raise HTTPException(
                status_code=400, 
                detail="Must provide either video file or video_url"
            )
        
        # Check ComfyUI connection
        if not check_comfyui_connection():
            raise HTTPException(
                status_code=503,
                detail="ComfyUI server is not accessible. Please ensure ComfyUI is running with 'python main.py' command."
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
        try:
            upload_file_to_comfyui(video_content, video_filename)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload video to ComfyUI: {str(e)}"
            )
        
        # Load and modify workflow (node 168 for prompts, node 173 for video, node 229 for output)
        try:
            workflow = load_and_modify_workflow(
                video_filename, 
                output_prefix, 
                positive_prompt, 
                negative_prompt
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load/modify workflow: {str(e)}"
            )
        
        # Queue workflow
        print("Queueing workflow...")
        try:
            queue_result = queue_prompt(workflow)
            prompt_id = queue_result.get('prompt_id')
            print(f"Queued successfully: {prompt_id}")
        except ComfyUIError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow validation failed: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue workflow: {str(e)}"
            )
        
        # Wait for completion with status monitoring
        timestamp = int(time.time())
        output_filename = f"{output_prefix}_{timestamp}.mp4"
        output_path = DOWNLOAD_DIR / output_filename
        
        print("Monitoring workflow execution...")
        success_filename, error_message = await poll_for_video_with_status_check(
            prompt_id, output_prefix, output_path
        )
        
        if success_filename and not error_message:
            print(f"Success! Output: {output_path}")
            
            result = {
                "message": "WanVideo generation completed successfully",
                "output_file": str(output_path),
                "download_url": f"/download/{output_filename}",
                "queue_id": prompt_id,
                "client_id": queue_result.get("client_id"),
                "processing_time": f"Completed within {INITIAL_WAIT_SECONDS + MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS} seconds",
                "prompts_used": {
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt
                },
                "workflow_status": "completed"
            }
        else:
            # Workflow failed or timed out
            error_detail = {
                "message": "WanVideo generation failed",
                "error": error_message,
                "queue_id": prompt_id,
                "workflow_status": "failed",
                "prompts_used": {
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt
                }
            }
            
            # Don't include output file info for failed workflows
            raise HTTPException(status_code=500, detail=error_detail)
            
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
        
        # else:
        #     # Workflow failed or timed out
        #     raise HTTPException(
        #         status_code=500,
        #         detail={
        #             "message": "Workflow execution failed",
        #             "error": error_message,
        #             "queue_id": prompt_id,
        #             "workflow_status": "failed",
        #             "prompts_used": {
        #                 "positive_prompt": positive_prompt,
        #                 "negative_prompt": negative_prompt
        #             }
        #         }
        #     )
            
    except HTTPException:
        raise
    except ComfyUIError as e:
        raise HTTPException(status_code=500, detail=f"ComfyUI Error: {str(e)}")
    except Exception as e:
        print(f"Error during WanVideo generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/workflow-status/{prompt_id}")
async def get_workflow_status_endpoint(prompt_id: str):
    """Get the current status of a workflow by prompt ID."""
    try:
        status, error_msg, details = check_workflow_status(prompt_id)
        
        result = {
            "prompt_id": prompt_id,
            "status": status,
            "error_message": error_msg,
            "details": details
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")

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
    """Health check endpoint with enhanced ComfyUI connectivity check."""
    try:
        # Test connection to ComfyUI
        comfyui_accessible = check_comfyui_connection()
        comfyui_status = "healthy" if comfyui_accessible else "unreachable"
        
        # Get queue status if accessible
        queue_info = {}
        if comfyui_accessible:
            try:
                queue_status = get_queue_status()
                queue_info = {
                    "running": len(queue_status.get("queue_running", [])),
                    "pending": len(queue_status.get("queue_pending", []))
                }
            except:
                queue_info = {"error": "Could not get queue status"}
    except:
        comfyui_status = "unreachable"
        queue_info = {"error": "ComfyUI not accessible"}
    
    # Check workflow file
    try:
        validate_workflow_file()
        workflow_status = "valid"
    except:
        workflow_status = "invalid"
    
    # Check OSS connection
    oss_status = "configured" if oss_bucket else "not_configured"
    
    return {
        "status": "healthy" if comfyui_status == "healthy" else "degraded",
        "comfyui_status": comfyui_status,
        "workflow_status": workflow_status,
        "oss_status": oss_status,
        "queue_info": queue_info,
        "server_address": SERVER_ADDRESS,
        "workflow_path": WORKFLOW_PATH,
        "max_processing_time": f"{INITIAL_WAIT_SECONDS + MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS} seconds",
        "output_directory": str(OUTPUT_DIR),
        "download_directory": str(DOWNLOAD_DIR),
        "workflow_nodes": {
            "video_input": "Node 173 (VHS_LoadVideo)",
            "text_prompts": "Node 168 (WanVideoTextEncode)", 
            "video_output": "Node 229 (VHS_VideoCombine)"
        },
        "error_detection": "enabled",
        "status_monitoring": "enabled"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ComfyUI WanVideo API with Enhanced Error Detection",
        "version": "1.0.0",
        "description": "Video generation service using ComfyUI WanVideo workflow with comprehensive error monitoring",
        "endpoints": {
            "wanvideo_generate": "POST /wanvideo-generate",
            "workflow_status": "GET /workflow-status/{prompt_id}",
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
        },
        "error_detection": {
            "connection_check": "Validates ComfyUI server accessibility",
            "workflow_validation": "Checks for node errors and validation issues",
            "status_monitoring": "Monitors workflow execution status in real-time",
            "queue_monitoring": "Tracks position in ComfyUI processing queue",
            "execution_history": "Analyzes workflow execution history for errors",
            "early_termination": "Stops processing immediately upon error detection"
        },
        "troubleshooting": {
            "comfyui_not_running": "Ensure ComfyUI is started with 'python main.py'",
            "workflow_errors": "Check /workflow-status/{prompt_id} for detailed error info",
            "timeout_issues": "Check ComfyUI logs and server resources",
            "connection_problems": "Verify SERVER_ADDRESS and network connectivity"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)