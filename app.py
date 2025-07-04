import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from typing import List
import os
from pathlib import Path
import shutil
import tempfile
import logging
import json
import uuid
from contextlib import asynccontextmanager
import requests
import gdown  # For Google Drive downloads
import zipfile

# Import the detector class (assumed to be in the same directory)
from improved_yolo_similarity_detector import ImprovedYOLOSimilarityDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths to YOLO model files (ensure these are on the server)
YOLO_FILES_DIR = Path("yolo_files")
YOLO_WEIGHTS = str(YOLO_FILES_DIR / "yolov4.weights")
YOLO_CONFIG = str(YOLO_FILES_DIR / "yolov4.cfg")
YOLO_NAMES = str(YOLO_FILES_DIR / "coco.names")

# YOLO download URLs
YOLO_URLS = {
    "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
    "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
}

# Temporary directory for uploads
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Base directory for output
OUTPUT_BASE_DIR = Path("similar_matches")
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

def download_file(url: str, filepath: str) -> bool:
    """Download a file from URL to filepath"""
    try:
        logger.info(f"Downloading {os.path.basename(filepath)} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Downloaded {downloaded // (1024*1024)}MB/{total_size // (1024*1024)}MB ({percent:.1f}%)")
        
        logger.info(f"✅ Successfully downloaded {os.path.basename(filepath)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {os.path.basename(filepath)}: {e}")
        return False

def download_yolo_files():
    """Download YOLO model files if they don't exist"""
    logger.info("🔍 Checking YOLO model files...")
    
    # Create yolo_files directory if it doesn't exist
    YOLO_FILES_DIR.mkdir(exist_ok=True)
    
    files_to_download = []
    for filename, url in YOLO_URLS.items():
        filepath = YOLO_FILES_DIR / filename
        if not filepath.exists():
            files_to_download.append((filename, url, str(filepath)))
        else:
            logger.info(f"✅ {filename} already exists")
    
    if not files_to_download:
        logger.info("✅ All YOLO files are present")
        return True
    
    logger.info(f"📥 Need to download {len(files_to_download)} YOLO files...")
    
    # Download missing files
    success_count = 0
    for filename, url, filepath in files_to_download:
        if download_file(url, filepath):
            success_count += 1
        else:
            logger.error(f"❌ Failed to download {filename}")
    
    if success_count == len(files_to_download):
        logger.info("✅ All YOLO files downloaded successfully!")
        return True
    else:
        logger.error(f"❌ Failed to download {len(files_to_download) - success_count} files")
        return False

def validate_yolo_files():
    """Validate that YOLO model files exist, download if missing"""
    # First try to download missing files
    if not download_yolo_files():
        missing_files = []
        for file in [YOLO_WEIGHTS, YOLO_CONFIG, YOLO_NAMES]:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing YOLO files after download attempt: {missing_files}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to download required YOLO files: {missing_files}. Please check your internet connection."
            )
    
    logger.info("YOLO files validated successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    validate_yolo_files()
    yield
    # Shutdown (nothing to do here)

app = FastAPI(
    title="YOLO Similarity Detection API",
    version="1.0.0",
    description="API for detecting similarity between non-living objects in uploaded images using YOLOv4.",
    lifespan=lifespan
)

@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {"message": "YOLO Similarity Detection API is running", "status": "healthy", "version": "1.0.0"}

@app.post("/download-yolo-files/", summary="Manually download YOLO files")
async def download_yolo_files_endpoint():
    """Manually trigger YOLO files download"""
    try:
        success = download_yolo_files()
        if success:
            return {"status": "success", "message": "All YOLO files downloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to download some YOLO files")
    except Exception as e:
        logger.error(f"Error downloading YOLO files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/detect-similarity-upload/", summary="Run similarity detection with uploaded images")
async def detect_similarity_upload(
    pickup_images: List[UploadFile] = File(..., description="Images from the pickup location"),
    warehouse_images: List[UploadFile] = File(..., description="Images from the warehouse"),
    similarity_threshold: float = Form(0.75, description="Threshold for similarity detection (0.0 to 1.0)")
):
    """
    Upload images for pickup and warehouse to detect similar non-living objects.
    Returns a JSON report and a run_id to download matched images.
    """
    try:
        # Generate unique run_id
        run_id = str(uuid.uuid4())
        output_folder = OUTPUT_BASE_DIR / run_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # Create temporary directories for uploaded images
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_pickup_dir, \
             tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_warehouse_dir:

            # Save pickup images
            for image in pickup_images:
                if not any(image.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.filename}")
                file_path = Path(temp_pickup_dir) / image.filename
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(image.file, f)

            # Save warehouse images
            for image in warehouse_images:
                if not any(image.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.filename}")
                file_path = Path(temp_warehouse_dir) / image.filename
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(image.file, f)

            # Initialize and run detector
            detector = ImprovedYOLOSimilarityDetector(
                pickup_folder=temp_pickup_dir,
                warehouse_folder=temp_warehouse_dir,
                output_folder=str(output_folder)
            )
            detector.similarity_threshold = similarity_threshold
            result = detector.run_similarity_detection()

            # Check if detection returned results
            if isinstance(result, dict) and 'error' in result:
                raise HTTPException(status_code=400, detail=result['error'])

            # Load additional reports for backward compatibility
            report_dir = output_folder / "reports"
            summary_file = report_dir / "summary.json"
            unmatched_file = report_dir / "unmatched_images_report.json"
            
            summary_data = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)

            # Construct comprehensive response
            response = {
                "status": "success",
                "message": "Similarity detection completed",
                "run_id": run_id,
                "summary": result.get('summary', summary_data),
                "matches": result.get('matches', []),
                "unmatched_report": result.get('unmatched_report', {}),
                "download_instructions": {
                    "matched_images": "Use GET /download-match-image/{run_id}/{match_id}/{image_type} with image_type as 'pickup', 'warehouse', or 'combined'",
                    "unmatched_images": "Use GET /download-unmatched-image/{run_id}/{image_type}/{filename} with image_type as 'pickup' or 'warehouse'",
                    "reports": {
                        "summary_json": "GET /download-summary/{run_id}",
                        "unmatched_json": "GET /download-unmatched-json/{run_id}",
                        "similarity_json": "GET /download-similarity-report/{run_id}",
                        "all_reports_zip": "GET /download-all-reports/{run_id}"
                    }
                }
            }

            return response

    except Exception as e:
        logger.error(f"Error during similarity detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/unmatched-report/{run_id}", summary="Get unmatched images report for a specific run")
async def get_unmatched_report(run_id: str):
    """
    Get the detailed unmatched images report for a specific run_id.
    """
    try:
        report_dir = OUTPUT_BASE_DIR / run_id / "reports"
        unmatched_file = report_dir / "unmatched_images_report.json"
        
        if not unmatched_file.exists():
            raise HTTPException(status_code=404, detail=f"Unmatched report not found for run {run_id}")
        
        with open(unmatched_file, 'r') as f:
            unmatched_data = json.load(f)
        
        return {
            "status": "success",
            "run_id": run_id,
            "unmatched_report": unmatched_data
        }

    except Exception as e:
        logger.error(f"Error getting unmatched report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download-unmatched-image/{run_id}/{image_type}/{filename}", summary="Download a specific unmatched flagged image")
async def download_unmatched_image(run_id: str, image_type: str, filename: str):
    """
    Download an unmatched flagged image using run_id, image_type, and filename.
    image_type: 'pickup' or 'warehouse'.
    """
    try:
        base_dir = OUTPUT_BASE_DIR / run_id / "flag"
        if image_type == "pickup":
            dir_path = base_dir / "pickup_unmatched"
        elif image_type == "warehouse":
            dir_path = base_dir / "warehouse_unmatched"
        else:
            raise HTTPException(status_code=400, detail="Invalid image_type. Use 'pickup' or 'warehouse'")

        image_path = dir_path / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Unmatched {image_type} image '{filename}' not found in run {run_id}")

        return FileResponse(image_path, media_type="image/jpeg", filename=filename)

    except Exception as e:
        logger.error(f"Error downloading unmatched image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download-match-image/{run_id}/{match_id}/{image_type}", summary="Download a specific match image")
async def download_match_image(run_id: str, match_id: str, image_type: str):
    """
    Download a match image using run_id and match_id.
    image_type: 'pickup', 'warehouse', or 'combined'.
    """
    try:
        base_dir = OUTPUT_BASE_DIR / run_id
        if image_type == "pickup":
            dir_path = base_dir / "pickup_matches"
        elif image_type == "warehouse":
            dir_path = base_dir / "warehouse_matches"
        elif image_type == "combined":
            dir_path = base_dir / "combined_matches"
        else:
            raise HTTPException(status_code=400, detail="Invalid image_type. Use 'pickup', 'warehouse', or 'combined'")

        image_files = list(dir_path.glob(f"{match_id}_*.jpg"))
        if not image_files:
            raise HTTPException(status_code=404, detail=f"No {image_type} image found for match_id {match_id} in run {run_id}")

        return FileResponse(image_files[0], media_type="image/jpeg", filename=image_files[0].name)

    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download-summary/{run_id}", summary="Download summary.json file")
async def download_summary_json(run_id: str):
    """
    Download the summary.json file for a specific run_id.
    """
    try:
        report_dir = OUTPUT_BASE_DIR / run_id / "reports"
        summary_file = report_dir / "summary.json"
        
        if not summary_file.exists():
            raise HTTPException(status_code=404, detail=f"Summary file not found for run {run_id}")
        
        return FileResponse(
            summary_file, 
            media_type="application/json", 
            filename=f"summary_{run_id}.json"
        )

    except Exception as e:
        logger.error(f"Error downloading summary file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download-unmatched-json/{run_id}", summary="Download unmatched_images_report.json file")
async def download_unmatched_report_json(run_id: str):
    """
    Download the unmatched_images_report.json file for a specific run_id.
    """
    try:
        report_dir = OUTPUT_BASE_DIR / run_id / "reports"
        unmatched_file = report_dir / "unmatched_images_report.json"
        
        if not unmatched_file.exists():
            raise HTTPException(status_code=404, detail=f"Unmatched report file not found for run {run_id}")
        
        return FileResponse(
            unmatched_file, 
            media_type="application/json", 
            filename=f"unmatched_report_{run_id}.json"
        )

    except Exception as e:
        logger.error(f"Error downloading unmatched report file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download-similarity-report/{run_id}", summary="Download similarity_report.json file")
async def download_similarity_report_json(run_id: str):
    """
    Download the similarity_report_{timestamp}.json file for a specific run_id.
    """
    try:
        report_dir = OUTPUT_BASE_DIR / run_id / "reports"
        
        # Find the similarity report file (has timestamp in filename)
        similarity_files = list(report_dir.glob("similarity_report_*.json"))
        if not similarity_files:
            raise HTTPException(status_code=404, detail=f"Similarity report file not found for run {run_id}")
        
        # Get the most recent similarity report if multiple exist
        latest_similarity_file = max(similarity_files, key=os.path.getctime)
        
        return FileResponse(
            latest_similarity_file, 
            media_type="application/json", 
            filename=f"similarity_report_{run_id}.json"
        )

    except Exception as e:
        logger.error(f"Error downloading similarity report file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download-all-reports/{run_id}", summary="Download all report files as ZIP")
async def download_all_reports_zip(run_id: str):
    """
    Download all report files (summary.json, unmatched_images_report.json, similarity_report.json) as a ZIP file.
    """
    try:
        import zipfile
        from io import BytesIO
        
        report_dir = OUTPUT_BASE_DIR / run_id / "reports"
        if not report_dir.exists():
            raise HTTPException(status_code=404, detail=f"Reports directory not found for run {run_id}")
        
        # Create a temporary ZIP file
        import tempfile
        temp_zip_path = tempfile.mktemp(suffix='.zip')
        
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add summary.json if it exists
            summary_file = report_dir / "summary.json"
            if summary_file.exists():
                zip_file.write(summary_file, "summary.json")
            
            # Add unmatched report if it exists
            unmatched_file = report_dir / "unmatched_images_report.json"
            if unmatched_file.exists():
                zip_file.write(unmatched_file, "unmatched_images_report.json")
            
            # Add similarity report if it exists
            similarity_files = list(report_dir.glob("similarity_report_*.json"))
            if similarity_files:
                latest_similarity_file = max(similarity_files, key=os.path.getctime)
                zip_file.write(latest_similarity_file, f"similarity_report.json")
        
        return FileResponse(
            temp_zip_path,
            media_type="application/zip",
            filename=f"reports_{run_id}.zip"
        )

    except Exception as e:
        logger.error(f"Error creating reports ZIP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def download_image_from_url(image_url: str, save_path: str) -> bool:
    """Download an image from URL and save it to the specified path"""
    try:
        logger.info(f"Downloading image from {image_url}")
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check if response contains image data
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            logger.warning(f"URL does not appear to contain an image: {content_type}")
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"✅ Successfully downloaded image to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download image from {image_url}: {e}")
        return False

def process_json_images(json_data: dict, temp_dir: str, prefix: str) -> bool:
    """Process JSON data to download images from URLs using product ID as filename"""
    try:
        images = json_data.get('images', [])
        product_id = json_data.get('product', {}).get('id', 'unknown_product')
        
        if not images:
            logger.warning(f"No images found in JSON data for {prefix}")
            return False
        
        success_count = 0
        for i, image_url in enumerate(images):
            # Use product ID and index for filename
            filename = f"{product_id}_{i}.jpg"
            file_path = os.path.join(temp_dir, filename)
            
            if download_image_from_url(image_url, file_path):
                success_count += 1
            else:
                logger.error(f"Failed to download image {i} for product {product_id}")
        
        if success_count > 0:
            logger.info(f"✅ Successfully downloaded {success_count}/{len(images)} images for {prefix} (product: {product_id})")
            return True
        else:
            logger.error(f"❌ Failed to download any images for {prefix} (product: {product_id})")
            return False
            
    except Exception as e:
        logger.error(f"Error processing JSON images for {prefix}: {str(e)}")
        return False

@app.post("/process-json-images/", summary="Process JSON file and download images")
async def process_json_images_endpoint(
    json_file: UploadFile = File(..., description="JSON file containing image URLs"),
    prefix: str = Form(..., description="Prefix for output files")
):
    """
    Process a JSON file containing image URLs and download the images to a temporary directory.
    Returns the list of downloaded image file paths.
    """
    try:
        # Validate and save the uploaded JSON file
        if not json_file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Uploaded file is not a JSON file")
        
        # Create a temporary directory for this upload
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            # Save the uploaded JSON file
            json_path = Path(temp_dir) / json_file.filename
            with open(json_path, "wb") as f:
                shutil.copyfileobj(json_file.file, f)
            
            # Load JSON data
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Download images specified in the JSON data
            success = process_json_images(json_data, temp_dir, prefix)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to download images from JSON data")
            
            # Collect downloaded image file paths
            downloaded_files = [str(Path(temp_dir) / f) for f in os.listdir(temp_dir) if f.endswith('.jpg')]
            
            return {
                "status": "success",
                "message": "Images processed and downloaded",
                "downloaded_files": downloaded_files
            }

    except Exception as e:
        logger.error(f"Error processing JSON images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/detect-similarity-json/", summary="Run similarity detection with JSON files containing image URLs")
async def detect_similarity_json(
    pickup_json: UploadFile = File(..., description="JSON file with pickup data containing image URLs"),
    warehouse_json: UploadFile = File(..., description="JSON file with warehouse data containing image URLs"),
    similarity_threshold: float = Form(0.75, description="Threshold for similarity detection (0.0 to 1.0)")
):
    """
    Upload JSON files containing image URLs for pickup and warehouse to detect similar non-living objects.
    The JSON files should contain 'images' array with URLs and 'product' object with 'id'.
    Returns a JSON report and a run_id to download matched images.
    """
    try:
        # Validate JSON file types
        if not pickup_json.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Pickup file must be a JSON file")
        if not warehouse_json.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Warehouse file must be a JSON file")
        
        # Generate unique run_id
        run_id = str(uuid.uuid4())
        output_folder = OUTPUT_BASE_DIR / run_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # Create temporary directories for downloaded images
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_pickup_dir, \
             tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_warehouse_dir:

            # Read and parse pickup JSON
            try:
                pickup_content = await pickup_json.read()
                pickup_data = json.loads(pickup_content.decode('utf-8'))
                logger.info(f"Loaded pickup JSON with product ID: {pickup_data.get('product', {}).get('id', 'unknown')}")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid pickup JSON format: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading pickup JSON: {str(e)}")

            # Read and parse warehouse JSON
            try:
                warehouse_content = await warehouse_json.read()
                warehouse_data = json.loads(warehouse_content.decode('utf-8'))
                logger.info(f"Loaded warehouse JSON with product ID: {warehouse_data.get('product', {}).get('id', 'unknown')}")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid warehouse JSON format: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading warehouse JSON: {str(e)}")

            # Download pickup images from URLs
            pickup_success = process_json_images(pickup_data, temp_pickup_dir, "pickup")
            if not pickup_success:
                raise HTTPException(status_code=400, detail="Failed to download pickup images from URLs")

            # Download warehouse images from URLs  
            warehouse_success = process_json_images(warehouse_data, temp_warehouse_dir, "warehouse")
            if not warehouse_success:
                raise HTTPException(status_code=400, detail="Failed to download warehouse images from URLs")

            # Check if we have any images to process
            pickup_images = list(Path(temp_pickup_dir).glob("*.jpg"))
            warehouse_images = list(Path(temp_warehouse_dir).glob("*.jpg"))
            
            if not pickup_images:
                raise HTTPException(status_code=400, detail="No pickup images were successfully downloaded")
            if not warehouse_images:
                raise HTTPException(status_code=400, detail="No warehouse images were successfully downloaded")

            logger.info(f"Processing {len(pickup_images)} pickup images and {len(warehouse_images)} warehouse images")

            # Initialize and run detector
            detector = ImprovedYOLOSimilarityDetector(
                pickup_folder=temp_pickup_dir,
                warehouse_folder=temp_warehouse_dir,
                output_folder=str(output_folder)
            )
            detector.similarity_threshold = similarity_threshold
            result = detector.run_similarity_detection()

            # Check if detection returned results
            if isinstance(result, dict) and 'error' in result:
                raise HTTPException(status_code=400, detail=result['error'])

            # Load additional reports for backward compatibility
            report_dir = output_folder / "reports"
            summary_file = report_dir / "summary.json"
            unmatched_file = report_dir / "unmatched_images_report.json"
            
            summary_data = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)

            # Construct comprehensive response with product information
            response = {
                "status": "success",
                "message": "Similarity detection completed using JSON image URLs",
                "run_id": run_id,
                "pickup_product": {
                    "id": pickup_data.get('product', {}).get('id', 'unknown'),
                    "name": pickup_data.get('product', {}).get('name', 'unknown'),
                    "images_processed": len(pickup_images)
                },
                "warehouse_product": {
                    "id": warehouse_data.get('product', {}).get('id', 'unknown'),
                    "name": warehouse_data.get('product', {}).get('name', 'unknown'),
                    "images_processed": len(warehouse_images)
                },
                "summary": result.get('summary', summary_data),
                "matches": result.get('matches', []),
                "unmatched_report": result.get('unmatched_report', {}),
                "download_instructions": {
                    "matched_images": "Use GET /download-match-image/{run_id}/{match_id}/{image_type} with image_type as 'pickup', 'warehouse', or 'combined'",
                    "unmatched_images": "Use GET /download-unmatched-image/{run_id}/{image_type}/{filename} with image_type as 'pickup' or 'warehouse'",
                    "reports": {
                        "summary_json": "GET /download-summary/{run_id}",
                        "unmatched_json": "GET /download-unmatched-json/{run_id}",
                        "similarity_json": "GET /download-similarity-report/{run_id}",
                        "all_reports_zip": "GET /download-all-reports/{run_id}"
                    }
                }
            }

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during JSON-based similarity detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/detect-similarity-from-files/", summary="Run similarity detection using local JSON files")
async def detect_similarity_from_files(
    pickup_json_path: str = Form(..., description="Path to pickup JSON file (e.g., 'pick.json')"),
    warehouse_json_path: str = Form(..., description="Path to warehouse JSON file (e.g., 'ware.json')"),
    similarity_threshold: float = Form(0.75, description="Threshold for similarity detection (0.0 to 1.0)")
):
    """
    Use local JSON files containing image URLs for pickup and warehouse to detect similar non-living objects.
    The JSON files should contain 'images' array with URLs and 'product' object with 'id'.
    """
    try:
        # Validate that JSON files exist
        pickup_path = Path(pickup_json_path)
        warehouse_path = Path(warehouse_json_path)
        
        if not pickup_path.exists():
            raise HTTPException(status_code=404, detail=f"Pickup JSON file not found: {pickup_json_path}")
        if not warehouse_path.exists():
            raise HTTPException(status_code=404, detail=f"Warehouse JSON file not found: {warehouse_json_path}")
        
        # Generate unique run_id
        run_id = str(uuid.uuid4())
        output_folder = OUTPUT_BASE_DIR / run_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # Create temporary directories for downloaded images
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_pickup_dir, \
             tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_warehouse_dir:

            # Read and parse pickup JSON
            try:
                with open(pickup_path, 'r', encoding='utf-8') as f:
                    pickup_data = json.load(f)
                logger.info(f"Loaded pickup JSON from {pickup_json_path} with product ID: {pickup_data.get('product', {}).get('id', 'unknown')}")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid pickup JSON format in {pickup_json_path}: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading pickup JSON from {pickup_json_path}: {str(e)}")

            # Read and parse warehouse JSON
            try:
                with open(warehouse_path, 'r', encoding='utf-8') as f:
                    warehouse_data = json.load(f)
                logger.info(f"Loaded warehouse JSON from {warehouse_json_path} with product ID: {warehouse_data.get('product', {}).get('id', 'unknown')}")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid warehouse JSON format in {warehouse_json_path}: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading warehouse JSON from {warehouse_json_path}: {str(e)}")

            # Download pickup images from URLs
            pickup_success = process_json_images(pickup_data, temp_pickup_dir, "pickup")
            if not pickup_success:
                raise HTTPException(status_code=400, detail="Failed to download pickup images from URLs")

            # Download warehouse images from URLs  
            warehouse_success = process_json_images(warehouse_data, temp_warehouse_dir, "warehouse")
            if not warehouse_success:
                raise HTTPException(status_code=400, detail="Failed to download warehouse images from URLs")

            # Check if we have any images to process
            pickup_images = list(Path(temp_pickup_dir).glob("*.jpg"))
            warehouse_images = list(Path(temp_warehouse_dir).glob("*.jpg"))
            
            if not pickup_images:
                raise HTTPException(status_code=400, detail="No pickup images were successfully downloaded")
            if not warehouse_images:
                raise HTTPException(status_code=400, detail="No warehouse images were successfully downloaded")

            logger.info(f"Processing {len(pickup_images)} pickup images and {len(warehouse_images)} warehouse images")

            # Initialize and run detector
            detector = ImprovedYOLOSimilarityDetector(
                pickup_folder=temp_pickup_dir,
                warehouse_folder=temp_warehouse_dir,
                output_folder=str(output_folder)
            )
            detector.similarity_threshold = similarity_threshold
            result = detector.run_similarity_detection()

            # Check if detection returned results
            if isinstance(result, dict) and 'error' in result:
                raise HTTPException(status_code=400, detail=result['error'])

            # Load additional reports for backward compatibility
            report_dir = output_folder / "reports"
            summary_file = report_dir / "summary.json"
            
            summary_data = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)

            # Construct comprehensive response with product information
            response = {
                "status": "success",
                "message": "Similarity detection completed using local JSON files",
                "run_id": run_id,
                "pickup_product": {
                    "id": pickup_data.get('product', {}).get('id', 'unknown'),
                    "name": pickup_data.get('product', {}).get('name', 'unknown'),
                    "images_processed": len(pickup_images),
                    "source_file": pickup_json_path
                },
                "warehouse_product": {
                    "id": warehouse_data.get('product', {}).get('id', 'unknown'),
                    "name": warehouse_data.get('product', {}).get('name', 'unknown'),
                    "images_processed": len(warehouse_images),
                    "source_file": warehouse_json_path
                },
                "summary": result.get('summary', summary_data),
                "matches": result.get('matches', []),
                "unmatched_report": result.get('unmatched_report', {}),
                "download_instructions": {
                    "matched_images": "Use GET /download-match-image/{run_id}/{match_id}/{image_type} with image_type as 'pickup', 'warehouse', or 'combined'",
                    "unmatched_images": "Use GET /download-unmatched-image/{run_id}/{image_type}/{filename} with image_type as 'pickup' or 'warehouse'",
                    "reports": {
                        "summary_json": "GET /download-summary/{run_id}",
                        "unmatched_json": "GET /download-unmatched-json/{run_id}",
                        "similarity_json": "GET /download-similarity-report/{run_id}",
                        "all_reports_zip": "GET /download-all-reports/{run_id}"
                    }
                }
            }

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file-based similarity detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)