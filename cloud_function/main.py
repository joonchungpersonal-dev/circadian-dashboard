"""Cloud Function to ingest HealthKit data from Auto Export app."""

import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import functions_framework
from flask import Request
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
API_KEY = os.environ.get("API_KEY", "")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "sleep-health-dashboard-healthkit-exports")


def validate_api_key(request: Request) -> Tuple[bool, Optional[str]]:
    """
    Validate the API key from request headers.

    Args:
        request: Flask request object

    Returns:
        Tuple of (is_valid, error_message)
    """
    provided_key = request.headers.get("X-API-Key", "")

    if not API_KEY:
        logger.warning("API_KEY environment variable not set")
        return False, "Server configuration error"

    if not provided_key:
        return False, "Missing X-API-Key header"

    if provided_key != API_KEY:
        return False, "Invalid API key"

    return True, None


def extract_payload(request: Request) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Extract payload from request, handling multiple content types.

    Auto Export may send data as:
    - application/json: Direct JSON body
    - multipart/form-data: JSON file attachment
    - application/x-www-form-urlencoded: Form data

    Args:
        request: Flask request object

    Returns:
        Tuple of (payload_dict, error_message)
    """
    content_type = request.content_type or ""
    logger.info(f"Content-Type: {content_type}")

    try:
        # Try JSON body first (most common)
        if "application/json" in content_type:
            payload = request.get_json(force=True)
            if payload:
                logger.info("Parsed JSON body successfully")
                return payload, None

        # Try multipart form data (file upload)
        if "multipart/form-data" in content_type:
            files = request.files
            for filename, file_obj in files.items():
                logger.info(f"Processing file: {filename}")
                content = file_obj.read().decode("utf-8")
                payload = json.loads(content)
                return payload, None

        # Try form data
        if "application/x-www-form-urlencoded" in content_type:
            form_data = request.form.to_dict()
            # Check if there's a 'data' or 'payload' field with JSON
            for key in ["data", "payload", "json"]:
                if key in form_data:
                    payload = json.loads(form_data[key])
                    return payload, None
            # Return form data as-is
            if form_data:
                return form_data, None

        # Fallback: try to parse body as JSON regardless of content type
        try:
            payload = request.get_json(force=True)
            if payload:
                logger.info("Parsed body as JSON (fallback)")
                return payload, None
        except Exception:
            pass

        # Try raw data
        raw_data = request.get_data(as_text=True)
        if raw_data:
            try:
                payload = json.loads(raw_data)
                return payload, None
            except json.JSONDecodeError:
                logger.warning(f"Could not parse raw data: {raw_data[:200]}...")

        return None, "Could not parse request body"

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None, f"Invalid JSON: {str(e)}"
    except Exception as e:
        logger.error(f"Error extracting payload: {e}")
        return None, f"Error processing request: {str(e)}"


def write_to_gcs(payload: Dict, request: Request) -> Tuple[Optional[str], Optional[str]]:
    """
    Write payload to GCS bucket.

    Args:
        payload: Parsed payload dict
        request: Original request (for metadata)

    Returns:
        Tuple of (blob_path, error_message)
    """
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)

        # Generate filename: exports/{date}/{timestamp}_healthkit.json
        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        timestamp = int(now.timestamp())
        blob_path = f"exports/{date_str}/{timestamp}_healthkit.json"

        # Add ingestion metadata
        enriched_payload = {
            "ingestion_metadata": {
                "received_at": now.isoformat() + "Z",
                "source_ip": request.remote_addr,
                "content_type": request.content_type,
                "user_agent": request.headers.get("User-Agent", ""),
            },
            "data": payload
        }

        # Write to GCS
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(enriched_payload, indent=2, default=str),
            content_type="application/json"
        )

        logger.info(f"Wrote to gs://{GCS_BUCKET}/{blob_path}")
        return blob_path, None

    except Exception as e:
        logger.error(f"GCS write error: {e}")
        return None, f"Failed to write to storage: {str(e)}"


@functions_framework.http
def healthkit_ingest(request: Request) -> Tuple[Dict[str, Any], int]:
    """
    Main Cloud Function entry point.

    Handles incoming HealthKit data exports from Auto Export app.

    Args:
        request: Flask request object

    Returns:
        Tuple of (response_dict, status_code)
    """
    logger.info("="*50)
    logger.info("Received healthkit-ingest request")
    logger.info(f"Method: {request.method}")
    logger.info(f"Headers: {dict(request.headers)}")

    # Handle CORS preflight
    if request.method == "OPTIONS":
        return {}, 204

    # Validate API key
    is_valid, error = validate_api_key(request)
    if not is_valid:
        logger.warning(f"API key validation failed: {error}")
        return {"status": "error", "message": error}, 401

    # Extract payload
    payload, error = extract_payload(request)
    if error:
        logger.error(f"Payload extraction failed: {error}")
        return {"status": "error", "message": error}, 400

    if not payload:
        logger.error("Empty payload")
        return {"status": "error", "message": "Empty payload"}, 400

    # Log payload summary
    if isinstance(payload, dict):
        keys = list(payload.keys())[:10]
        logger.info(f"Payload keys: {keys}")

    # Write to GCS
    blob_path, error = write_to_gcs(payload, request)
    if error:
        logger.error(f"GCS write failed: {error}")
        return {"status": "error", "message": error}, 500

    logger.info("Request processed successfully")
    logger.info("="*50)

    return {
        "status": "success",
        "blob": blob_path,
        "bucket": GCS_BUCKET
    }, 200
