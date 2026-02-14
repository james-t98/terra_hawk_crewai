"""
TerraHawk Reports API — Lambda function for listing and fetching reports from S3.
Deploy behind API Gateway (REST or HTTP API).

Endpoints (via httpMethod + path):
  GET  /reports/{farm_id}                → List today's reports (or ?date=YYYY-MM-DD)
  GET  /reports/{farm_id}/{report_type}  → Fetch a specific report's JSON content

Environment variables:
  S3_BUCKET       — S3 bucket name (required)
  AWS_REGION_NAME — AWS region (default: eu-west-1)
"""
import json
import os
import boto3
from datetime import datetime
from botocore.exceptions import ClientError


s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION_NAME", "eu-west-1"))
BUCKET = os.environ.get("S3_BUCKET", "")

# The report types the frontend expects
REPORT_TYPES = [
    "master_report",
    "vision_analysis",
    "sensor_analysis",
    "weather_report",
    "compliance_report",
]

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,OPTIONS",
    "Content-Type": "application/json",
}


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def _list_reports(farm_id: str, date: str) -> dict:
    """List available reports for a farm on a given date."""
    prefix = f"{farm_id}/{date}/reports/"

    try:
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    except ClientError as e:
        return _resp(500, {"error": str(e)})

    contents = response.get("Contents", [])

    if not contents:
        return _resp(200, {
            "farm_id": farm_id,
            "date": date,
            "reports_available": False,
            "message": f"No reports generated yet for {date}",
            "reports": [],
        })

    # Group by report_type — pick the latest .json file for each type
    reports = []
    for rt in REPORT_TYPES:
        # Find all .json files matching this report type
        matches = [
            obj for obj in contents
            if obj["Key"].rsplit("/", 1)[-1].startswith(rt) and obj["Key"].endswith(".json")
        ]
        if not matches:
            continue

        # Sort by key descending to get newest
        matches.sort(key=lambda x: x["Key"], reverse=True)
        latest = matches[0]

        # Generate presigned URL (1 hour)
        try:
            presigned = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET, "Key": latest["Key"]},
                ExpiresIn=3600,
            )
        except ClientError:
            presigned = None

        reports.append({
            "report_type": rt,
            "key": latest["Key"],
            "size": _format_size(latest["Size"]),
            "size_bytes": latest["Size"],
            "last_modified": latest["LastModified"].isoformat() if hasattr(latest["LastModified"], "isoformat") else str(latest["LastModified"]),
            "presigned_url": presigned,
        })

    return _resp(200, {
        "farm_id": farm_id,
        "date": date,
        "reports_available": len(reports) > 0,
        "reports": reports,
    })


def _fetch_report(farm_id: str, report_type: str, date: str) -> dict:
    """Fetch the latest report content for a specific type."""
    prefix = f"{farm_id}/{date}/reports/{report_type}_"

    try:
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    except ClientError as e:
        return _resp(500, {"error": str(e)})

    contents = response.get("Contents", [])

    # Find latest .json file
    json_files = [obj for obj in contents if obj["Key"].endswith(".json")]
    if not json_files:
        return _resp(404, {
            "error": "report_not_found",
            "message": f"No {report_type.replace('_', ' ')} report found for {date}",
            "farm_id": farm_id,
            "report_type": report_type,
            "date": date,
        })

    json_files.sort(key=lambda x: x["Key"], reverse=True)
    latest_key = json_files[0]["Key"]

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=latest_key)
        content = obj["Body"].read().decode("utf-8")
        parsed = json.loads(content)
    except (ClientError, json.JSONDecodeError) as e:
        return _resp(500, {"error": "read_failed", "message": str(e)})

    return _resp(200, {
        "farm_id": farm_id,
        "report_type": report_type,
        "date": date,
        "key": latest_key,
        "size": _format_size(json_files[0]["Size"]),
        "last_modified": json_files[0]["LastModified"].isoformat() if hasattr(json_files[0]["LastModified"], "isoformat") else str(json_files[0]["LastModified"]),
        "content": parsed,
    })


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def handler(event, context):
    """Lambda entry point."""
    if not BUCKET:
        return _resp(500, {"error": "S3_BUCKET not configured"})

    method = event.get("httpMethod", "GET")

    # Handle CORS preflight
    if method == "OPTIONS":
        return _resp(200, {})

    path = event.get("path", "")
    path_params = event.get("pathParameters") or {}
    query = event.get("queryStringParameters") or {}

    # Parse path: /reports/{farm_id} or /reports/{farm_id}/{report_type}
    farm_id = path_params.get("farm_id")
    report_type = path_params.get("report_type")

    # Fallback: parse from path manually if proxy integration
    if not farm_id:
        parts = [p for p in path.strip("/").split("/") if p]
        if len(parts) >= 2 and parts[0] == "reports":
            farm_id = parts[1]
        if len(parts) >= 3:
            report_type = parts[2]

    if not farm_id:
        return _resp(400, {"error": "farm_id is required"})

    date = query.get("date", datetime.now().strftime("%Y-%m-%d"))

    if report_type:
        return _fetch_report(farm_id, report_type, date)
    else:
        return _list_reports(farm_id, date)
