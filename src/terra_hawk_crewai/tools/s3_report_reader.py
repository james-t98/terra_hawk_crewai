#!/usr/bin/env python3
"""
S3 Report Reader — Retrieves previous reports from S3 for historical comparison.
Used by Master Chief to compare current analysis against past reports and identify trends.
"""
import os
import json
from typing import Type, Dict, Any, List, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3ReportReaderInput(BaseModel):
    """Input schema for S3ReportReader."""

    farm_id: str = Field(..., description="Farm ID for partitioned lookup (e.g., 'FARM-001').")
    report_type: str = Field(
        ...,
        description=(
            "Type of report to retrieve: 'vision_analysis', 'weather_report', "
            "'sensor_analysis', 'compliance_report', or 'master_report'."
        ),
    )
    date: Optional[str] = Field(
        default=None,
        description="Date in YYYY-MM-DD format. If omitted, searches the last 7 days.",
    )
    limit: int = Field(
        default=3,
        description="Maximum number of reports to return (newest first). Default: 3.",
        ge=1,
        le=20,
    )
    region: str = Field(default="eu-west-1", description="AWS region where the bucket is located.")


class S3ReportReader(BaseTool):
    name: str = "S3 Report Reader"
    description: str = (
        "Reads previous analysis reports from S3 for historical comparison. "
        "Reports are stored in partitioned paths: bucket/farm_id/date/reports/report_type_timestamp.md. "
        "Returns report contents with metadata. Useful for identifying trends across days."
    )
    args_schema: Type[BaseModel] = S3ReportReaderInput

    def _run(
        self,
        farm_id: str,
        report_type: str,
        date: str = None,
        limit: int = 3,
        region: str = "eu-west-1",
    ) -> Dict[str, Any]:
        """
        Read reports from S3 bucket.

        Args:
            farm_id: Farm ID for partitioned lookup
            report_type: Type of report to find
            date: Specific date (YYYY-MM-DD) or None for last 7 days
            limit: Max reports to return
            region: AWS region

        Returns:
            Dictionary containing report contents and metadata
        """
        if not BOTO3_AVAILABLE:
            return {
                "success": False,
                "error": "boto3 library not installed",
                "details": "Please install boto3: pip install boto3",
            }

        bucket_name = os.environ.get("S3_BUCKET")
        if not bucket_name:
            return {
                "success": False,
                "error": "S3_BUCKET environment variable not set",
                "details": "Please configure the S3_BUCKET environment variable.",
            }

        try:
            s3_client = boto3.client("s3", region_name=region)

            # Determine which date prefixes to search
            if date:
                date_prefixes = [date]
            else:
                # Search last 7 days
                today = datetime.now()
                date_prefixes = [
                    (today - timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(7)
                ]

            # Collect matching report keys across dates
            matching_keys: List[Dict[str, Any]] = []

            for d in date_prefixes:
                prefix = f"{farm_id}/{d}/reports/"
                try:
                    paginator = s3_client.get_paginator("list_objects_v2")
                    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                        for obj in page.get("Contents", []):
                            key = obj["Key"]
                            # Match report_type in filename
                            filename = key.rsplit("/", 1)[-1]
                            if filename.startswith(report_type):
                                matching_keys.append({
                                    "key": key,
                                    "date": d,
                                    "last_modified": obj["LastModified"].isoformat(),
                                    "size_bytes": obj["Size"],
                                })
                except ClientError:
                    continue  # Skip dates with no data

            if not matching_keys:
                return {
                    "success": True,
                    "farm_id": farm_id,
                    "report_type": report_type,
                    "reports_found": 0,
                    "message": f"No {report_type} reports found for {farm_id}"
                              + (f" on {date}" if date else " in the last 7 days"),
                    "reports": [],
                }

            # Sort by key descending (newest first) and limit
            matching_keys.sort(key=lambda x: x["key"], reverse=True)
            matching_keys = matching_keys[:limit]

            # Fetch report contents
            reports: List[Dict[str, Any]] = []
            for meta in matching_keys:
                try:
                    response = s3_client.get_object(Bucket=bucket_name, Key=meta["key"])
                    content = response["Body"].read().decode("utf-8")

                    # Try to parse as JSON for structured comparison
                    parsed = None
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        pass

                    reports.append({
                        "key": meta["key"],
                        "date": meta["date"],
                        "last_modified": meta["last_modified"],
                        "size_bytes": meta["size_bytes"],
                        "content": parsed if parsed else content,
                        "is_json": parsed is not None,
                    })
                except ClientError as e:
                    reports.append({
                        "key": meta["key"],
                        "date": meta["date"],
                        "error": f"Failed to read: {e.response['Error']['Code']}",
                    })

            return {
                "success": True,
                "farm_id": farm_id,
                "report_type": report_type,
                "reports_found": len(reports),
                "date_range": f"{matching_keys[-1]['date']} to {matching_keys[0]['date']}",
                "reports": reports,
            }

        except NoCredentialsError:
            return {
                "success": False,
                "error": "AWS credentials not found",
                "details": "Please configure AWS credentials.",
            }
        except ClientError as e:
            return {
                "success": False,
                "error": f"AWS S3 error: {e.response['Error']['Code']}",
                "details": e.response["Error"]["Message"],
            }
        except Exception as e:
            return {
                "success": False,
                "error": "Unexpected error",
                "details": str(e),
            }
