#!/usr/bin/env python3
import json
from typing import Type, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3ReportWriterInput(BaseModel):
    """Input schema for S3ReportWriter."""

    bucket_name: str = Field(..., description="Name of the S3 bucket to store the report (e.g., 'drone-images-replicate').")
    farm_id: str = Field(..., description="Farm ID for partitioning (e.g., 'FARM-001').")
    report_content: str = Field(..., description="The markdown content of the report to be written.")
    report_type: str = Field(..., description="Type of report: 'vision_analysis', 'weather_report', 'sensor_analysis', 'financial_analysis', 'compliance_report', 'master_report', 'mission_plan', 'realtime_monitoring', 'image_analysis', or 'maintenance_prediction'.")
    date: str = Field(default=None, description="Date for partitioning in YYYY-MM-DD format. If not provided, uses current date.")
    region: str = Field(default="eu-west-1", description="AWS region where the bucket is located (e.g., 'eu-west-1').")


class S3ReportWriter(BaseTool):
    name: str = "S3 Report Writer"
    description: str = (
        "Writes analysis reports in markdown format to an S3 bucket with partitioned structure. "
        "Automatically creates partitioned paths: bucket_name/farm_id/date/reports/report_type_timestamp.md "
        "(or bucket_name/farm_id/date/reports/drone/report_type_timestamp.md for drone reports). "
        "Useful for storing agricultural analysis reports, farm operation summaries, "
        "and image interpretation results in a centralized, organized location."
    )
    args_schema: Type[BaseModel] = S3ReportWriterInput

    def _run(
        self,
        bucket_name: str,
        farm_id: str,
        report_content: str,
        report_type: str,
        date: str = None,
        region: str = "eu-west-1",
    ) -> Dict[str, Any]:
        """
        Write a markdown report to S3 bucket with partitioned structure.

        Args:
            bucket_name: Name of the S3 bucket
            farm_id: Farm ID for partitioning
            report_content: The markdown content to write
            report_type: Type of report (vision_analysis, weather_report, sensor_analysis, financial_analysis, compliance_report, master_report, mission_plan, realtime_monitoring, image_analysis, maintenance_prediction)
            date: Date for partitioning in YYYY-MM-DD format (optional, defaults to today)
            region: AWS region (default: eu-west-1)

        Returns:
            Dictionary containing success status, S3 URI, and console URL

        Note:
            Drone reports (mission_plan, realtime_monitoring, image_analysis, maintenance_prediction)
            are stored under reports/drone/ subdirectory for organizational purposes.
        """
        if not BOTO3_AVAILABLE:
            return {
                "success": False,
                "error": "boto3 library not installed",
                "details": "Please install boto3: pip install boto3"
            }

        # Validate report_type
        valid_report_types = ['vision_analysis', 'weather_report', 'sensor_analysis', 'financial_analysis', 'compliance_report', 'master_report', 'mission_plan', 'realtime_monitoring', 'image_analysis', 'maintenance_prediction']
        if report_type not in valid_report_types:
            return {
                "success": False,
                "error": "Invalid report_type",
                "details": f"report_type must be one of: {', '.join(valid_report_types)}"
            }

        # Use current date if not provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename_md = f"{report_type}_{timestamp}.md"
        report_filename_json = f"{report_type}_{timestamp}.json"

        # Define drone report types
        drone_report_types = ['mission_plan', 'realtime_monitoring', 'image_analysis', 'maintenance_prediction']

        # Construct the partitioned S3 keys
        # For drone reports: farm_id/date/reports/drone/report_type_timestamp.{md,json}
        # For other reports: farm_id/date/reports/report_type_timestamp.{md,json}
        if report_type in drone_report_types:
            base_path = f"{farm_id}/{date}/reports/drone"
        else:
            base_path = f"{farm_id}/{date}/reports"

        s3_key = f"{base_path}/{report_filename_md}"
        s3_key_json = f"{base_path}/{report_filename_json}"

        common_metadata = {
            'generated_by': 'smart_farm_flow',
            'farm_id': farm_id,
            'report_type': report_type,
            'date': date,
            'created_at': datetime.now().isoformat()
        }

        try:
            # Initialize S3 client
            s3_client = boto3.client('s3', region_name=region)

            # Upload the markdown report
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=report_content.encode('utf-8'),
                ContentType='text/markdown',
                Metadata=common_metadata,
            )

            # Also upload a JSON version for frontend consumption
            # If content is valid JSON, write it directly; otherwise wrap it
            try:
                json.loads(report_content)
                json_body = report_content
            except (json.JSONDecodeError, ValueError):
                json_body = json.dumps({"raw_content": report_content})

            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key_json,
                Body=json_body.encode('utf-8'),
                ContentType='application/json',
                Metadata=common_metadata,
            )

            # Construct URLs
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            s3_console_url = (
                f"https://s3.console.aws.amazon.com/s3/object/{bucket_name}"
                f"?region={region}&prefix={s3_key}"
            )

            return {
                "success": True,
                "message": "Report successfully written to S3 (md + json)",
                "bucket": bucket_name,
                "key": s3_key,
                "key_json": s3_key_json,
                "farm_id": farm_id,
                "report_type": report_type,
                "date": date,
                "s3_uri": s3_uri,
                "s3_console_url": s3_console_url,
                "region": region
            }

        except NoCredentialsError:
            return {
                "success": False,
                "error": "AWS credentials not found",
                "details": "Please configure AWS credentials using AWS CLI or environment variables"
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            return {
                "success": False,
                "error": f"AWS S3 error: {error_code}",
                "details": error_message
            }
        except Exception as e:
            return {
                "success": False,
                "error": "Unexpected error occurred",
                "details": str(e)
            }
