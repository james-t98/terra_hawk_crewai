#!/usr/bin/env python3
import os
from typing import Type, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import boto3
    from boto3.dynamodb.conditions import Key, Attr
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class DynamoDBVisionRetrieverInput(BaseModel):
    """Input schema for DynamoDBVisionRetriever."""

    farm_id: str = Field(..., description="The farm ID to retrieve vision/detection data for.")
    date: str = Field(default=None, description="Optional date filter in format YYYY-MM-DD. If provided, only returns data for that specific date.")
    limit: int = Field(default=50, description="Number of latest detection records to retrieve (default: 50).", ge=1, le=200)


class DynamoDBVisionRetriever(BaseTool):
    name: str = "DynamoDB Vision Data Retriever"
    description: str = (
        "Retrieves crop detection and vision analysis data from a DynamoDB table. "
        "Returns computer vision model detections including bounding boxes, classifications, "
        "health status, confidence scores, and image URLs. "
        "Useful for analyzing crop health, disease detection, and visual crop monitoring data."
    )
    args_schema: Type[BaseModel] = DynamoDBVisionRetrieverInput

    def _convert_decimal(self, obj: Any) -> Any:
        """Helper to convert Decimal types from DynamoDB to float/int"""
        if isinstance(obj, Decimal):
            return float(obj) if obj % 1 else int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimal(item) for item in obj]
        return obj

    def _parse_detections(self, detections_list: List[Dict]) -> List[Dict[str, Any]]:
        """Parse and format detection data from DynamoDB format"""
        formatted_detections = []

        for detection in detections_list:
            # Extract detection details
            detection_data = detection.get('M', {})

            # Parse bounding box
            bbox = detection_data.get('bbox', {}).get('M', {})
            bbox_formatted = {
                'x': float(bbox.get('x', {}).get('N', 0)),
                'y': float(bbox.get('y', {}).get('N', 0)),
                'width': float(bbox.get('width', {}).get('N', 0)),
                'height': float(bbox.get('height', {}).get('N', 0))
            }

            formatted_detections.append({
                'label': detection_data.get('label', {}).get('S', 'unknown'),
                'class_id': int(detection_data.get('class_id', {}).get('N', 0)),
                'confidence': float(detection_data.get('confidence', {}).get('N', 0)),
                'isHealthy': detection_data.get('isHealthy', {}).get('BOOL', True),
                'bbox': bbox_formatted
            })

        return formatted_detections

    def _run(
        self,
        farm_id: str,
        date: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Retrieve vision/detection data from DynamoDB.

        Args:
            farm_id: The farm ID to query
            date: Optional date filter (YYYY-MM-DD format)
            limit: Number of latest records to retrieve (default: 50)

        Returns:
            Dictionary containing detection records and metadata
        """
        if not BOTO3_AVAILABLE:
            return {
                "success": False,
                "error": "boto3 library not installed",
                "details": "Please install boto3: pip install boto3"
            }

        # Get configuration from environment variables
        table_name = os.environ.get("DYNAMODB_VISION_TABLE_NAME", "terra-hawk-capture-moments")
        region = os.environ.get("AWS_REGION_NAME", "eu-west-1")

        try:
            # Initialize DynamoDB resource
            dynamodb = boto3.resource('dynamodb', region_name=region)
            table = dynamodb.Table(table_name)

            # Build query parameters
            query_params = {
                'KeyConditionExpression': Key('farm_id').eq(farm_id),
                'ScanIndexForward': False,  # Sort descending (newest first)
                'Limit': limit
            }

            # Add date filter if provided
            if date:
                query_params['KeyConditionExpression'] = query_params['KeyConditionExpression'] & Key('timestamp').begins_with(date)

            # Query the table
            response = table.query(**query_params)

            items = response.get('Items', [])

            if not items:
                return {
                    "success": True,
                    "farm_id": farm_id,
                    "date_filter": date,
                    "records_count": 0,
                    "message": f"No vision data found for farm_id: {farm_id}" + (f" on date: {date}" if date else ""),
                    "records": []
                }

            # Convert Decimal types and format data
            formatted_records = []
            for item in items:
                # Convert all Decimal types to native Python types
                converted_item = self._convert_decimal(item)

                # Parse detections if present
                if 'detections' in item and isinstance(item['detections'], dict) and 'L' in item['detections']:
                    converted_item['detections'] = self._parse_detections(item['detections']['L'])

                formatted_records.append(converted_item)

            # Calculate summary statistics
            total_detections = sum(
                len(record.get('detections', []))
                for record in formatted_records
            )

            healthy_count = sum(
                sum(1 for det in record.get('detections', []) if det.get('isHealthy', True))
                for record in formatted_records
            )

            unhealthy_count = total_detections - healthy_count

            # Get unique crop types and fields
            crop_types = set(record.get('crop_name', 'Unknown') for record in formatted_records)
            field_names = set(record.get('field_name', 'Unknown') for record in formatted_records)
            primary_classes = set(record.get('primary_class', 'Unknown') for record in formatted_records)

            return {
                "success": True,
                "farm_id": farm_id,
                "date_filter": date,
                "records_count": len(formatted_records),
                "total_detections": total_detections,
                "healthy_detections": healthy_count,
                "unhealthy_detections": unhealthy_count,
                "latest_timestamp": formatted_records[0].get('timestamp') if formatted_records else None,
                "crop_types": list(crop_types),
                "field_names": list(field_names),
                "detection_classes": list(primary_classes),
                "records": formatted_records
            }

        except Exception as e:
            error_message = str(e)

            # Check for specific error types
            if "ResourceNotFoundException" in error_message:
                return {
                    "success": False,
                    "error": "DynamoDB table not found",
                    "details": f"Table '{table_name}' does not exist in region '{region}'"
                }
            elif "AccessDeniedException" in error_message or "UnrecognizedClientException" in error_message:
                return {
                    "success": False,
                    "error": "AWS credentials error",
                    "details": "Please configure AWS credentials with DynamoDB read permissions"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to retrieve vision data",
                    "details": error_message
                }
