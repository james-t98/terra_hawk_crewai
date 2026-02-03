#!/usr/bin/env python3
import os
from typing import Type, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import boto3
    from boto3.dynamodb.conditions import Key
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class SensorDataRetrieverInput(BaseModel):
    """Input schema for SensorDataRetriever."""

    farm_id: str = Field(..., description="The farm ID to retrieve sensor data for.")
    limit: int = Field(default=10, description="Number of latest sensor readings to retrieve (default: 10).", ge=1, le=100)


class SensorDataRetriever(BaseTool):
    name: str = "Sensor Data Retriever"
    description: str = (
        "Retrieves the latest IoT sensor readings from a DynamoDB table. "
        "Returns soil moisture, temperature, pH levels, and other agricultural sensor metrics. "
        "Useful for analyzing crop health, irrigation needs, and environmental conditions."
    )
    args_schema: Type[BaseModel] = SensorDataRetrieverInput

    def _convert_decimal(self, obj: Any) -> Any:
        """Helper to convert Decimal types from DynamoDB to float/int"""
        if isinstance(obj, Decimal):
            return float(obj) if obj % 1 else int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimal(item) for item in obj]
        return obj

    def _format_timestamp(self, timestamp: Any) -> str:
        """Convert Unix timestamp to readable datetime string"""
        if isinstance(timestamp, Decimal):
            timestamp = int(timestamp)
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(timestamp)

    def _run(
        self,
        farm_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve latest sensor readings from DynamoDB.

        Args:
            farm_id: The farm ID to query
            limit: Number of latest readings to retrieve (default: 10)

        Returns:
            Dictionary containing sensor readings and metadata
        """
        if not BOTO3_AVAILABLE:
            return {
                "success": False,
                "error": "boto3 library not installed",
                "details": "Please install boto3: pip install boto3"
            }

        # Get configuration from environment variables
        table_name = os.environ.get("DYNAMODB_TABLE_NAME", "SensorData")
        region = os.environ.get("AWS_REGION", "eu-west-1")

        try:
            # Initialize DynamoDB resource
            dynamodb = boto3.resource('dynamodb', region_name=region)
            table = dynamodb.Table(table_name)

            # Query using the FarmIdTimestampIndex GSI
            response = table.query(
                IndexName='FarmIdTimestampIndex',
                KeyConditionExpression=Key('farm_id').eq(farm_id),
                ScanIndexForward=False,  # Sort descending (newest first)
                Limit=limit
            )

            items = response.get('Items', [])

            if not items:
                return {
                    "success": True,
                    "farm_id": farm_id,
                    "readings_count": 0,
                    "message": f"No sensor data found for farm_id: {farm_id}",
                    "readings": []
                }

            # Convert Decimal types and format data
            formatted_readings = []
            for item in items:
                # Convert all Decimal types to native Python types
                converted_item = self._convert_decimal(item)

                # Add formatted timestamp for readability
                if 'timestamp' in converted_item:
                    converted_item['timestamp_formatted'] = self._format_timestamp(converted_item['timestamp'])

                formatted_readings.append(converted_item)

            # Calculate summary statistics
            zones = set(item.get('field_zone', 'Unknown') for item in formatted_readings)
            sensor_types = set(item.get('sensor_type', 'Unknown') for item in formatted_readings)

            return {
                "success": True,
                "farm_id": farm_id,
                "readings_count": len(formatted_readings),
                "latest_timestamp": self._format_timestamp(formatted_readings[0].get('timestamp')) if formatted_readings else None,
                "zones_covered": list(zones),
                "sensor_types": list(sensor_types),
                "readings": formatted_readings
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
                    "error": "Failed to retrieve sensor data",
                    "details": error_message
                }
