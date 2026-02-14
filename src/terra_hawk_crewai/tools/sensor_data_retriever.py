#!/usr/bin/env python3
import os
import time
from typing import Type, List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
from collections import defaultdict

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import boto3
    from boto3.dynamodb.conditions import Key, Attr
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class SensorDataRetrieverInput(BaseModel):
    """Input schema for SensorDataRetriever."""

    farm_id: str = Field(..., description="The farm ID to retrieve sensor data for.")
    limit: int = Field(default=10, description="Number of latest sensor readings to retrieve (default: 10).", ge=1, le=100)
    date: Optional[str] = Field(default=None, description="Optional date filter in YYYY-MM-DD format. Only returns readings from that date.")
    aggregate: bool = Field(default=False, description="If True, return per-zone aggregated statistics (avg, min, max) instead of raw readings.")


class SensorDataRetriever(BaseTool):
    name: str = "Sensor Data Retriever"
    description: str = (
        "Retrieves the latest IoT sensor readings from a DynamoDB table. "
        "Returns soil moisture, temperature, pH levels, and other agricultural sensor metrics. "
        "Supports date filtering (YYYY-MM-DD) and per-zone aggregation (avg/min/max). "
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
        except Exception:
            return str(timestamp)

    def _aggregate_by_zone(self, readings: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate sensor readings per zone, computing avg/min/max for numeric fields.

        Returns:
            Dictionary with per-zone aggregated statistics.
        """
        zone_data: Dict[str, List[Dict]] = defaultdict(list)
        for r in readings:
            zone = r.get('field_zone', 'Unknown')
            zone_data[zone].append(r)

        # Numeric fields we want to aggregate
        numeric_fields = ['soil_moisture', 'temperature', 'ph', 'humidity',
                          'nitrogen_level', 'phosphorus_level', 'potassium_level']

        aggregated = {}
        for zone, records in zone_data.items():
            zone_stats: Dict[str, Any] = {
                'zone': zone,
                'readings_count': len(records),
            }

            for field in numeric_fields:
                values = [r[field] for r in records if field in r and isinstance(r[field], (int, float))]
                if values:
                    zone_stats[field] = {
                        'avg': round(sum(values) / len(values), 2),
                        'min': round(min(values), 2),
                        'max': round(max(values), 2),
                        'count': len(values),
                    }

            # Time range
            timestamps = [r.get('timestamp') for r in records if r.get('timestamp')]
            if timestamps:
                zone_stats['time_range'] = {
                    'earliest': self._format_timestamp(min(timestamps)),
                    'latest': self._format_timestamp(max(timestamps)),
                }

            # Sensor types present
            zone_stats['sensor_types'] = list(set(r.get('sensor_type', 'Unknown') for r in records))

            aggregated[zone] = zone_stats

        return aggregated

    def _run(
        self,
        farm_id: str,
        limit: int = 10,
        date: str = None,
        aggregate: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve latest sensor readings from DynamoDB.

        Args:
            farm_id: The farm ID to query
            limit: Number of latest readings to retrieve (default: 10)
            date: Optional date filter (YYYY-MM-DD format)
            aggregate: If True, return per-zone aggregated stats instead of raw readings

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

            # Build query parameters
            key_condition = Key('farm_id').eq(farm_id)

            # Add date filter if provided
            if date:
                key_condition = key_condition & Key('timestamp').begins_with(date)

            query_params = {
                'IndexName': 'FarmIdTimestampIndex',
                'KeyConditionExpression': key_condition,
                'ScanIndexForward': False,  # Sort descending (newest first)
                'Limit': limit,
            }

            # Query with retry
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = table.query(**query_params)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt
                        time.sleep(delay)
                    else:
                        raise

            items = response.get('Items', [])

            if not items:
                return {
                    "success": True,
                    "farm_id": farm_id,
                    "date_filter": date,
                    "readings_count": 0,
                    "message": f"No sensor data found for farm_id: {farm_id}" + (f" on date: {date}" if date else ""),
                    "readings": []
                }

            # Convert Decimal types and format data
            formatted_readings = []
            for item in items:
                converted_item = self._convert_decimal(item)
                if 'timestamp' in converted_item:
                    converted_item['timestamp_formatted'] = self._format_timestamp(converted_item['timestamp'])
                formatted_readings.append(converted_item)

            # Calculate summary statistics
            zones = set(item.get('field_zone', 'Unknown') for item in formatted_readings)
            sensor_types = set(item.get('sensor_type', 'Unknown') for item in formatted_readings)

            result = {
                "success": True,
                "farm_id": farm_id,
                "date_filter": date,
                "readings_count": len(formatted_readings),
                "latest_timestamp": self._format_timestamp(formatted_readings[0].get('timestamp')) if formatted_readings else None,
                "zones_covered": list(zones),
                "sensor_types": list(sensor_types),
            }

            if aggregate:
                result["aggregation"] = self._aggregate_by_zone(formatted_readings)
                result["readings"] = []  # Don't send raw readings when aggregated
            else:
                result["readings"] = formatted_readings

            return result

        except Exception as e:
            error_message = str(e)

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
