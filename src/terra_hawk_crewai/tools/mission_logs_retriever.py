#!/usr/bin/env python3
import os
from typing import Type, Dict, Any, Optional, List
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import boto3
    from boto3.dynamodb.conditions import Key, Attr
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class MissionLogsRetrieverInput(BaseModel):
    """Input schema for MissionLogsRetriever."""

    farm_id: Optional[str] = Field(None, description="Farm ID to retrieve missions for.")
    drone_id: Optional[str] = Field(None, description="Specific drone ID to query missions for.")
    mission_type: Optional[str] = Field(None, description="Filter by mission type: surveillance, spraying, mapping, or inspection.")
    mission_status: Optional[str] = Field(None, description="Filter by status: planned, in_progress, completed, failed, or aborted.")
    days_back: int = Field(default=30, description="Number of days to look back (default: 30).", ge=1, le=365)
    limit: int = Field(default=50, description="Maximum number of missions to retrieve (default: 50).", ge=1, le=500)


class MissionLogsRetriever(BaseTool):
    name: str = "Mission Logs Retriever"
    description: str = (
        "Retrieves drone mission logs including surveillance, spraying, mapping, and inspection missions. "
        "Returns detailed mission data: flight paths, coverage areas, images captured, autonomous operations, "
        "costs, and detection summaries. Can filter by farm, drone, mission type, and status. "
        "Essential for operational analysis, mission planning, and fleet performance evaluation."
    )
    args_schema: Type[BaseModel] = MissionLogsRetrieverInput

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
        """Convert Unix timestamp (milliseconds) to readable datetime string"""
        if isinstance(timestamp, Decimal):
            timestamp = float(timestamp)
        try:
            return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(timestamp)

    def _calculate_mission_duration(self, start: Any, end: Any) -> Optional[str]:
        """Calculate mission duration in human-readable format"""
        try:
            start_ms = float(start) if isinstance(start, Decimal) else start
            end_ms = float(end) if isinstance(end, Decimal) else end
            duration_sec = (end_ms - start_ms) / 1000

            hours = int(duration_sec // 3600)
            minutes = int((duration_sec % 3600) // 60)
            seconds = int(duration_sec % 60)

            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except:
            return None

    def _calculate_mission_efficiency(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate mission efficiency metrics"""
        efficiency = {}

        planned_area = mission.get('planned_area_hectares', 0)
        actual_area = mission.get('actual_area_hectares', 0)

        if planned_area > 0:
            coverage_ratio = (actual_area / planned_area) * 100
            efficiency['coverage_percentage'] = round(coverage_ratio, 2)
            efficiency['area_variance_hectares'] = round(actual_area - planned_area, 2)

        completeness = mission.get('coverage_completeness', 0)
        efficiency['completeness_percentage'] = round(completeness, 2)

        autonomous_pct = mission.get('autonomous_percentage', 0)
        efficiency['autonomous_percentage'] = round(autonomous_pct, 2)

        manual_interventions = mission.get('manual_interventions', 0)
        efficiency['manual_interventions'] = manual_interventions

        if manual_interventions > 0:
            efficiency['intervention_level'] = "HIGH" if manual_interventions > 5 else "MODERATE"
        else:
            efficiency['intervention_level'] = "NONE"

        return efficiency

    def _run(
        self,
        farm_id: Optional[str] = None,
        drone_id: Optional[str] = None,
        mission_type: Optional[str] = None,
        mission_status: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Retrieve mission logs from DynamoDB.

        Args:
            farm_id: Filter by farm ID
            drone_id: Filter by specific drone
            mission_type: Filter by mission type
            mission_status: Filter by mission status
            days_back: Days to look back
            limit: Maximum missions to retrieve

        Returns:
            Dictionary containing mission logs and analytics
        """
        if not BOTO3_AVAILABLE:
            return {
                "success": False,
                "error": "boto3 library not installed",
                "details": "Please install boto3: pip install boto3"
            }

        # Get configuration from environment variables
        table_name = os.environ.get("MISSION_LOGS_TABLE", "MissionLogs")
        region = os.environ.get("AWS_REGION", "eu-west-1")

        try:
            # Initialize DynamoDB resource
            dynamodb = boto3.resource('dynamodb', region_name=region)
            table = dynamodb.Table(table_name)

            # Calculate timestamp threshold
            threshold_time = datetime.now() - timedelta(days=days_back)
            threshold_timestamp = int(threshold_time.timestamp() * 1000)

            # Determine which index/query to use
            if mission_status:
                # Use MissionStatusIndex GSI
                key_condition = Key('mission_status').eq(mission_status) & Key('mission_start_timestamp').gte(threshold_timestamp)
                response = table.query(
                    IndexName='MissionStatusIndex',
                    KeyConditionExpression=key_condition,
                    ScanIndexForward=False,
                    Limit=limit
                )
            elif farm_id and mission_type:
                # Use FarmIdMissionTypeIndex GSI
                key_condition = Key('farm_id').eq(farm_id) & Key('mission_type').eq(mission_type)
                response = table.query(
                    IndexName='FarmIdMissionTypeIndex',
                    KeyConditionExpression=key_condition,
                    ScanIndexForward=False,
                    Limit=limit
                )
            elif drone_id:
                # Use primary key query
                key_condition = Key('drone_id').eq(drone_id) & Key('mission_start_timestamp').gte(threshold_timestamp)
                response = table.query(
                    KeyConditionExpression=key_condition,
                    ScanIndexForward=False,
                    Limit=limit
                )
            elif farm_id:
                # Scan with farm_id filter (less efficient but necessary)
                response = table.scan(
                    FilterExpression=Attr('farm_id').eq(farm_id) & Attr('mission_start_timestamp').gte(threshold_timestamp),
                    Limit=limit
                )
            else:
                return {
                    "success": False,
                    "error": "Insufficient query parameters",
                    "details": "At least one of farm_id, drone_id, or mission_status must be provided"
                }

            items = response.get('Items', [])

            # Apply additional filters if needed
            if mission_type and not (farm_id and mission_type):
                items = [item for item in items if item.get('mission_type') == mission_type]

            if mission_status and not mission_status:
                items = [item for item in items if item.get('mission_status') == mission_status]

            if not items:
                return {
                    "success": True,
                    "query_params": {
                        "farm_id": farm_id,
                        "drone_id": drone_id,
                        "mission_type": mission_type,
                        "mission_status": mission_status,
                        "days_back": days_back
                    },
                    "missions_count": 0,
                    "message": "No missions found matching the specified criteria",
                    "missions": []
                }

            # Process missions
            formatted_missions = []
            analytics = {
                "by_type": {},
                "by_status": {},
                "by_drone": {},
                "total_area_hectares": 0,
                "total_images": 0,
                "total_distance_km": 0
            }

            for item in items:
                converted_item = self._convert_decimal(item)

                # Format timestamps
                if 'mission_start_timestamp' in converted_item:
                    converted_item['mission_start_formatted'] = self._format_timestamp(converted_item['mission_start_timestamp'])
                if 'mission_end_timestamp' in converted_item:
                    converted_item['mission_end_formatted'] = self._format_timestamp(converted_item['mission_end_timestamp'])
                    converted_item['mission_duration'] = self._calculate_mission_duration(
                        converted_item['mission_start_timestamp'],
                        converted_item['mission_end_timestamp']
                    )

                # Calculate efficiency metrics
                converted_item['efficiency_metrics'] = self._calculate_mission_efficiency(converted_item)

                formatted_missions.append(converted_item)

                # Build analytics
                m_type = converted_item.get('mission_type', 'unknown')
                m_status = converted_item.get('mission_status', 'unknown')
                m_drone = converted_item.get('drone_id', 'unknown')

                analytics["by_type"][m_type] = analytics["by_type"].get(m_type, 0) + 1
                analytics["by_status"][m_status] = analytics["by_status"].get(m_status, 0) + 1
                analytics["by_drone"][m_drone] = analytics["by_drone"].get(m_drone, 0) + 1

                analytics["total_area_hectares"] += converted_item.get('actual_area_hectares', 0)
                analytics["total_images"] += converted_item.get('images_captured', 0)
                analytics["total_distance_km"] += converted_item.get('total_distance_km', 0)

            # Round analytics totals
            analytics["total_area_hectares"] = round(analytics["total_area_hectares"], 2)
            analytics["total_distance_km"] = round(analytics["total_distance_km"], 2)

            return {
                "success": True,
                "query_params": {
                    "farm_id": farm_id,
                    "drone_id": drone_id,
                    "mission_type": mission_type,
                    "mission_status": mission_status,
                    "days_back": days_back,
                    "limit": limit
                },
                "missions_count": len(formatted_missions),
                "time_range": {
                    "from": threshold_time.strftime('%Y-%m-%d'),
                    "to": datetime.now().strftime('%Y-%m-%d')
                },
                "analytics": analytics,
                "missions": formatted_missions
            }

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
                    "error": "Failed to retrieve mission logs",
                    "details": error_message
                }
