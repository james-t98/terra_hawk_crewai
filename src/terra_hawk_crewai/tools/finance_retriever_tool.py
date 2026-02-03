#!/usr/bin/env python3
import os
from typing import Type, Dict, Any, List
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    import boto3
    from boto3.dynamodb.conditions import Key
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class FinanceRetrieverInput(BaseModel):
    """Input schema for FinanceRetriever."""

    farm_id: str = Field(..., description="The farm ID to retrieve financial data for.")
    days_back: int = Field(default=30, description="Number of days of historical data to retrieve (default: 30 days).", ge=1, le=365)


class FinanceRetriever(BaseTool):
    name: str = "Finance Retriever"
    description: str = (
        "Retrieves financial data from DynamoDB including revenue, expenses, and operational costs. "
        "Returns comprehensive financial metrics for budget analysis, forecasting, and ROI calculations. "
        "Useful for analyzing farm profitability, cost optimization, and financial planning."
    )
    args_schema: Type[BaseModel] = FinanceRetrieverInput

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

    def _query_table(self, table, farm_id: str, start_timestamp: int, date_field: str) -> List[Dict]:
        """Query a DynamoDB table for records within the date range"""
        try:
            response = table.query(
                KeyConditionExpression=Key('farm_id').eq(farm_id) & Key(date_field).gte(start_timestamp),
                ScanIndexForward=False  # Sort descending (newest first)
            )
            return response.get('Items', [])
        except Exception as e:
            print(f"Error querying table: {e}")
            return []

    def _run(
        self,
        farm_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Retrieve financial data from DynamoDB tables.

        Args:
            farm_id: The farm ID to query
            days_back: Number of days of historical data (default: 30)

        Returns:
            Dictionary containing revenue, expenses, and operational costs
        """
        if not BOTO3_AVAILABLE:
            return {
                "success": False,
                "error": "boto3 library not installed",
                "details": "Please install boto3: pip install boto3"
            }

        # Get configuration from environment variables
        region = os.environ.get("AWS_REGION", "eu-west-1")
        revenue_table_name = os.environ.get("REVENUE_TABLE_NAME", "farm-revenue-table")
        expenses_table_name = os.environ.get("EXPENSES_TABLE_NAME", "farm-expenses-table")
        operational_table_name = os.environ.get("OPERATIONAL_TABLE_NAME", "drone-operational-costs-table")

        # Calculate start timestamp
        start_date = datetime.now() - timedelta(days=days_back)
        start_timestamp = int(start_date.timestamp())

        try:
            # Initialize DynamoDB resource
            dynamodb = boto3.resource('dynamodb', region_name=region)

            # Get table references
            revenue_table = dynamodb.Table(revenue_table_name)
            expenses_table = dynamodb.Table(expenses_table_name)
            operational_table = dynamodb.Table(operational_table_name)

            # Query all tables
            revenue_items = self._query_table(revenue_table, farm_id, start_timestamp, 'revenue_date')
            expense_items = self._query_table(expenses_table, farm_id, start_timestamp, 'expense_date')
            operational_items = self._query_table(operational_table, farm_id, start_timestamp, 'operation_date')

            # Convert Decimal types
            revenue_data = [self._convert_decimal(item) for item in revenue_items]
            expense_data = [self._convert_decimal(item) for item in expense_items]
            operational_data = [self._convert_decimal(item) for item in operational_items]

            # Calculate summary statistics
            total_revenue = sum(item.get('amount', 0) for item in revenue_data)
            total_expenses = sum(item.get('amount', 0) for item in expense_data)
            total_operational_costs = sum(
                (item.get('fuel_cost', 0) + item.get('battery_cost', 0) +
                 item.get('maintenance_cost', 0) + item.get('labor_cost', 0) +
                 item.get('materials_cost', 0))
                for item in operational_data
            )

            # Get unique zones from operational data
            zones = list(set(item.get('zone', 'Unknown') for item in operational_data if item.get('zone')))

            return {
                "success": True,
                "farm_id": farm_id,
                "analysis_period_days": days_back,
                "start_date": self._format_timestamp(start_timestamp),
                "end_date": self._format_timestamp(int(datetime.now().timestamp())),
                "summary": {
                    "total_revenue": total_revenue,
                    "total_expenses": total_expenses,
                    "total_operational_costs": total_operational_costs,
                    "total_costs": total_expenses + total_operational_costs,
                    "net_profit": total_revenue - (total_expenses + total_operational_costs),
                    "profit_margin_percentage": ((total_revenue - (total_expenses + total_operational_costs)) / total_revenue * 100) if total_revenue > 0 else 0
                },
                "zones_covered": zones,
                "revenue_records_count": len(revenue_data),
                "expense_records_count": len(expense_data),
                "operational_records_count": len(operational_data),
                "revenue_data": revenue_data,
                "expense_data": expense_data,
                "operational_data": operational_data
            }

        except Exception as e:
            error_message = str(e)

            # Check for specific error types
            if "ResourceNotFoundException" in error_message:
                return {
                    "success": False,
                    "error": "DynamoDB table not found",
                    "details": f"One or more tables do not exist in region '{region}'. Check: {revenue_table_name}, {expenses_table_name}, {operational_table_name}"
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
                    "error": "Failed to retrieve financial data",
                    "details": error_message
                }
