from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from terra_hawk_crewai.tools import FinanceRetriever
from typing import List, Tuple, Any
from pydantic import BaseModel, Field
import json
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class FinancialOverview(BaseModel):
    """Financial overview metrics"""
    total_revenue: float = Field(..., description="Total revenue in EUR")
    total_expenses: float = Field(..., description="Total expenses in EUR")
    total_operational_costs: float = Field(..., description="Total operational costs in EUR")
    net_profit: float = Field(..., description="Net profit in EUR")
    profit_margin_percentage: float = Field(..., description="Profit margin as percentage")


class RevenueByType(BaseModel):
    """Revenue breakdown by type"""
    revenue_type: str = Field(..., description="Type of revenue")
    amount: float = Field(..., description="Amount in EUR")
    percentage_of_total: float = Field(..., description="Percentage of total revenue")


class RevenueAnalysis(BaseModel):
    """Revenue analysis"""
    revenue_by_type: List[RevenueByType] = Field(..., description="Revenue breakdown by type")
    top_revenue_sources: List[str] = Field(..., description="Top revenue sources")
    revenue_trend: str = Field(..., description="Revenue trend: Increasing, Stable, or Decreasing")


class ExpenseByCategory(BaseModel):
    """Expense breakdown by category"""
    category: str = Field(..., description="Expense category")
    amount: float = Field(..., description="Amount in EUR")
    percentage_of_total: float = Field(..., description="Percentage of total expenses")


class ExpenseAnalysis(BaseModel):
    """Expense analysis"""
    expense_by_category: List[ExpenseByCategory] = Field(..., description="Expense breakdown by category")
    top_expense_categories: List[str] = Field(..., description="Top expense categories")
    expense_trend: str = Field(..., description="Expense trend: Increasing, Stable, or Decreasing")


class OperationByType(BaseModel):
    """Operation statistics by type"""
    operation_type: str = Field(..., description="Type of operation")
    count: int = Field(..., description="Number of operations")
    total_cost: float = Field(..., description="Total cost in EUR")


class OperationalAnalysis(BaseModel):
    """Operational analysis"""
    total_flight_time_minutes: float = Field(..., description="Total drone flight time in minutes")
    total_area_covered_hectares: float = Field(..., description="Total area covered in hectares")
    average_cost_per_hectare: float = Field(..., description="Average cost per hectare")
    operations_by_type: List[OperationByType] = Field(..., description="Operations breakdown by type")


class ROIByZone(BaseModel):
    """ROI metrics per zone"""
    zone: str = Field(..., description="Zone name")
    revenue: float = Field(..., description="Revenue from zone")
    costs: float = Field(..., description="Total costs for zone")
    roi_percentage: float = Field(..., description="ROI as percentage")
    status: str = Field(..., description="Status: Profitable, Break-even, or Loss-making")


class CashFlowAnalysis(BaseModel):
    """Cash flow analysis"""
    cash_inflow: float = Field(..., description="Total cash inflow")
    cash_outflow: float = Field(..., description="Total cash outflow")
    net_cash_flow: float = Field(..., description="Net cash flow")
    cash_flow_status: str = Field(..., description="Cash flow status: Positive, Neutral, or Negative")


class Forecasts(BaseModel):
    """Financial forecasts"""
    projected_monthly_revenue: float = Field(..., description="Projected revenue for next month")
    projected_monthly_expenses: float = Field(..., description="Projected expenses for next month")
    projected_monthly_profit: float = Field(..., description="Projected profit for next month")


class FinancialAnalysis(BaseModel):
    """Financial analysis data"""
    summary: str = Field(..., description="Brief narrative summary of financial analysis and key findings")
    farm_id: str = Field(..., description="Farm ID that was analyzed")
    analysis_period: str = Field(..., description="Time range analyzed")
    total_days_analyzed: int = Field(..., description="Number of days analyzed", ge=1)
    financial_overview: FinancialOverview = Field(..., description="Financial overview metrics")
    revenue_analysis: RevenueAnalysis = Field(..., description="Revenue analysis")
    expense_analysis: ExpenseAnalysis = Field(..., description="Expense analysis")
    operational_analysis: OperationalAnalysis = Field(..., description="Operational analysis")
    roi_by_zone: List[ROIByZone] = Field(..., description="ROI per zone")
    cash_flow_analysis: CashFlowAnalysis = Field(..., description="Cash flow analysis")
    forecasts: Forecasts = Field(..., description="Financial forecasts")
    recommendations: List[str] = Field(..., description="Actionable financial recommendations")
    alerts: List[str] = Field(..., description="Critical financial issues")


class FinancialAnalysisResult(BaseModel):
    """Structured output for the financial task"""
    financial_analysis: FinancialAnalysis = Field(..., description="Financial analysis data")


@CrewBase
class FinanceCrew():
    """FinanceCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    def validate_financial_analysis_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """
        Validates the output from the financial_task.
        Ensures the result contains valid JSON with required fields and proper data types.

        Args:
            result: TaskOutput from the financial_task

        Returns:
            Tuple of (is_valid, content_or_error_message)
        """
        try:
            # Parse the JSON output
            data = json.loads(result.raw)

            # Check for required top-level fields
            required_fields = ['financial_analysis']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (False, f"Missing required fields: {', '.join(missing_fields)}. Please ensure all fields are included.")

            # Validate financial_analysis
            if not isinstance(data.get('financial_analysis'), dict):
                return (False, "The 'financial_analysis' field must be an object.")

            financial_analysis = data['financial_analysis']
            required_analysis_fields = ['summary', 'farm_id', 'analysis_period', 'total_days_analyzed',
                             'financial_overview', 'revenue_analysis', 'expense_analysis',
                             'operational_analysis', 'roi_by_zone', 'cash_flow_analysis',
                             'forecasts', 'recommendations', 'alerts']
            missing_analysis_fields = [field for field in required_analysis_fields if field not in financial_analysis]

            if missing_analysis_fields:
                return (False, f"financial_analysis is missing fields: {', '.join(missing_analysis_fields)}. Please ensure all fields are included.")

            # Validate financial_overview
            required_overview_fields = ['total_revenue', 'total_expenses', 'total_operational_costs', 'net_profit', 'profit_margin_percentage']
            if not isinstance(financial_analysis.get('financial_overview'), dict):
                return (False, "The 'financial_overview' field must be an object.")

            missing_overview_fields = [field for field in required_overview_fields if field not in financial_analysis['financial_overview']]
            if missing_overview_fields:
                return (False, f"financial_overview is missing fields: {', '.join(missing_overview_fields)}.")

            # Validate revenue_analysis
            if not isinstance(financial_analysis.get('revenue_analysis'), dict):
                return (False, "The 'revenue_analysis' field must be an object.")

            if not isinstance(financial_analysis['revenue_analysis'].get('revenue_by_type'), list):
                return (False, "revenue_analysis.revenue_by_type must be an array.")

            # Validate expense_analysis
            if not isinstance(financial_analysis.get('expense_analysis'), dict):
                return (False, "The 'expense_analysis' field must be an object.")

            if not isinstance(financial_analysis['expense_analysis'].get('expense_by_category'), list):
                return (False, "expense_analysis.expense_by_category must be an array.")

            # Validate operational_analysis
            if not isinstance(financial_analysis.get('operational_analysis'), dict):
                return (False, "The 'operational_analysis' field must be an object.")

            if not isinstance(financial_analysis['operational_analysis'].get('operations_by_type'), list):
                return (False, "operational_analysis.operations_by_type must be an array.")

            # Validate roi_by_zone
            if not isinstance(financial_analysis.get('roi_by_zone'), list):
                return (False, "The 'roi_by_zone' field must be an array.")

            # Validate each ROI entry
            required_roi_fields = ['zone', 'revenue', 'costs', 'roi_percentage', 'status']
            for idx, roi in enumerate(financial_analysis['roi_by_zone']):
                if not isinstance(roi, dict):
                    return (False, f"roi_by_zone[{idx}] must be an object.")

                missing_roi_fields = [field for field in required_roi_fields if field not in roi]
                if missing_roi_fields:
                    return (False, f"roi_by_zone[{idx}] is missing fields: {', '.join(missing_roi_fields)}.")

            # Validate cash_flow_analysis
            required_cash_flow_fields = ['cash_inflow', 'cash_outflow', 'net_cash_flow', 'cash_flow_status']
            if not isinstance(financial_analysis.get('cash_flow_analysis'), dict):
                return (False, "The 'cash_flow_analysis' field must be an object.")

            missing_cash_flow_fields = [field for field in required_cash_flow_fields if field not in financial_analysis['cash_flow_analysis']]
            if missing_cash_flow_fields:
                return (False, f"cash_flow_analysis is missing fields: {', '.join(missing_cash_flow_fields)}.")

            # Validate forecasts
            required_forecast_fields = ['projected_monthly_revenue', 'projected_monthly_expenses', 'projected_monthly_profit']
            if not isinstance(financial_analysis.get('forecasts'), dict):
                return (False, "The 'forecasts' field must be an object.")

            missing_forecast_fields = [field for field in required_forecast_fields if field not in financial_analysis['forecasts']]
            if missing_forecast_fields:
                return (False, f"forecasts is missing fields: {', '.join(missing_forecast_fields)}.")

            # Validate recommendations and alerts are lists
            if not isinstance(financial_analysis.get('recommendations'), list):
                return (False, "The 'recommendations' field must be an array of strings.")

            if not isinstance(financial_analysis.get('alerts'), list):
                return (False, "The 'alerts' field must be an array of strings.")

            return (True, result.raw)

        except json.JSONDecodeError:
            return (False, "Output is not valid JSON. Please return a properly formatted JSON string.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}. Please check the output format.")

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst'], # type: ignore[index]
            verbose=True,
            tools=[FinanceRetriever()]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def financial_task(self) -> Task:
        return Task(
            config=self.tasks_config['financial_task'], # type: ignore[index]
            guardrail=self.validate_financial_analysis_output,
            output_pydantic=FinancialAnalysisResult,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FinanceCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
