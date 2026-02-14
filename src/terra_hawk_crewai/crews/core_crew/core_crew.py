from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from terra_hawk_crewai.tools import S3ReportReader
from typing import List, Tuple, Any
from pydantic import BaseModel, Field
import json
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class MasterAnalysis(BaseModel):
    """Master analysis data"""
    executive_summary: str = Field(..., description="Brief executive summary (3-5 sentences)")
    critical_alerts: List[str] = Field(..., description="Critical alerts requiring immediate attention")
    vision_summary: str = Field(..., description="Summary of crop health findings from vision analysis")
    weather_summary: str = Field(..., description="Summary of weather conditions and agricultural impact")
    sensor_summary: str = Field(..., description="Summary of soil health and irrigation needs")
    compliance_summary: str = Field(..., description="Summary of compliance status")
    cross_functional_insights: List[str] = Field(..., description="Insights connecting different data sources")
    strategic_recommendations: List[str] = Field(..., description="Prioritized strategic recommendations")
    operational_priorities: List[str] = Field(..., description="Next 24-48 hour priorities by zone")
    overall_farm_status: str = Field(..., description="Overall farm status: Excellent, Good, Attention Needed, or Critical")


class MasterReportResult(BaseModel):
    """Structured output for the master chief task"""
    master_analysis: MasterAnalysis = Field(..., description="Master analysis data")


@CrewBase
class CoreCrew():
    """CoreCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    def validate_master_report_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """
        Validates the output from the master_chief_task.
        Ensures the result contains valid JSON with required fields and proper data types.

        Args:
            result: TaskOutput from the master_chief_task

        Returns:
            Tuple of (is_valid, content_or_error_message)
        """
        try:
            # Parse the JSON output
            data = json.loads(result.raw)

            # Check for required top-level fields
            required_fields = ['master_analysis']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (False, f"Missing required fields: {', '.join(missing_fields)}. Please ensure all fields are included.")

            # Validate master_analysis
            if not isinstance(data.get('master_analysis'), dict):
                return (False, "The 'master_analysis' field must be an object.")

            master_analysis = data['master_analysis']
            required_analysis_fields = ['executive_summary', 'critical_alerts', 'vision_summary', 'weather_summary',
                                       'sensor_summary', 'compliance_summary',
                                       'cross_functional_insights', 'strategic_recommendations',
                                       'operational_priorities', 'overall_farm_status']
            missing_analysis_fields = [field for field in required_analysis_fields if field not in master_analysis]

            if missing_analysis_fields:
                return (False, f"master_analysis is missing fields: {', '.join(missing_analysis_fields)}. Please ensure all fields are included.")

            # Validate list fields
            list_fields = ['critical_alerts', 'cross_functional_insights', 'strategic_recommendations', 'operational_priorities']
            for field in list_fields:
                if not isinstance(master_analysis.get(field), list):
                    return (False, f"The '{field}' field must be an array of strings.")

            # Validate overall_farm_status
            valid_statuses = ['Excellent', 'Good', 'Attention Needed', 'Critical']
            if master_analysis.get('overall_farm_status') not in valid_statuses:
                return (False, f"overall_farm_status must be one of: {', '.join(valid_statuses)}.")

            return (True, result.raw)

        except json.JSONDecodeError:
            return (False, "Output is not valid JSON. Please return a properly formatted JSON string.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}. Please check the output format.")

    @agent
    def master_chief(self) -> Agent:
        return Agent(
            config=self.agents_config['master_chief'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            tools=[S3ReportReader()]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def master_chief_task(self) -> Task:
        return Task(
            config=self.tasks_config['master_chief_task'], # type: ignore[index]
            guardrail=self.validate_master_report_output,
            output_pydantic=MasterReportResult
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the CoreCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
