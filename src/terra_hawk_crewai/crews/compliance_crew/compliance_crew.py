from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Tuple, Any
from pydantic import BaseModel, Field
import json
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class ComplianceAnalysis(BaseModel):
    """Compliance analysis data"""
    summary: str = Field(..., description="Brief summary of compliance analysis")
    topic: str = Field(..., description="Topic analyzed")
    date: str = Field(..., description="Date of analysis")
    classification: str = Field(..., description="Classification: Positive, Neutral, or Negative")
    operational_impact: str = Field(..., description="Impact on current agricultural processes")
    nitrogen_emissions_relevance: str = Field(..., description="Relevance to nitrogen emissions")
    recommendations: List[str] = Field(..., description="Recommended actions or considerations")


class ComplianceAnalysisResult(BaseModel):
    """Structured output for the compliance task"""
    compliance_analysis: ComplianceAnalysis = Field(..., description="Compliance analysis data")


@CrewBase
class ComplianceCrew():
    """ComplianceCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    def validate_compliance_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """
        Validates the output from the compliance_task.
        Ensures the result contains valid JSON with required fields and proper data types.

        Args:
            result: TaskOutput from the compliance_task

        Returns:
            Tuple of (is_valid, content_or_error_message)
        """
        try:
            # Parse the JSON output
            data = json.loads(result.raw)

            # Check for required top-level fields
            required_fields = ['compliance_analysis']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (False, f"Missing required fields: {', '.join(missing_fields)}. Please ensure all fields are included.")

            # Validate compliance_analysis
            if not isinstance(data.get('compliance_analysis'), dict):
                return (False, "The 'compliance_analysis' field must be an object.")

            compliance_analysis = data['compliance_analysis']
            required_analysis_fields = ['summary', 'topic', 'date', 'classification',
                             'operational_impact', 'nitrogen_emissions_relevance', 'recommendations']
            missing_analysis_fields = [field for field in required_analysis_fields if field not in compliance_analysis]

            if missing_analysis_fields:
                return (False, f"compliance_analysis is missing fields: {', '.join(missing_analysis_fields)}. Please ensure all fields are included.")

            # Validate classification
            valid_classifications = ['Positive', 'Neutral', 'Negative']
            if compliance_analysis.get('classification') not in valid_classifications:
                return (False, f"classification must be one of: {', '.join(valid_classifications)}.")

            # Validate recommendations is a list
            if not isinstance(compliance_analysis.get('recommendations'), list):
                return (False, "The 'recommendations' field must be an array of strings.")

            return (True, result.raw)

        except json.JSONDecodeError:
            return (False, "Output is not valid JSON. Please return a properly formatted JSON string.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}. Please check the output format.")

    @agent
    def compliance_officer(self) -> Agent:
        return Agent(
            config=self.agents_config['compliance_officer'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=2
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def compliance_task(self) -> Task:
        return Task(
            config=self.tasks_config['compliance_task'], # type: ignore[index]
            guardrail=self.validate_compliance_output,
            output_pydantic=ComplianceAnalysisResult
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ComplianceCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
