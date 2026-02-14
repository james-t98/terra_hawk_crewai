from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Tuple, Any
from pydantic import BaseModel, Field
from pathlib import Path
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


class EUAIActActionItem(BaseModel):
    """Action item for EU AI Act compliance"""
    priority: str = Field(..., description="Priority: High, Medium, or Low")
    action: str = Field(..., description="Description of action needed")
    deadline_category: str = Field(..., description="Deadline category: immediate, Q1_2026, Q2_2026, or ongoing")


class EUAIActAssessment(BaseModel):
    """EU AI Act compliance assessment"""
    summary: str = Field(..., description="Brief assessment summary")
    system_classification: str = Field(..., description="Risk level classification with justification")
    risk_level: str = Field(..., description="Risk level: minimal_risk, limited_risk, or high_risk")
    transparency_obligations: List[str] = Field(..., description="List of transparency requirements")
    human_oversight_requirements: List[str] = Field(..., description="List of human oversight measures needed")
    data_governance_requirements: List[str] = Field(..., description="List of data governance actions")
    documentation_requirements: List[str] = Field(..., description="List of technical documentation needs")
    logging_requirements: List[str] = Field(..., description="List of record-keeping requirements")
    security_requirements: List[str] = Field(..., description="List of accuracy/robustness/cybersecurity needs")
    compliance_gaps: List[str] = Field(..., description="Current gaps in compliance")
    action_items: List[EUAIActActionItem] = Field(..., description="List of action items")
    overall_compliance_status: str = Field(..., description="Compliant, Partially Compliant, or Non-Compliant")


class EUAIActAssessmentResult(BaseModel):
    """Structured output for the EU AI Act task"""
    eu_ai_act_assessment: EUAIActAssessment = Field(..., description="EU AI Act compliance assessment")


class CombinedComplianceResult(BaseModel):
    """Combined output from all compliance analyses"""
    compliance_analysis: ComplianceAnalysis = Field(..., description="Nitrogen emissions compliance analysis")
    eu_ai_act_assessment: EUAIActAssessment = Field(..., description="EU AI Act compliance assessment")


# --- EU AI Act Cache ---
EU_AI_ACT_CACHE_DIR = Path.home() / ".terra_hawk_cache"
EU_AI_ACT_CACHE_FILE = EU_AI_ACT_CACHE_DIR / "eu_ai_act_assessment.json"
EU_AI_ACT_CACHE_TTL = 7 * 24 * 3600  # 7 days — assessment rarely changes


def _get_cached_eu_ai_act() -> str | None:
    """Return cached EU AI Act assessment JSON string if fresh, else None."""
    import time
    if EU_AI_ACT_CACHE_FILE.exists():
        try:
            data = json.loads(EU_AI_ACT_CACHE_FILE.read_text())
            if time.time() - data.get("timestamp", 0) < EU_AI_ACT_CACHE_TTL:
                return data["result"]
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _set_cached_eu_ai_act(result_json: str):
    """Cache an EU AI Act assessment result."""
    import time
    EU_AI_ACT_CACHE_DIR.mkdir(exist_ok=True)
    EU_AI_ACT_CACHE_FILE.write_text(json.dumps({
        "timestamp": time.time(),
        "result": result_json,
    }))


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

    def validate_eu_ai_act_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """Validates the output from the eu_ai_act_task."""
        try:
            data = json.loads(result.raw)

            if 'eu_ai_act_assessment' not in data:
                return (False, "Missing required 'eu_ai_act_assessment' field.")

            if not isinstance(data.get('eu_ai_act_assessment'), dict):
                return (False, "The 'eu_ai_act_assessment' field must be an object.")

            assessment = data['eu_ai_act_assessment']
            required_fields = ['summary', 'system_classification', 'risk_level',
                             'transparency_obligations', 'human_oversight_requirements',
                             'data_governance_requirements', 'documentation_requirements',
                             'logging_requirements', 'security_requirements',
                             'compliance_gaps', 'action_items', 'overall_compliance_status']
            missing = [f for f in required_fields if f not in assessment]
            if missing:
                return (False, f"eu_ai_act_assessment is missing fields: {', '.join(missing)}.")

            valid_risk_levels = ['minimal_risk', 'limited_risk', 'high_risk']
            if assessment.get('risk_level') not in valid_risk_levels:
                return (False, f"risk_level must be one of: {', '.join(valid_risk_levels)}.")

            valid_statuses = ['Compliant', 'Partially Compliant', 'Non-Compliant']
            if assessment.get('overall_compliance_status') not in valid_statuses:
                return (False, f"overall_compliance_status must be one of: {', '.join(valid_statuses)}.")

            for field in ['transparency_obligations', 'human_oversight_requirements',
                         'data_governance_requirements', 'documentation_requirements',
                         'logging_requirements', 'security_requirements', 'compliance_gaps', 'action_items']:
                if not isinstance(assessment.get(field), list):
                    return (False, f"{field} must be an array.")

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

    @agent
    def eu_ai_act_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['eu_ai_act_analyst'], # type: ignore[index]
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
            output_pydantic=ComplianceAnalysisResult,
            async_execution=True
        )

    @task
    def eu_ai_act_task(self) -> Task:
        return Task(
            config=self.tasks_config['eu_ai_act_task'], # type: ignore[index]
            guardrail=self.validate_eu_ai_act_output,
            output_pydantic=EUAIActAssessmentResult,
            async_execution=True,
            callback=self._cache_eu_ai_act_result
        )

    def _cache_eu_ai_act_result(self, result: TaskOutput):
        """Cache the EU AI Act assessment after successful completion."""
        try:
            # Validate it's good JSON before caching
            json.loads(result.raw)
            _set_cached_eu_ai_act(result.raw)
        except (json.JSONDecodeError, Exception):
            pass

    def validate_combined_compliance_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """Validates the combined compliance output."""
        try:
            data = json.loads(result.raw)
            for field in ['compliance_analysis', 'eu_ai_act_assessment']:
                if field not in data:
                    return (False, f"Missing required field: '{field}'.")
                if not isinstance(data[field], dict):
                    return (False, f"'{field}' must be an object.")
            return (True, result.raw)
        except json.JSONDecodeError:
            return (False, "Output is not valid JSON.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}")

    @agent
    def compliance_aggregator(self) -> Agent:
        return Agent(
            config=self.agents_config['compliance_aggregator'], # type: ignore[index]
            verbose=True
        )

    @task
    def compliance_aggregation_task(self) -> Task:
        return Task(
            config=self.tasks_config['compliance_aggregation_task'], # type: ignore[index]
            guardrail=self.validate_combined_compliance_output,
            output_pydantic=CombinedComplianceResult,
            context=[self.compliance_task(), self.eu_ai_act_task()]
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
