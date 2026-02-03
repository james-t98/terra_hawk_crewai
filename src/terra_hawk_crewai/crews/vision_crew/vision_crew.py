from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from terra_hawk_crewai.tools import S3ImageRetriever
from typing import List, Tuple, Any
from pydantic import BaseModel, Field
import json
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class ImageAnalysis(BaseModel):
    """Structured output for individual image analysis"""
    Key: str = Field(..., description="The image key/filename")
    Description: str = Field(..., description="Analytical description of the image")
    Classification: str = Field(..., description="Classification of plant health: Healthy or Diseased")
    Confidence: float = Field(..., description="Confidence score between 0.0 and 100.0", ge=0.0, le=100.0)


class ImageAnalyzerResult(BaseModel):
    """Structured output for the image analyzer task"""
    analysis_data: List[ImageAnalysis] = Field(..., description="List of analysis results for each image")


@CrewBase
class VisionCrew():
    """VisionCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    def validate_image_retriever_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """
        Validates the output from the image_retriever_task.
        Ensures the result contains valid JSON with required fields and at least one image.

        Args:
            result: TaskOutput from the image_retriever_task

        Returns:
            Tuple of (is_valid, content_or_error_message)
        """
        try:
            # Parse the JSON output
            data = json.loads(result.raw)

            # Check for required top-level fields
            required_fields = ['bucket_name', 'num_images_found', 'region', 'images']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (False, f"Missing required fields: {', '.join(missing_fields)}. Please ensure all fields are included.")

            # Validate images array
            if not isinstance(data['images'], list):
                return (False, "The 'images' field must be a list. Please return a valid array of images.")

            if len(data['images']) == 0:
                return (False, "No images found in the response. Please ensure the API returned at least one image.")

            # Validate each image has required fields
            required_image_fields = ['key', 'content_type', 's3_uri', 'presigned_url']
            for idx, image in enumerate(data['images']):
                missing_image_fields = [field for field in required_image_fields if field not in image]
                if missing_image_fields:
                    return (False, f"Image at index {idx} is missing fields: {', '.join(missing_image_fields)}. Please ensure all image fields are included.")

            # Validate num_images_found matches actual count
            if data['num_images_found'] != len(data['images']):
                return (False, f"num_images_found ({data['num_images_found']}) does not match actual image count ({len(data['images'])}). Please correct the count.")

            return (True, result.raw)

        except json.JSONDecodeError:
            return (False, "Output is not valid JSON. Please return a properly formatted JSON string.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}. Please check the output format.")

    @agent
    def image_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['image_retriever'], # type: ignore[index]
            verbose=True,
            tools=[S3ImageRetriever()]
        )
    
    @agent
    def image_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['image_analyzer'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def image_retriever_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_retriever_task'], # type: ignore[index]
            tools=[S3ImageRetriever()],
            guardrail=self.validate_image_retriever_output
        )
    
    @task
    def image_analyzer_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_analyzer_task'], # type: ignore[index]
            output_pydantic=ImageAnalyzerResult
        )

    @crew
    def crew(self) -> Crew:
        """Creates the VisionCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
