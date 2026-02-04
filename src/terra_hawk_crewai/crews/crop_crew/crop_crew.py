from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from terra_hawk_crewai.tools import WeatherAPITool, SensorDataRetriever, S3ImageRetriever
from typing import List, Tuple, Any
from pydantic import BaseModel, Field
import json
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class WeatherData(BaseModel):
    """Weather metrics"""
    temperature_c: float = Field(..., description="Current temperature in Celsius")
    feels_like_c: float = Field(..., description="Feels like temperature in Celsius")
    condition: str = Field(..., description="Weather condition description")
    humidity: int = Field(..., description="Humidity percentage", ge=0, le=100)
    wind_kph: float = Field(..., description="Wind speed in km/h", ge=0)


class AirQuality(BaseModel):
    """Air quality metrics"""
    aqi: int = Field(..., description="Air Quality Index (US EPA scale)", ge=1, le=6)
    pm2_5: float = Field(..., description="PM2.5 concentration")
    pm10: float = Field(..., description="PM10 concentration")
    o3: float = Field(..., description="Ozone concentration")
    no2: float = Field(..., description="Nitrogen dioxide concentration")
    so2: float = Field(..., description="Sulfur dioxide concentration")
    co: float = Field(..., description="Carbon monoxide concentration")


class AgriculturalAssessment(BaseModel):
    """Agricultural operations assessment"""
    flight_clearance_status: bool = Field(..., description="Safe for drone operations")
    disease_risk_level: str = Field(..., description="Disease risk level: Low, Medium, or High")
    disease_risk_percentage: float = Field(..., description="Disease risk percentage", ge=0, le=100)
    nitrogen_volatilization_risk: str = Field(..., description="Nitrogen volatilization risk: Low, Medium, or High")
    optimal_operations: List[str] = Field(..., description="List of recommended agricultural operations with timing")


class WeatherAnalysis(BaseModel):
    """Weather analysis data"""
    summary: str = Field(..., description="Brief narrative summary of weather conditions and agricultural impact")
    location: str = Field(..., description="Location for which weather data was retrieved")
    date: str = Field(..., description="Date of the weather forecast")
    weather_data: WeatherData = Field(..., description="Current weather metrics")
    air_quality: AirQuality = Field(..., description="Air quality metrics")
    agricultural_assessment: AgriculturalAssessment = Field(..., description="Agricultural operations assessment")


class WeatherReportResult(BaseModel):
    """Structured output for the weather task"""
    weather_analysis: WeatherAnalysis = Field(..., description="Weather analysis data")


class SoilHealthMetric(BaseModel):
    """Soil health metrics per zone"""
    zone_name: str = Field(..., description="Field zone name")
    average_moisture: float = Field(..., description="Average moisture percentage", ge=0, le=100)
    average_temperature: float = Field(..., description="Average temperature in Celsius")
    average_ph: float = Field(..., description="Average pH level", ge=0, le=14)
    status: str = Field(..., description="Status: Good, Attention Needed, or Critical")


class IrrigationRecommendation(BaseModel):
    """Irrigation recommendation per zone"""
    zone: str = Field(..., description="Zone name")
    action: str = Field(..., description="Action: Increase, Maintain, or Reduce")
    priority: str = Field(..., description="Priority: High, Medium, or Low")
    reasoning: str = Field(..., description="Explanation for the recommendation")


class SensorAnalysis(BaseModel):
    """Sensor analysis data"""
    summary: str = Field(..., description="Brief narrative summary of sensor analysis and key findings")
    farm_id: str = Field(..., description="Farm ID that was analyzed")
    readings_count: int = Field(..., description="Total number of sensor readings analyzed", ge=0)
    analysis_period: str = Field(..., description="Time range of sensor data analyzed")
    zones_analyzed: List[str] = Field(..., description="List of field zones covered")
    soil_health_metrics: List[SoilHealthMetric] = Field(..., description="Per-zone soil health analysis")
    irrigation_recommendations: List[IrrigationRecommendation] = Field(..., description="Irrigation recommendations per zone")
    environmental_correlations: List[str] = Field(..., description="Insights linking weather and sensor data")
    alerts: List[str] = Field(..., description="Critical issues requiring immediate attention")


class SensorAnalysisResult(BaseModel):
    """Structured output for the crop sensor task"""
    sensor_analysis: SensorAnalysis = Field(..., description="Sensor analysis data")

class ImageAnalysis(BaseModel):
    """Structured output for individual image analysis"""
    Key: str = Field(..., description="The image key/filename")
    Description: str = Field(..., description="Analytical description of the image")
    Classification: str = Field(..., description="Classification of plant health: Healthy or Diseased")
    Confidence: float = Field(..., description="Confidence score between 0.0 and 100.0", ge=0.0, le=100.0)


class ImageAnalyzerResult(BaseModel):
    """Structured output for the image analyzer task"""
    analysis_data: List[ImageAnalysis] = Field(..., description="List of analysis results for each image")


class VisionAnalysis(BaseModel):
    """Vision analysis from image processing"""
    analysis_data: List[ImageAnalysis] = Field(..., description="List of analysis results for each image")


class CombinedCropAnalysis(BaseModel):
    """Combined output from all crop crew analyses"""
    weather_analysis: WeatherAnalysis = Field(..., description="Weather analysis data")
    sensor_analysis: SensorAnalysis = Field(..., description="Sensor analysis data")
    vision_analysis: VisionAnalysis = Field(..., description="Vision/image analysis data")

@CrewBase
class CropCrew():
    """CropCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    def validate_weather_report_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """
        Validates the output from the weather_task.
        Ensures the result contains valid JSON with required fields and proper data types.

        Args:
            result: TaskOutput from the weather_task

        Returns:
            Tuple of (is_valid, content_or_error_message)
        """
        try:
            # Parse the JSON output
            data = json.loads(result.raw)

            # Check for required top-level fields
            required_fields = ['weather_analysis']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (False, f"Missing required fields: {', '.join(missing_fields)}. Please ensure all fields are included.")

            # Validate weather_analysis
            if not isinstance(data.get('weather_analysis'), dict):
                return (False, "The 'weather_analysis' field must be an object.")

            weather_analysis = data['weather_analysis']
            required_analysis_fields = ['summary', 'location', 'date', 'weather_data', 'air_quality', 'agricultural_assessment']
            missing_analysis_fields = [field for field in required_analysis_fields if field not in weather_analysis]

            if missing_analysis_fields:
                return (False, f"weather_analysis is missing fields: {', '.join(missing_analysis_fields)}. Please ensure all fields are included.")

            # Validate weather_data fields
            required_weather_fields = ['temperature_c', 'feels_like_c', 'condition', 'humidity', 'wind_kph']
            if not isinstance(weather_analysis.get('weather_data'), dict):
                return (False, "The 'weather_data' field must be an object. Please return valid weather data.")

            missing_weather_fields = [field for field in required_weather_fields if field not in weather_analysis['weather_data']]
            if missing_weather_fields:
                return (False, f"weather_data is missing fields: {', '.join(missing_weather_fields)}. Please ensure all weather fields are included.")

            # Validate air_quality fields
            required_air_quality_fields = ['aqi', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
            if not isinstance(weather_analysis.get('air_quality'), dict):
                return (False, "The 'air_quality' field must be an object. Please return valid air quality data.")

            missing_air_quality_fields = [field for field in required_air_quality_fields if field not in weather_analysis['air_quality']]
            if missing_air_quality_fields:
                return (False, f"air_quality is missing fields: {', '.join(missing_air_quality_fields)}. Please ensure all air quality fields are included.")

            # Validate agricultural_assessment fields
            required_assessment_fields = ['flight_clearance_status', 'disease_risk_level', 'disease_risk_percentage', 'nitrogen_volatilization_risk', 'optimal_operations']
            if not isinstance(weather_analysis.get('agricultural_assessment'), dict):
                return (False, "The 'agricultural_assessment' field must be an object. Please return valid assessment data.")

            missing_assessment_fields = [field for field in required_assessment_fields if field not in weather_analysis['agricultural_assessment']]
            if missing_assessment_fields:
                return (False, f"agricultural_assessment is missing fields: {', '.join(missing_assessment_fields)}. Please ensure all assessment fields are included.")

            # Validate flight_clearance_status is boolean
            if not isinstance(weather_analysis['agricultural_assessment']['flight_clearance_status'], bool):
                return (False, "flight_clearance_status must be a boolean (true/false).")

            # Validate disease_risk_level is valid
            valid_risk_levels = ['Low', 'Medium', 'High']
            if weather_analysis['agricultural_assessment']['disease_risk_level'] not in valid_risk_levels:
                return (False, f"disease_risk_level must be one of: {', '.join(valid_risk_levels)}.")

            # Validate optimal_operations is a list
            if not isinstance(weather_analysis['agricultural_assessment']['optimal_operations'], list):
                return (False, "optimal_operations must be an array of strings.")

            return (True, result.raw)

        except json.JSONDecodeError:
            return (False, "Output is not valid JSON. Please return a properly formatted JSON string.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}. Please check the output format.")

    def validate_sensor_analysis_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """
        Validates the output from the crop_sensor_task.
        Ensures the result contains valid JSON with required fields and proper data types.

        Args:
            result: TaskOutput from the crop_sensor_task

        Returns:
            Tuple of (is_valid, content_or_error_message)
        """
        try:
            # Parse the JSON output
            data = json.loads(result.raw)

            # Check for required top-level fields
            required_fields = ['sensor_analysis']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (False, f"Missing required fields: {', '.join(missing_fields)}. Please ensure all fields are included.")

            # Validate sensor_analysis
            if not isinstance(data.get('sensor_analysis'), dict):
                return (False, "The 'sensor_analysis' field must be an object.")

            sensor_analysis = data['sensor_analysis']
            required_analysis_fields = ['summary', 'farm_id', 'readings_count', 'analysis_period', 'zones_analyzed',
                             'soil_health_metrics', 'irrigation_recommendations', 'environmental_correlations', 'alerts']
            missing_analysis_fields = [field for field in required_analysis_fields if field not in sensor_analysis]

            if missing_analysis_fields:
                return (False, f"sensor_analysis is missing fields: {', '.join(missing_analysis_fields)}. Please ensure all fields are included.")

            # Validate zones_analyzed is a list
            if not isinstance(sensor_analysis.get('zones_analyzed'), list):
                return (False, "The 'zones_analyzed' field must be an array of zone names.")

            # Validate soil_health_metrics is a list
            if not isinstance(sensor_analysis.get('soil_health_metrics'), list):
                return (False, "The 'soil_health_metrics' field must be an array of soil health objects.")

            if len(sensor_analysis['soil_health_metrics']) == 0:
                return (False, "soil_health_metrics must contain at least one zone analysis.")

            # Validate each soil health metric
            required_metric_fields = ['zone_name', 'average_moisture', 'average_temperature', 'average_ph', 'status']
            for idx, metric in enumerate(sensor_analysis['soil_health_metrics']):
                if not isinstance(metric, dict):
                    return (False, f"soil_health_metrics[{idx}] must be an object.")

                missing_metric_fields = [field for field in required_metric_fields if field not in metric]
                if missing_metric_fields:
                    return (False, f"soil_health_metrics[{idx}] is missing fields: {', '.join(missing_metric_fields)}.")

            # Validate irrigation_recommendations is a list
            if not isinstance(sensor_analysis.get('irrigation_recommendations'), list):
                return (False, "The 'irrigation_recommendations' field must be an array.")

            # Validate each irrigation recommendation
            required_irrigation_fields = ['zone', 'action', 'priority', 'reasoning']
            for idx, rec in enumerate(sensor_analysis['irrigation_recommendations']):
                if not isinstance(rec, dict):
                    return (False, f"irrigation_recommendations[{idx}] must be an object.")

                missing_irrigation_fields = [field for field in required_irrigation_fields if field not in rec]
                if missing_irrigation_fields:
                    return (False, f"irrigation_recommendations[{idx}] is missing fields: {', '.join(missing_irrigation_fields)}.")

            # Validate environmental_correlations is a list
            if not isinstance(sensor_analysis.get('environmental_correlations'), list):
                return (False, "The 'environmental_correlations' field must be an array of strings.")

            # Validate alerts is a list
            if not isinstance(sensor_analysis.get('alerts'), list):
                return (False, "The 'alerts' field must be an array of strings.")

            return (True, result.raw)

        except json.JSONDecodeError:
            return (False, "Output is not valid JSON. Please return a properly formatted JSON string.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}. Please check the output format.")

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

    def validate_combined_output(self, result: TaskOutput) -> Tuple[bool, Any]:
        """
        Validates the combined output from all crop crew tasks.
        Ensures the result contains weather_analysis, sensor_analysis, and vision_analysis.

        Args:
            result: TaskOutput from the aggregation task

        Returns:
            Tuple of (is_valid, content_or_error_message)
        """
        try:
            # Parse the JSON output
            data = json.loads(result.raw)

            # Check for required top-level fields
            required_fields = ['weather_analysis', 'sensor_analysis', 'vision_analysis']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return (False, f"Missing required fields: {', '.join(missing_fields)}. Please ensure all analysis sections are included.")

            # Validate each section is a dictionary
            for field in required_fields:
                if not isinstance(data[field], dict):
                    return (False, f"The '{field}' field must be an object.")

            return (True, result.raw)

        except json.JSONDecodeError:
            return (False, "Output is not valid JSON. Please return a properly formatted JSON string.")
        except Exception as e:
            return (False, f"Validation error: {str(e)}. Please check the output format.")

    @agent
    def weather_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['weather_analyst'], # type: ignore[index]
            verbose=True,
            tools=[WeatherAPITool()],
            allow_delegation=True
        )

    @agent
    def crop_sensor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['crop_sensor_agent'], # type: ignore[index]
            verbose=True,
            tools=[SensorDataRetriever()],
            reasoning=True,
            allow_delegation=True
        )
    
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

    @agent
    def analysis_aggregator(self) -> Agent:
        return Agent(
            config=self.agents_config['analysis_aggregator'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def weather_task(self) -> Task:
        return Task(
            config=self.tasks_config['weather_task'], # type: ignore[index]
            guardrail=self.validate_weather_report_output,
            output_pydantic=WeatherReportResult
        )

    @task
    def crop_sensor_task(self) -> Task:
        return Task(
            config=self.tasks_config['crop_sensor_task'], # type: ignore[index]
            guardrail=self.validate_sensor_analysis_output,
            output_pydantic=SensorAnalysisResult,
        )
    
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

    @task
    def aggregation_task(self) -> Task:
        return Task(
            config=self.tasks_config['aggregation_task'], # type: ignore[index]
            guardrail=self.validate_combined_output,
            output_pydantic=CombinedCropAnalysis
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CropCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
