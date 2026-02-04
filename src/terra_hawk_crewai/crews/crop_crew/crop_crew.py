from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from terra_hawk_crewai.tools import WeatherAPITool, SensorDataRetriever, DynamoDBVisionRetriever
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

class BoundingBox(BaseModel):
    """Bounding box coordinates for detections"""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")


class Detection(BaseModel):
    """Individual detection from computer vision model"""
    label: str = Field(..., description="Detection label/class name")
    class_id: int = Field(..., description="Class ID from model")
    confidence: float = Field(..., description="Confidence score (0-100)", ge=0.0, le=100.0)
    isHealthy: bool = Field(..., description="Health status of detected crop")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")


class DetectionMetrics(BaseModel):
    """Performance metrics from vision system"""
    detection_count: int = Field(..., description="Total number of detections")
    fps: float = Field(..., description="Frames per second")
    latency: float = Field(..., description="Processing latency in ms")


class VisionRecord(BaseModel):
    """Complete vision detection record"""
    timestamp: str = Field(..., description="ISO timestamp of detection")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    crop_name: str = Field(..., description="Name of the crop being monitored")
    field_name: str = Field(..., description="Field or zone name")
    primary_class: str = Field(..., description="Primary detection class")
    detections: List[Detection] = Field(..., description="List of all detections")
    metrics: DetectionMetrics = Field(..., description="Performance metrics")
    s3_url: str = Field(..., description="S3 URL of the image")
    https_url: str = Field(..., description="HTTPS URL for image access")


class VisionAnalysisSummary(BaseModel):
    """Summary statistics for vision analysis"""
    total_records: int = Field(..., description="Total number of detection records analyzed")
    total_detections: int = Field(..., description="Total number of individual detections")
    healthy_detections: int = Field(..., description="Number of healthy crop detections")
    unhealthy_detections: int = Field(..., description="Number of unhealthy crop detections")
    health_percentage: float = Field(..., description="Percentage of healthy detections", ge=0.0, le=100.0)
    crop_types: List[str] = Field(..., description="List of crop types detected")
    field_names: List[str] = Field(..., description="List of field names covered")
    detection_classes: List[str] = Field(..., description="List of detection classes found")
    key_findings: List[str] = Field(..., description="Key analytical findings and insights")


class VisionAnalysis(BaseModel):
    """Vision analysis from DynamoDB detection data"""
    summary: VisionAnalysisSummary = Field(..., description="Summary statistics and insights")
    records: List[VisionRecord] = Field(..., description="Detailed detection records")


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
    def vision_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['vision_analyzer'], # type: ignore[index]
            verbose=True,
            tools=[DynamoDBVisionRetriever()]
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
    def vision_analyzer_task(self) -> Task:
        return Task(
            config=self.tasks_config['vision_analyzer_task'], # type: ignore[index]
            output_pydantic=VisionAnalysis
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
