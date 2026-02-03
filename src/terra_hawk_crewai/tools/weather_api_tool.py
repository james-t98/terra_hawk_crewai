#!/usr/bin/env python3
import os
from typing import Type

import requests
from pydantic import BaseModel, Field

from crewai.tools import BaseTool


class WeatherAPIToolInput(BaseModel):
    """Input schema for WeatherAPITool."""
    location: str = Field(..., description="City name or location to get the weather for.")


class WeatherAPITool(BaseTool):
    name: str = "Weather API Tool"
    description: str = (
        "A tool to get the current weather and air quality data for a given location using a weather API. "
        "Returns temperature, conditions, humidity, wind speed, and detailed air quality metrics (PM2.5, PM10, O3, NO2, SO2, CO). "
        "Useful for agricultural planning and crop management decisions."
    )
    args_schema: Type[BaseModel] = WeatherAPIToolInput

    def _run(self, location: str) -> str:
        """
        Get current weather and air quality data for a location.

        Args:
            location: City name or location to get weather for

        Returns:
            Formatted string with weather and air quality information
        """
        # Get API key from environment variable
        api_key = os.environ.get("WEATHER_API_KEY")

        if not api_key:
            return "Error: WEATHER_API_KEY environment variable not set. Please configure your API key."

        # Make API request
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=yes"

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                return f"""The current weather data in {location} is:
                    Current temperature: {data['current']['temp_c']}°C
                    Condition: {data['current']['condition']['text']}
                    Humidity: {data['current']['humidity']}%
                    Wind speed: {data['current']['wind_kph']} kph
                    Air Quality Index (AQI): {data['current']['air_quality']['us-epa-index']}
                    Feels like: {data['current']['feelslike_c']}°C

                    Air Quality Stats:
                    PM2.5: {data['current']['air_quality']['pm2_5']} µg/m³
                    PM10: {data['current']['air_quality']['pm10']} µg/m³
                    O3: {data['current']['air_quality']['o3']} µg/m³
                    NO2: {data['current']['air_quality']['no2']} µg/m³
                    SO2: {data['current']['air_quality']['so2']} µg/m³
                    CO: {data['current']['air_quality']['co']} µg/m³"""

            elif response.status_code == 401:
                return "Error: Invalid API key. Please check your WEATHER_API_KEY configuration."
            elif response.status_code == 400:
                return f"Error: Location '{location}' not found. Please provide a valid city name."
            else:
                return f"Error fetching weather data. Status code: {response.status_code}"

        except requests.exceptions.Timeout:
            return "Error: Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to connect to weather API. {str(e)}"
        except (KeyError, ValueError) as e:
            return f"Error: Unexpected response format from API. {str(e)}"
