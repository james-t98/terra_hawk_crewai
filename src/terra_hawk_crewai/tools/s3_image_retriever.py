#!/usr/bin/env python3
from typing import Type, Dict, Any
import json
import requests

from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class S3ImageRetrieverInput(BaseModel):
    """Input schema for S3ImageRetriever."""

    bucket_name: str = Field(..., description="Name of the S3 bucket to retrieve images from (e.g., 'drone-images-replicate').")
    num_images: int = Field(..., description="Number of images to retrieve from the bucket.")
    region: str = Field(default="eu-west-1", description="AWS region where the bucket is located (e.g., 'eu-west-1').")
    generate_presigned: bool = Field(default=True, description="Whether to generate presigned URLs for the images.")
    presigned_expiration: int = Field(default=3600, description="Expiration time in seconds for presigned URLs.")


class S3ImageRetriever(BaseTool):
    name: str = "S3 Image Retriever"
    description: str = (
        "Retrieves a list of images from an S3 bucket via API. "
        "Returns metadata including image names, sizes, S3 URIs, and presigned URLs. "
        "Useful for accessing drone images or other files stored in S3 buckets."
    )
    args_schema: Type[BaseModel] = S3ImageRetrieverInput

    # API Configuration
    API_URL: str = "https://k68rmkajw8.execute-api.eu-west-1.amazonaws.com/prd/"

    def _run(
        self,
        bucket_name: str,
        num_images: int,
        region: str = "eu-west-1",
        generate_presigned: bool = True,
        presigned_expiration: int = 3600,
    ) -> Dict[str, Any]:
        """
        Retrieve images from S3 bucket via API.

        Args:
            bucket_name: Name of the S3 bucket
            num_images: Number of images to retrieve
            region: AWS region (default: eu-west-1)
            generate_presigned: Whether to generate presigned URLs (default: True)
            presigned_expiration: Expiration time for presigned URLs in seconds (default: 3600)

        Returns:
            Dictionary containing image metadata and presigned URLs
        """
        # Prepare request payload in Lambda event format
        payload = {
            "body": json.dumps({
                "bucket": bucket_name,
                "num_images": num_images,
                "region": region,
                "generate_presigned": generate_presigned,
                "presigned_expiration": presigned_expiration
            }),
            "httpMethod": "POST",
            "headers": {
                "Content-Type": "application/json"
            }
        }

        try:
            # Make POST request
            response = requests.post(
                self.API_URL,
                headers={"Content-Type": "application/json"},
                json=payload
            )

            if response.status_code == 200:
                # Get the full Lambda proxy response
                full_response = response.json()

                # Extract and parse the body (JSON string in Lambda proxy integration)
                body_string = full_response['body']
                data_dict = json.loads(body_string)

                return data_dict

            else:
                return {
                    "error": f"API request failed with status code {response.status_code}",
                    "details": response.text
                }

        except requests.exceptions.RequestException as e:
            return {
                "error": "Request failed",
                "details": str(e)
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "error": "Failed to parse API response",
                "details": str(e)
            }
