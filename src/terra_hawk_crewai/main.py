#!/usr/bin/env python
import os
from datetime import datetime
from crewai.flow import Flow, listen, start, and_
from crewai.flow.human_feedback import human_feedback, HumanFeedbackResult
from terra_hawk_crewai.tools.s3_report_writer import S3ReportWriter
from terra_hawk_crewai.crews.core_crew.core_crew import CoreCrew
from terra_hawk_crewai.crews.vision_crew.vision_crew import VisionCrew
from terra_hawk_crewai.crews.compliance_crew.compliance_crew import ComplianceCrew
from terra_hawk_crewai.crews.crop_crew.crop_crew import CropCrew
from terra_hawk_crewai.crews.finance_crew.finance_crew import FinanceCrew

class SmartFarmFlow(Flow):
    @start()
    def start_flow(self):
        self.state["date"] = datetime.now().strftime("%A, %d %m %Y")
        return ""

    @listen(start_flow)
    def initiate_vision_crew(self):
        vision_result = (
            VisionCrew()
            .crew()
            .kickoff(
                inputs={
                    "bucket_name": os.environ.get("S3_BUCKET"),
                    "num_images": os.environ.get("NUM_IMAGES"),
                    "region": os.environ.get("AWS_REGION_NAME"),
                }
            )
        )

        self.state["vision_analysis"] = vision_result.raw
        return "Vision Crew was run"

    @listen(start_flow)
    def initiate_crop_crew(self):
        crop_result = (
            CropCrew()
            .crew()
            .kickoff(
                inputs={
                    "date": self.state["date"],
                    "location": os.environ.get("LOCATION"),
                    "farm_id": os.environ.get("FARM_ID"),
                }
            )
        )

        self.state["sensor_analysis"] = crop_result.raw
        return "Crop Crew was run"

    @listen(start_flow)
    def initiate_finance_crew(self):
        finance_result = (
            FinanceCrew()
            .crew()
            .kickoff(
                inputs={
                    "date": self.state["date"],
                    "farm_id": os.environ.get("FARM_ID"),
                }
            )
        )
        
        self.state["financial_analysis"] = finance_result.raw
        return "Finance Crew was run"

    @listen(initiate_vision_crew)
    def initiate_compliance_crew(self):
        compliance_result = (
            ComplianceCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state["vision_analysis"],
                    "date": self.state["date"],
                }
            )
        )    

        self.state["compliance_analysis"] = compliance_result.raw
        return "Compliance Crew was run"
    
    @listen(and_(initiate_vision_crew, initiate_crop_crew, initiate_finance_crew, initiate_compliance_crew))
    @human_feedback(
        message="Would you like to submit the reports?",
        emit=["yes", "no"],
        llm="gpt-4o-mini",
        default_outcome="no"
    )
    def decision(self):
        core_result = (
            CoreCrew()
            .crew()
            .kickoff(inputs={
                "vision_analysis": self.state["vision_analysis"],
                "sensor_analysis": self.state["sensor_analysis"],
                "financial_analysis": self.state["financial_analysis"],
                "compliance_analysis": self.state["compliance_analysis"],
                "date": self.state["date"],
                "farm_id": os.environ.get("FARM_ID"),
                }
            )
        )
        self.state["summary"] = core_result.raw

        reports = [self.state["vision_analysis"], self.state["sensor_analysis"], self.state["financial_analysis"], self.state["compliance_analysis"]]
        for report in reports:
            if report == "":
                return "no"
        return "yes"
    
    @listen("yes")
    def submit_reports(self, result: HumanFeedbackResult):
        s3_writer = S3ReportWriter()
        current_date = datetime.now().strftime("%Y-%m-%d")
        bucket_name = os.environ.get("S3_BUCKET")
        farm_id = os.environ.get("FARM_ID")
        region = os.environ.get("AWS_REGION_NAME", "eu-west-1")

        # Define all reports to write
        reports_to_write = [
            {
                "content": self.state["vision_analysis"],
                "type": "vision_analysis",
                "name": "Vision Analysis"
            },
            {
                "content": self.state["sensor_analysis"],
                "type": "sensor_analysis",
                "name": "Sensor Analysis"
            },
            {
                "content": self.state["financial_analysis"],
                "type": "financial_analysis",
                "name": "Financial Analysis"
            },
            {
                "content": self.state["compliance_analysis"],
                "type": "compliance_report",
                "name": "Compliance Report"
            },
            {
                "content": self.state["summary"],
                "type": "master_report",
                "name": "Master Report"
            }
        ]

        # Write all reports to S3
        successful_writes = []
        failed_writes = []

        for report in reports_to_write:
            write_result = s3_writer._run(
                bucket_name=bucket_name,
                farm_id=farm_id,
                report_content=report["content"],
                report_type=report["type"],
                date=current_date,
                region=region
            )

            if write_result.get("success"):
                successful_writes.append(report["name"])
                print(f"✓ {report['name']} saved to: {write_result.get('s3_uri')}")
            else:
                failed_writes.append(report["name"])
                print(f"✗ {report['name']} failed: {write_result.get('error')} - {write_result.get('details')}")

        # Summary
        print(f"\n{'='*60}")
        print(f"Report Submission Summary:")
        print(f"{'='*60}")
        print(f"Successfully written: {len(successful_writes)}/{len(reports_to_write)}")
        if successful_writes:
            print(f"  ✓ {', '.join(successful_writes)}")
        if failed_writes:
            print(f"  ✗ Failed: {', '.join(failed_writes)}")
        print(f"{'='*60}\n")

        return f"Submitted {len(successful_writes)}/{len(reports_to_write)} reports. Approved by user: {result.feedback}"
    
    @listen("no")
    def no_reports(self, result: HumanFeedbackResult):
        return f"Shall not be submitting reports. Denied by user: {result.feedback}"

def kickoff():
    smart_farm_flow = SmartFarmFlow()
    smart_farm_flow.plot()
    smart_farm_flow.kickoff()

def plot():
    poem_flow = SmartFarmFlow()
    poem_flow.plot("my_flow_plot")


if __name__ == "__main__":
    kickoff()