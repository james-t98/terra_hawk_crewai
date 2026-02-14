#!/usr/bin/env python
import os
import json
from datetime import datetime
from typing import Literal
from crewai.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback, HumanFeedbackResult
from terra_hawk_crewai.tools.s3_report_writer import S3ReportWriter
from crewai import Agent, Task
from terra_hawk_crewai.crews.core_crew.core_crew import CoreCrew, MasterReportResult
from terra_hawk_crewai.crews.compliance_crew.compliance_crew import ComplianceCrew, _get_cached_eu_ai_act
from terra_hawk_crewai.crews.crop_crew.crop_crew import CropCrew
from terra_hawk_crewai.tools.s3_report_reader import S3ReportReader

# --- Cost estimation per model (USD per 1K tokens) ---
MODEL_COSTS = {
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt": 0.003, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    # Default fallback
    "default": {"prompt": 0.002, "completion": 0.010},
}


class SmartFarmFlow(Flow):
    def _track_usage(self, crew_name: str, result):
        """Track token usage from crew result."""
        if not self.state.get("token_usage"):
            self.state["token_usage"] = []

        usage = {
            "crew": crew_name,
            "timestamp": datetime.now().isoformat(),
        }

        if hasattr(result, 'token_usage') and result.token_usage:
            tu = result.token_usage
            usage.update({
                "total_tokens": getattr(tu, 'total_tokens', 0),
                "prompt_tokens": getattr(tu, 'prompt_tokens', 0),
                "completion_tokens": getattr(tu, 'completion_tokens', 0),
                "successful_requests": getattr(tu, 'successful_requests', 0),
            })

        self.state["token_usage"].append(usage)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = "default") -> float:
        """Estimate cost in USD for a given token count and model."""
        costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])
        return (prompt_tokens / 1000 * costs["prompt"]) + (completion_tokens / 1000 * costs["completion"])

    def _print_token_summary(self):
        """Print token usage summary with cost estimates."""
        if self.state.get("token_usage"):
            print(f"\n{'='*70}")
            print("Token Usage & Cost Summary:")
            print(f"{'='*70}")
            total_tokens = 0
            total_cost = 0.0
            for usage in self.state["token_usage"]:
                crew = usage["crew"]
                tokens = usage.get("total_tokens", 0)
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                total_tokens += tokens
                cost = self._estimate_cost(prompt, completion)
                total_cost += cost
                print(f"  {crew}: {tokens:,} tokens (prompt: {prompt:,}, completion: {completion:,}) — ~${cost:.4f}")
            print(f"  {'─'*60}")
            print(f"  TOTAL: {total_tokens:,} tokens — ~${total_cost:.4f} USD")
            if self.state.get("eu_ai_act_cached"):
                print(f"  💰 EU AI Act assessment served from cache (saved ~1 agent run)")
            print(f"{'='*70}\n")

    @start()
    def start_flow(self):
        self.state["date"] = datetime.now().strftime("%A, %d %m %Y")
        return ""

    @listen(start_flow)
    def initiate_crop_crew(self):
        crop_result = (
            CropCrew()
            .crew()
            .kickoff(
                inputs={
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "location": os.environ.get("LOCATION"),
                    "farm_id": os.environ.get("FARM_ID"),
                }
            )
        )

        # Parse the combined analysis result with error handling
        try:
            combined_analysis = json.loads(crop_result.raw)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse crop crew output as JSON: {e}")
            print(f"   Raw output (first 500 chars): {crop_result.raw[:500]}")
            # Attempt to extract JSON from the raw output
            combined_analysis = {"weather_analysis": {}, "sensor_analysis": {}, "vision_analysis": {}}

        # Extract individual analyses from the combined output
        self.state["vision_analysis"] = json.dumps(combined_analysis.get("vision_analysis", {}))
        self.state["sensor_analysis"] = json.dumps(combined_analysis.get("sensor_analysis", {}))
        self.state["weather_analysis"] = json.dumps(combined_analysis.get("weather_analysis", {}))

        # Store the full combined analysis as well
        self.state["crop_and_vision_analysis"] = crop_result.raw

        self._track_usage("crop_crew", crop_result)

        return "Crop Crew was run"

    @listen(initiate_crop_crew)
    def initiate_compliance_crew(self):
        # Check for cached EU AI Act assessment
        cached_eu_ai_act = _get_cached_eu_ai_act()
        if cached_eu_ai_act:
            self.state["eu_ai_act_cached"] = True
            print("✓ EU AI Act assessment loaded from cache (valid for 7 days)")

        compliance_result = (
            ComplianceCrew()
            .crew()
            .kickoff(
                inputs={
                    "weather_analysis": self.state.get("weather_analysis", "{}"),
                    "sensor_analysis": self.state.get("sensor_analysis", "{}"),
                    "vision_analysis": self.state.get("vision_analysis", "{}"),
                    "date": self.state["date"],
                }
            )
        )

        # Parse with error handling
        try:
            compliance_data = json.loads(compliance_result.raw)
            self.state["compliance_analysis"] = compliance_result.raw
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse compliance crew output as JSON: {e}")
            print(f"   Raw output (first 500 chars): {compliance_result.raw[:500]}")
            self.state["compliance_analysis"] = json.dumps({
                "compliance_analysis": {"summary": "Parse error", "topic": "", "date": "", "classification": "Neutral",
                                        "operational_impact": "", "nitrogen_emissions_relevance": "", "recommendations": []},
                "eu_ai_act_assessment": {}
            })

        self._track_usage("compliance_crew", compliance_result)
        return "Compliance Crew was run"

    @listen(initiate_compliance_crew)
    @human_feedback(
        message="Would you like to submit the reports?",
        emit=["yes", "no"],
        llm="gpt-4o-mini",
        default_outcome="no"
    )
    def decision(self) -> Literal["yes", "no"]:
        farm_id = os.environ.get("FARM_ID", "FARM-001")

        # --- Direct Master Chief agent (replaces CoreCrew) ---
        master_chief_agent = Agent(
            role="Master Agricultural Operations Analyst and Strategic Advisor",
            goal=(
                "Synthesize insights from vision analysis, weather forecasting, sensor monitoring, "
                "and compliance assessments to produce a comprehensive Master Operations Report that "
                "provides farm leadership with actionable intelligence for strategic decision-making "
                "and operational optimization."
            ),
            backstory=(
                "You are the Chief Agricultural Operations Analyst with over 20 years of experience "
                "in precision agriculture and farm management. Your expertise spans computer vision "
                "for crop health assessment, agrometeorology, soil science, and regulatory compliance. "
                "You excel at identifying cross-functional patterns and producing clear, actionable reports. "
                "Your Master Reports have consistently helped farms increase yields by 15-30%."
            ),
            verbose=True,
            reasoning=True,
            tools=[S3ReportReader()],
            llm="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        )

        # Build task description via concatenation to avoid f-string parsing issues
        task_description = (
            "You have received comprehensive operational data from all farm analysis systems "
            "for " + farm_id + " on " + self.state["date"] + ".\n\n"
            "Your inputs include:\n\n"
            + self.state["weather_analysis"] + "\n\n"
            + self.state["vision_analysis"] + "\n\n"
            + self.state["sensor_analysis"] + "\n\n"
            + self.state["compliance_analysis"] + "\n\n"
            "Synthesize all this information into a comprehensive Master Operations Report. "
            "Analyze cross-functional patterns and correlations between different data sources. "
            "Correlate weather conditions with sensor readings, link crop health from vision "
            "analysis to irrigation recommendations, and assess how compliance requirements "
            "impact operations.\n\n"
            "Use the S3 Report Reader tool to retrieve previous master_report and "
            "vision_analysis reports for " + farm_id + " to identify trends, improvements, "
            "or regressions compared to earlier analyses. Highlight any notable changes "
            "in crop health, sensor readings, or compliance status versus historical data.\n\n"
            "Create a comprehensive Master Operations Report structured as:\n"
            "1. Executive Summary\n"
            "2. Critical Alerts\n"
            "3. Vision & Crop Health Analysis\n"
            "4. Environmental Conditions\n"
            "5. Soil & Irrigation Assessment\n"
            "6. Compliance & Regulatory Status\n"
            "7. Cross-Functional Insights (including historical trends)\n"
            "8. Strategic Recommendations\n"
            "9. Operational Priorities (next 24-48 hours)"
        )

        task_expected_output = (
            'A neatly formatted JSON string:\n'
            '{"master_analysis": {"executive_summary": "...", "critical_alerts": ["..."], '
            '"vision_summary": "...", "weather_summary": "...", "sensor_summary": "...", '
            '"compliance_summary": "...", "cross_functional_insights": ["..."], '
            '"strategic_recommendations": ["..."], "operational_priorities": ["..."], '
            '"overall_farm_status": "Excellent|Good|Attention Needed|Critical"}}'
        )

        master_chief_task = Task(
            description=task_description,
            expected_output=task_expected_output,
            agent=master_chief_agent,
            output_pydantic=MasterReportResult,
        )

        core_result = master_chief_task.execute_sync()

        # Parse with error handling
        try:
            raw = core_result.raw if hasattr(core_result, "raw") else str(core_result)
            json.loads(raw)
            self.state["summary"] = raw
        except (json.JSONDecodeError, AttributeError) as e:
            print("⚠️ Failed to parse master chief output as JSON: " + str(e))
            self.state["summary"] = raw if hasattr(core_result, "raw") else str(core_result)

        # Track usage — direct agent doesn't return crew-level token_usage,
        # but we still record the step for the summary
        self._track_usage("master_chief (direct)", core_result)

        # --- Original CoreCrew approach (commented out) ---
        # core_result = (
        #     CoreCrew()
        #     .crew()
        #     .kickoff(inputs={
        #         "weather_analysis": self.state["weather_analysis"],
        #         "vision_analysis": self.state["vision_analysis"],
        #         "sensor_analysis": self.state["sensor_analysis"],
        #         "compliance_analysis": self.state["compliance_analysis"],
        #         "date": self.state["date"],
        #         "farm_id": os.environ.get("FARM_ID"),
        #     })
        # )
        # self._track_usage("core_crew", core_result)

        reports = [self.state["vision_analysis"], self.state["sensor_analysis"], self.state["compliance_analysis"]]
        for report in reports:
            if report == "" or report == "{}":
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
                "content": self.state["weather_analysis"],
                "type": "weather_report",
                "name": "Weather Report"
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

        self._print_token_summary()

        return f"Submitted {len(successful_writes)}/{len(reports_to_write)} reports. Approved by user: {result.feedback}"
    
    @listen("no")
    def no_reports(self, result: HumanFeedbackResult):
        self._print_token_summary()
        return f"Shall not be submitting reports. Denied by user: {result.feedback}"

def kickoff():
    smart_farm_flow = SmartFarmFlow()
    smart_farm_flow.plot()
    result = smart_farm_flow.kickoff()


if __name__ == "__main__":
    kickoff()
