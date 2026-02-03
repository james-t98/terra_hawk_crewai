#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback, HumanFeedbackResult
from terra_hawk_crewai.crews.poem_crew.poem_crew import PoemCrew

class PoemFlow(Flow):

    @start()
    def generate_sentence_count(self, crewai_trigger_payload: dict = None):
        print("Generating sentence count")

        # Use trigger payload if available
        if crewai_trigger_payload:
            # Example: use trigger data to influence sentence count
            self.state["sentence_count"] = crewai_trigger_payload.get('sentence_count', randint(1, 5))
            print(f"Using trigger payload: {crewai_trigger_payload}")
        else:
            self.state["sentence_count"] = randint(1, 5)

    @listen(generate_sentence_count)
    @human_feedback(
        message="Do you like this poem?",
        emit=["yes", "no"],
        llm="gpt-4o-mini",
        default_outcome="no"
    )
    def generate_poem(self):
        result = (
            PoemCrew()
            .crew()
            .kickoff(inputs={"sentence_count": self.state["sentence_count"]})
        )

        self.state["poem"] = result.raw

        if self.state["poem"]:
            return "yes"

        return "no"
    
    @listen("yes")
    def write_poem(self):
        return self.state["poem"]
    
    @listen("no")
    def dont_write_poem(self, result: HumanFeedbackResult):
        return f"Human response: {result.feedback}. Not writing the poem."


def kickoff():
    poem_flow = PoemFlow()
    # poem_flow.plot()
    result = poem_flow.kickoff()


def plot():
    poem_flow = PoemFlow()
    poem_flow.plot()


def run_with_trigger():
    """
    Run the flow with trigger payload.
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    poem_flow = PoemFlow()

    try:
        result = poem_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    kickoff()
