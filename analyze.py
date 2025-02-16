import os
import json
import inspect
import google.generativeai as genai
from litellm import completion
from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
load_dotenv()

@CrewBase
class Analyzer:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def __init__(self, model_class, model_all_hyperparameters, model_hyperparameters, model_domain):
        """
        Initialize the Analyzer with all necessary parameters.
        :param model_class: The model instance or class.
        :param model_all_hyperparameters: All possible hyperparameters (e.g. KNNParams).
        :param model_hyperparameters: The current hyperparameters (as a dict).
        :param model_domain: A description of the model domain.
        """
        self.model_class = model_class
        self.model_all_hyperparameters = model_all_hyperparameters
        self.model_hyperparameters = model_hyperparameters
        self.model_domain = model_domain

    @agent
    def analyzer(self) -> Agent:
        """
        Create and configure an agent for ML model analysis.
        """
        agent_obj = Agent(
            name="MLParamAnalyzer",
            role="ML Model Analyzer",
            goal=(
                f"Extract and return only the best possible param_grid that can be directly used for grid search "
                f"for the sklearn classification model {self.model_class} in the domain {self.model_domain}. "
                f"Consider all possible hyperparameters: {self.model_all_hyperparameters} and the current hyperparameters: {self.model_hyperparameters}."
            ),
            backstory="A skilled ML assistant trained to provide optimal hyperparameter configurations for sklearn models.",
            description="Analyzes sklearn classification models and provides a ready-to-use param_grid for grid search.",
            llm="gemini/gemini-pro"  # Use the proper provider string for Gemini.
        )
        return agent_obj

    @task
    def reccomending_task(self) -> Task:
        """
        Create a task to recommend the best param_grid for the given model.
        The prompt instructs the agent to output only the JSON param_grid.
        """
        return Task(
            description=(
                f"Given the current hyperparameters: {self.model_hyperparameters} and all possible hyperparameters: {self.model_all_hyperparameters} "
                f"for the sklearn classification model {self.model_class} in the domain {self.model_domain}, "
                "provide the best possible param_grid for grid search. "
                "Return ONLY a JSON object with the param_grid and no additional text or explanation."
            ),
            expected_output="A JSON object containing the best param_grid to directly use in grid search.",
            agent=self.analyzer(),  # Use the analyzer agent.
        )

    @crew
    def crew(self) -> Crew:
        """
        Create and return a crew for the entire analysis process.
        """
        return Crew(
            agents=[self.analyzer()],
            tasks=[self.reccomending_task()],
            process=Process.sequential  # Execute tasks sequentially.
        )
