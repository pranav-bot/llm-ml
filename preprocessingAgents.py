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
class PreprocessingAgent:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def __init__(self, domian, model_path, dataset_path, preprocessing_code, eda, current_metrics):
        self.domain = domian
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.preprcoessing_code = preprocessing_code
        self.eda = eda
        self.current_metrics = current_metrics

    
    @agent
    def preprocessor_optimizer(self) -> Agent:
        """
        Create and configure an agent to optimize data preprocessing for improved model performance.
        """
        agent_obj = Agent(
            name="ML Preprocessing Optimizer",
            role="Data Preprocessing Expert for ML",
            goal=(
                f"Analyze and improve the current preprocessing pipeline to maximize model performance "
                f"for the domain: {self.domain}. Consider the model located at {self.model_path}, the dataset at {self.dataset_path}, "
                f"and use insights from the exploratory data analysis (EDA): {self.eda} and current model metrics: {self.current_metrics}. "
                f"Provide an improved version of the preprocessing code that is compatible with the model and data, "
                f"and that is likely to improve or optimize the evaluation metrics."
            ),
            backstory="An experienced ML engineer specializing in optimizing preprocessing strategies to boost model performance.",
            description="Reviews datasets, EDA, and current metrics to suggest and generate enhanced preprocessing code tailored to specific model and domain needs.",
            llm="gemini/gemini-1.5-pro"
        )
        return agent_obj
    
    @task
    def preprocessing_task(self) -> Task:
        """
        Create a task to generate optimized preprocessing code based on current dataset, EDA, and model performance.
        The prompt instructs the agent to output only the improved preprocessing code.
        """
        return Task(
            description=(
                f"You are provided with:\n"
                f"- Domain: {self.domain}\n"
                f"- Dataset path: {self.dataset_path}\n"
                f"- Current model path: {self.model_path}\n"
                f"- Current preprocessing code:\n{self.preprcoessing_code}\n"
                f"- EDA insights:\n{self.eda}\n"
                f"- Current model performance metrics:\n{self.current_metrics}\n\n"
                "Your task is to analyze the given preprocessing code and improve it to maximize model performance.\n"
                "Take into account the domain, dataset structure, and model evaluation metrics.\n"
                "Return ONLY the complete improved preprocessing code in Python. Do not include explanations or additional text."
            ),
            expected_output="An improved version of the preprocessing code as a single Python script.",
            agent=self.preprocessor_optimizer(),  
        )
    
    @crew
    def crew(self) -> Crew:
        """
        Create and return a crew for the entire analysis process.
        """
        return Crew(
            agents=[self.preprocessor_optimizer()],
            tasks=[self.preprocessing_task()],
            process=Process.sequential  # Execute tasks sequentially.
        )



    