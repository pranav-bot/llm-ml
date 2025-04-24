import os
import json
import inspect
import google.generativeai as genai
from litellm import completion
from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CodeDocsSearchTool, CodeInterpreterTool
from dotenv import load_dotenv
load_dotenv()

@CrewBase
class ClassificationAgents:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def __init__(self, domian, model_path, dataset_path, preprocessing_code, improved_preprocessing_code, eda, current_metrics):
        self.domain = domian
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.preprcoessing_code = preprocessing_code
        self.improved_preprocessing_code = improved_preprocessing_code
        self.eda = eda
        self.current_metrics = current_metrics

    @agent
    def advanced_model_optimizer(self) -> Agent:
        """
        Create and configure an agent that generates training code to optimize model performance.
        The agent uses scikit-learn classification documentation via CodeDocsSearchTool to write the best possible code.
        It generates code that applies the improved preprocessing for training, while also saving a model
        trained on the old preprocessing code. The agent can choose a different model if it promises better metrics.
        """
        agent_obj = Agent(
            name="Advanced ML Model Optimizer",
            role="ML Model Engineer with Documentation Integration",
            goal=(
                f"Using the domain: {self.domain}, dataset path: {self.dataset_path}, "
                f"and current model path: {self.model_path}, analyze the provided old preprocessing code, "
                f"improved preprocessing code, EDA insights: {self.eda}, and current evaluation metrics: {self.current_metrics}. "
                f"Access the scikit-learn classification documentation via CodeDocsSearchTool (docs_url='https://scikit-learn.org/stable/supervised_learning.html') "
                f"to design the best code that both maximizes the evaluation metrics with the improved preprocessing pipeline "
                f"and saves a model trained using the old preprocessing code. Consider using alternative sklearn classifiers if "
                f"they offer better performance."
            ),
            backstory=(
                "An experienced ML engineer who leverages up-to-date scikit-learn documentation and best practices to generate "
                "code that optimizes model performance. Specializes in constructing pipelines that integrate enhanced preprocessing "
                "with robust model training and model persistence strategies."
            ),
            description=(
                "Analyzes and integrates enhanced preprocessing techniques along with scikit-learn's best practices to write code that "
                "optimizes evaluation metrics. The generated code applies the improved preprocessing for training while also training and saving "
                "a model based on the old preprocessing code, with the flexibility to adopt alternative models as needed."
            ),
            llm="gemini/gemini-2.5-pro-preview-03-25",
            tools=[CodeDocsSearchTool(docs_url='https://scikit-learn.org/stable/supervised_learning.html'), CodeInterpreterTool()]
        )
        return agent_obj
    
    @task
    def model_training_code_task(self) -> Task:
        """
        Create a task to generate a complete Python training pipeline script.
        The script should use the improved preprocessing code for training a model that maximizes evaluation metrics,
        and it must also train a parallel model using the old preprocessing code and save that model.
        The agent has access to the scikit-learn classification documentation via CodeDocsSearchTool.
        It must produce a self-contained script with all necessary steps (imports, data loading, preprocessing,
        model training, evaluation, and saving the model).
        """
        return Task(
            description=(
                f"You are provided with the following inputs:\n"
                f"- Domain: {self.domain}\n"
                f"- Dataset path: {self.dataset_path}\n"
                f"- Current model path: {self.model_path}\n"
                f"- Old preprocessing code:\n{self.preprcoessing_code}\n"
                f"- Improved preprocessing code:\n{self.improved_preprocessing_code}\n"
                f"- EDA insights:\n{self.eda}\n"
                f"- Current evaluation metrics: {self.current_metrics}\n\n"
                f"Using the scikit-learn classification documentation available via CodeDocsSearchTool (docs_url='https://scikit-learn.org/stable/supervised_learning.html'), "
                f"generate the best possible and complete Python code that constructs a full training pipeline for a classification model. "
                f"The script must:\n"
                f"  1. Load the dataset from {self.dataset_path}.\n"
                f"  2. Process the data with the improved preprocessing code to train a model that yields the best evaluation metrics.\n"
                f"  3. Train and save a parallel model using the old preprocessing code (for example, with joblib or pickle), "
                f"maintaining the original processing logic.\n"
                f"  4. Optionally select a different sklearn classifier if it can improve performance based on the current metrics.\n"
                f"  5. Include all necessary imports and code segments so that the entire script is self-contained and runnable.\n\n"
                f"Return ONLY the complete final Python code without any additional explanations or text."
                f"  6. Print the output of the difference between the evaluation metrics models at the end\n\n"
            ),
            expected_output=(
                "A complete, runnable Python script that integrates improved preprocessing for model training and "
                "prints the output of the difference between the evaluation metrics models at the end, including all necessary components."
            ),
            agent=self.advanced_model_optimizer()
        )
    
    @crew
    def crew(self) -> Crew:
        """
        Create and return a crew for the entire analysis process.
        """
        return Crew(
            agents=[self.advanced_model_optimizer()],
            tasks=[self.model_training_code_task()],
            process=Process.sequential  # Execute tasks sequentially.
        )



        