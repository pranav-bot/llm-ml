import os
import json
import inspect
import google.generativeai as genai
from litellm import completion
from crewai import Crew, Agent, Task
from sklearn.utils import all_estimators
from dotenv import load_dotenv
load_dotenv()


# Configure the google-generativeai client.
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_sklearn_model_params(model_name):
    """
    Extracts constructor parameters from a scikit-learn classifier and
    returns a dictionary mapping each parameter to either:
      - "continuous" (if numeric),
      - A list of allowed values (if enumerated, based on known mappings), or
      - The default value (wrapped in a list) if not otherwise specified.
    If a parameter has no default, it is marked as "unspecified".
    """
    classifiers = dict(all_estimators(type_filter="classifier"))
    if model_name not in classifiers:
        return {"error": f"Model {model_name} not found in sklearn classifiers"}
    
    model_class = classifiers[model_name]
    sig = inspect.signature(model_class)
    params_info = {}
    
    # Known enumerations for some common hyperparameters.
    known_enums = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "multi_class": ["auto", "ovr", "multinomial"],
        "loss": ["hinge", "log", "squared_loss", "perceptron"],
        "learning_rate": ["optimal", "constant", "invscaling", "adaptive"],
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "decision_function_shape": ["ovo", "ovr"],
        # Add more known parameters if needed.
    }
    
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            values = "unspecified"  # No default value provided.
        else:
            default = param.default
            if isinstance(default, bool):
                values = [True, False]
            elif isinstance(default, (int, float)):
                values = "continuous"
            elif isinstance(default, str):
                # If we know the parameter has enumerated values, return that array.
                if name in known_enums:
                    values = known_enums[name]
                else:
                    values = [default]
            elif default is None:
                values = [None]
            else:
                values = [str(default)]
        params_info[name] = values
    return params_info

# Define the CrewAI agent.
agent = Agent(
    name="MLParamExtractor",
    role="ML Model Analyzer",
    goal="Extract and return hyperparameters of sklearn classification models in JSON format.",
    backstory="A skilled ML assistant trained to provide detailed hyperparameter configurations for sklearn models.",
    description="Analyzes sklearn classification models and provides parameter details.",
    llm="gemini/gemini-pro"  # Use the proper provider string for Gemini.
)

# Define the CrewAI task that uses our parameter-extraction function.
task = Task(
    description="Extract the hyperparameters for the given sklearn classification model and return them in JSON format.",
    expected_output="A JSON object containing all possible hyperparameters of the requested model.",
    agent=agent,
    function=get_sklearn_model_params  # This function will be executed.
)

# Create the Crew with our agent and task.
crew = Crew(agents=[agent], tasks=[task])

# Run Crew: Provide the sklearn classifier name as input.
model_name = "RandomForestClassifier"  # Example input.
result = crew.kickoff(inputs={"model_name": model_name})

# Convert CrewOutput to a JSON-serializable dictionary.
# If CrewOutput provides a method like `to_dict()`, use that; otherwise, use __dict__.
print(result)
