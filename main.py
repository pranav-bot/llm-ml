import pickle
from analyze import Analyzer
from knn import KNNParams  # Assuming KNNParams is defined in knn.py
import sys
sys.path.append('/home/shin0bi/dev/llm-ml/')

def model_name_and_hyperparameters(model_path):
    """
    Loads the pickled model and extracts its hyperparameters if available.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Inspect the type of the model
    model_type = model

    if hasattr(model, 'get_params'):
        params = model.get_params()
        return model_type, params
    else:
        print("This model does not expose hyperparameters via get_params.")
        return model_type, {}

def main():
    model_info = model_name_and_hyperparameters('best_knn_model.pkl')
    model_class, model_hyperparameters = model_info[0], model_info[1]
    
    # Create the Analyzer instance with all necessary parameters.
    # If KNNParams is a dict and you need it hashable, you might use:
    #    frozenset(KNNParams.items())
    # However, here we assume KNNParams is already in an acceptable format.
    analyzer_agent = Analyzer(
        model_class=model_class,
        model_all_hyperparameters=KNNParams,
        model_hyperparameters=model_hyperparameters,
        model_domain='iris classification dataset'
    )
    
    # Call the crew method (which now takes no parameters) to build the crew.
    crew_instance = analyzer_agent.crew()
    results = crew_instance.kickoff()
    if isinstance(results, list) and results:
        best_parameters = results[0]
    else:
        best_parameters = results
    
    # Print only the best parameters output.
    print("Best Parameters:")
    print(results.raw)

if __name__ == "__main__":
    main()
