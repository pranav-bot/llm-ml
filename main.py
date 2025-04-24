import pickle
import sys
import json
from inputModule import InputModule
from preprocessingAgents import PreprocessingAgent
from classificationAgents import ClassificationAgents
from regressionAgent import RegressionAgents

def main():
    choice = input("CLassification and Regression")

    if choice=='1':
        print(1)
        userInput = InputModule()
        userInput.take_input()
        userInput.display_inputs()

        eda = userInput.eda()

        preprocessing_agent = PreprocessingAgent(
            domian=userInput.domain,
            model_path=userInput.model_path,
            dataset_path=userInput.dataset_path,
            preprocessing_code=userInput.preprocessing_code,
            eda=userInput.eda(),
            current_metrics=userInput.current_metrics()
            
        )

            # Call the crew method (which now takes no parameters) to build the crew.
        pp_crew = preprocessing_agent.crew()
        pp_results = pp_crew.kickoff()

        classification_agent = ClassificationAgents(
            domian=userInput.domain,
            model_path=userInput.model_path,
            dataset_path=userInput.dataset_path,
            preprcoessing_code=userInput.preprocessing_code,
            improved_preprocessing_code= pp_results,
            eda=userInput.eda(),
            current_metrics=userInput.current_metrics()
        )

        ca_crew = classification_agent.crew()
        ca_results = ca_crew.kickoff()

        print(ca_results.raw)
        


        filename = "eda.json"

        with open(filename, 'w') as json_file:
            json.dump(userInput.eda(), json_file, indent=4) 
    
    else:
        print(2)
        userInput = InputModule()
        userInput.take_input()
        userInput.display_inputs()

        preprocessing_agent = PreprocessingAgent(
            domian=userInput.domain,
            model_path=userInput.model_path,
            dataset_path=userInput.dataset_path,
            preprocessing_code=userInput.preprocessing_code,
            eda=userInput.eda(),
            current_metrics=userInput.current_metrics()
            
        )

            # Call the crew method (which now takes no parameters) to build the crew.
        pp_crew = preprocessing_agent.crew()
        pp_results = pp_crew.kickoff()

        regression_agent = RegressionAgents(
            domian=userInput.domain,
            model_path=userInput.model_path,
            dataset_path=userInput.dataset_path,
            preprocessing_code=userInput.preprocessing_code,
            improved_preprocessing_code= pp_results,
            eda=userInput.eda(),
            current_metrics=userInput.current_metrics()
        )

        ra_crew = regression_agent.crew()
        ra_results = ra_crew.kickoff()

        print(ra_results.raw)
        


        filename = "eda.json"

        with open(filename, 'w') as json_file:
            json.dump(userInput.eda(), json_file, indent=4) 
        

if __name__ == "__main__":
    main()
