import datetime
import json
import re

from matplotlib import pyplot as plt
import numpy as np


TEST_TO_VARIABLE_NAME = {
    "atanasova_counterfactual": "atanasova_counterfact",
    "atanasova_input_from_expl": "atanasova_input_from_expl",
    "cc_shap-posthoc": "cc_shap-posthoc",
    "turpin": "turpin",
    "cc_shap-cot": "cc_shap-cot",
    "loo-posthoc": "loo_cosim_posthoc",
    "loo-cot": "loo_cosim_cot",
}


def load_json_file(file_path):
    """
    Load a JSON file and return its contents as a dictionary.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON content as a dictionary.
    :raises: FileNotFoundError, json.JSONDecodeError
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from file '{file_path}'.")
        print(f"Details: {e}")
        raise


class ExperimentResult:
    def __init__(self, results_json: dict):
        self.results_json = results_json
    
    @property
    def args(self):
        return self.results_json["args"]
    
    @property
    def model(self) -> dict:
        return self.results_json["model"]
    
    @property
    def examples(self) -> dict:
        return self.results_json["samples"]
    
    @property
    def explainer(self) -> str:
        return self.results_json["explainer"]
    
    @property
    def tests(self) -> list[str]:
        return self.results_json["tests"]
    
    @property
    def loo_threshold(self) -> float | None:
        return self.results_json.get("sentence_similarity_threshold")
    
    @property
    def time_elapsed(self) -> str:
        since_epoch = datetime.datetime.strptime(self.results_json["time_elapsed"], "%H:%M:%S.%f")
        time_elapsed = datetime.timedelta(
            hours=since_epoch.hour,
            minutes=since_epoch.minute,
            seconds=since_epoch.second,
            microseconds=since_epoch.microsecond
        )
        return time_elapsed
    
    def __repr__(self):
        model = f"Model: {self.model['full_model_name']} ({self.model['dtype']})"
        tests = f"Tests: {self.tests}"
        explainer = f"Explainer: {self.explainer})"
        examples = f"Examples: {len(self.examples)}"
        args = f"Args: {self.args}"
        time_elapsed = f"Time elapsed: {self.time_elapsed}"
        loo_threshold = f"LOO sim threshold: {self.loo_threshold}"

        return "\n".join((model, tests, explainer, examples, args, time_elapsed, loo_threshold))

    def examples_names(self) -> list[str]:
        return list(self.examples.keys())

    def get_example(self, example_name: str) -> dict:
        return self.examples[example_name]
    
    def get_variable(self, variable):
        cc_shap_cot_values = []
        for example_name in self.examples_names():
            cc_shap_score = self.get_example(example_name)[variable]
            cc_shap_cot_values.append(float(cc_shap_score))

        return np.array(cc_shap_cot_values)

    def describe(self, variable):
        variable_values = self.get_variable(variable)

        print("Mean: ", variable_values.mean())
        print("Min: ", variable_values.min())
        print("Max: ", variable_values.max())
        print("Std dev: ", variable_values.std())
    
    def mean(self, variable):
        variable_values = self.get_variable(variable)
        return variable_values.mean()

    def boxplot(self, variable):
        cc_shap_cot_values = self.get_variable(variable)

        plt.boxplot(cc_shap_cot_values, orientation="horizontal")
        plt.xlim((-1.0, 1.0))
        plt.show()


# Transforms a cumulative array to a array of differences
def cumsum_to_differences(cumsum_array):
    return np.array([
        cumsum_array[idx] - cumsum_array[idx - 1]
        if idx != 0 else cumsum_array[idx]
        for idx, _ in enumerate(cumsum_array)
    ])


def find_arg(input_str, arg_name) -> str | None:
    pattern = rf"{arg_name}=(?:'([^']*)'|(\d+))"
    match = re.search(pattern, input_str)
    if match:
        return match.group(1) or int(match.group(2))
    return None


if __name__ == "__main__":
    # Check if the cumsum_to_differences is correct
    in_expl_cumsum = np.array([0, 1, 1, 2, 3, 3])
    print("Cumulative:", in_expl_cumsum)
    print("Differences:", cumsum_to_differences(in_expl_cumsum))