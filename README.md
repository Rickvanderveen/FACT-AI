# FACT-AI
This is the repository for the project course FACT-AI. This contains a reproducibility study of the paper *On Measuring Faithfulness or Self-consistency of Natural Language Explanations* by [Parcalabescu & Frank, 2024](https://doi.org/10.18653/v1/2024.acl-long.329). Additionally, we added new experiments and extensions.

## Installation
There are two conda environments used in this project, `fact` and `fact2`.

### Install the `fact` environment
To install the conda environment you can run:

```
conda env create -f environment_fact.yml
conda activate fact
python3 -m spacy download en_core_web_sm
```

Finally, Wordnet from NLTK needs to be installed. This can be done by running the following python code:
```python
import nltk
nltk.download("wordnet")
```

### Install the `fact2` environment
To install the `fact2` conda environment you can run:

```
conda env create -f environment_fact2.yml
conda activate fact2
python3 -m spacy download en_core_web_sm
```

If the NLTK wordnet has not been downloaded yet (explained in the `fact` environment install) this needs to be done to run the code. Wordnet can be installed the following code:
```python
import nltk
nltk.download("wordnet")
```

## Usage
Running a test works in two steps.
1. In the file `faithfulness.py`, select which tests you want to use (by commenting or uncommenting a certain test in the `TESTS` list, line 103).
2. run `python3 faithfulness.py <dataset> <model_name> <n_samples> <explainer> <explainer_max_evaluation> <loo_similarity_threshold>`

Additional arguments can be
- `--result_dir=<own_directory>` to store the raw output in a certain directory
- `--classify_pred` to use the original way of using a seperate classification step to predict the label of the prompt instead of using the output of the explainer. This is only relevant for the tests that use CC-SHAP post-hoc

Not all models can be run with the same conda environment. \
`fact` is used for: LLama2, Falcon and Mistral \
`fact2` is used for newer models: Falcon 3, Phi 3 and Phi 4

### Running experiments
The reproducibility experiments can be done by setting the following tests: `['atanasova_counterfactual', 'atanasova_input_from_expl', 'cc_shap-posthoc', 'turpin', 'cc_shap-cot']` and running the following command.
```
conda activate fact
python3 faithfulness.py <dataset> <model_name> 100 "partition" 500 0.68 --result_dir="reproducibility_results"
```
- `<model_name>` is either `llama2-7b-chat`, `falcon-7b-chat` or `mistral-7b-chat` \
- `<dataset>` is either `comve`, `esnli` or `disambiguation_qa`

The LOO measure experiment can be done by setting the following tests: `['loo-posthoc', 'loo-cot']` and running the following command:
```
python3 faithfulness.py <dataset> <model_name> 100 "partition" 500 0.68 --result_dir="loo_test"
```
- `<model_name>` is either `llama2-7b-chat`, `falcon-7b-chat` or `mistral-7b-chat` \
- `<dataset>` is either `comve`, `esnli` or `disambiguation_qa`

The speed/time experiment for CC-SHAP and LOO is done by setting the following tests: `['cc_shap-posthoc', 'cc_shap-cot', 'loo-posthoc', 'loo-cot', 'loo-posthoc-slow', 'loo-cot-slow']`\
and running the command:
```
conda activate fact
python3 faithfulness.py comve falcon-7b-chat 100 "partition" 500 0.68 --result_dir="timing"
```

The alternative prompt experiment is done by setting the following tests:
`[cc_shap-cot-other-input, loo-cot-other-input]`
```
python3 faithfulness.py <dataset> <model_name> 100 "partition" 500 0.68 --result_dir="alt_prompt"
```

-`<model_name>` is either `falcon-7b-chat` or `falcon3-7B-chat`
- `<dataset>` is either `comve`, `esnli` or `disambiguation_qa`

### Results
There are two notebooks. 
- `LOO_analysis.ipynb` This notebook provides the evaluation of the Leave One Out (LOO) cosine similarity measure and the determination of the similarity threshold. This notebook can be executed using the `fact2` environment.
- `result_analysis.ipynb` In this notebook, the results for all the experiments are transformed from there raw output data (located in the `results` folder) into human readable dataframes. This notebook can be executed using the `fact` environment.