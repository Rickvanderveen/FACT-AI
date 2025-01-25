import argparse
import datetime
from pathlib import Path
import time
import torch
from accelerate import Accelerator
import pandas as pd
import json
import pipeline
import shap

import random
import spacy
from transformers.utils import logging as hf_logging
import logging

import cc_shap_logger as cc_log
import cc_shap
from tests import faithfulness_test_atanasova_etal_input_from_expl, faithfulness_test_atanasova_etal_counterfact, faithfulness_test_lanham_etal, faithfulness_test_turpin_etal

default_log_level = logging.INFO
cc_log.setup_logger(default_log_level)

logger = logging.getLogger("shap")
logger_question = logging.getLogger("shap.prediction")
logger_answer = logging.getLogger("shap.answer")
logger_explanation = logging.getLogger("shap.explanation")

logger.info(f"Cuda is available: {torch.cuda.is_available()}")

torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

hf_logging.set_verbosity_error()

nlp = spacy.load("en_core_web_sm")
random.seed(42)


max_new_tokens = 100

parser = argparse.ArgumentParser(description="Testing faithfulness of LLMs")
# Add arguments
parser.add_argument(
    "c_task",
    type=str,
    help="The task to perform."
)
parser.add_argument(
    "model_name",
    type=str,
    help="The name of the model to use."
)
parser.add_argument(
    "number_of_samples",
    type=int,
    help="The number of samples to process."
)
parser.add_argument(
    "explainer_type",
    type=str,
    help="The type of explainer to use (default: auto)."
)
parser.add_argument(
    "max_evaluations",
    type=int,
    help="The maximum number of iterations for the explainer to compute the shap values"
)
parser.add_argument(
    "--classify_pred",
    action="store_true",
    help="To use a seperate classify to predict the label or use them from the explanation"
)

# Parse the arguments
args = parser.parse_args()

logger.info(f"Args: {args}")

# Access the arguments
c_task = args.c_task
model_name = args.model_name
num_samples = args.number_of_samples
explainer_type = args.explainer_type
explainer_max_evaluations = args.max_evaluations
use_separate_classify_prediction = args.classify_pred

visualize = False

TESTS = [
    'atanasova_counterfactual',
    'atanasova_input_from_expl',
    'cc_shap-posthoc',
    'turpin',
    # 'lanham', # Needs a helper model
    'cc_shap-cot',
]

LABELS = {
    'comve': ['A', 'B'], # ComVE
    'causal_judgment': ['A', 'B'],
    'disambiguation_qa': ['A', 'B', 'C'],
    'logical_deduction_five_objects': ['A', 'B', 'C', 'D', 'E'],
    'esnli': ['A', 'B', 'C'],
}

dtype = torch.float32 if ('llama2-7b' in model_name) or ("llama3-8B" in model_name) else torch.float16
full_model_name = pipeline.full_model_name(model_name)
model_pipeline = pipeline.Pipeline.from_pretrained(
    model_name,
    dtype,
    max_new_tokens,
    TESTS
)

algorithm_types = ["exact", "permutation", "partition", "tree", "additive", "linear", "deep"]
if explainer_type in algorithm_types:
    explainer = shap.Explainer(
        model_pipeline.model,
        model_pipeline.tokenizer,
        algorithm=explainer_type,
        silent=True
    )
elif explainer_type == "random":
    explainer = shap.explainers.other.Random(
        model_pipeline.model,
        model_pipeline.tokenizer
    )
else:
    logger.error(f"Explainer type `{explainer_type}` is not used")
    quit()

logger.info(f"Using the {str(explainer)} explainer")

############################# 
############################# run experiments on data
############################# 
res_dict = {}
formatted_inputs, correct_answers, wrong_answers = [], [], []
correct_predictions, correct_predictions_cot = 0, 0
atanasova_counterfact_count, atanasova_input_from_expl_test_count, turpin_test_count, count, cc_shap_post_hoc_sum, cc_shap_cot_sum = 0, 0, 0, 0, 0, 0
lanham_early_count, lanham_mistake_count, lanham_paraphrase_count, lanham_filler_count = 0, 0, 0, 0

logger.info("Preparing data...")
###### ComVE tests
if c_task == 'comve':
    # read in the ComVE data from the csv file
    data = pd.read_csv('SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALL data/Test Data/subtaskA_test_data.csv')
    data = data.sample(frac=1, random_state=42) # shuffle the data
    # read in the ComVE annotations from the csv file
    gold_answers = pd.read_csv('SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALL data/Test Data/subtaskA_gold_answers.csv', header=None, names=['id', 'answer'])

    for idx, sent0, sent1 in zip(data['id'], data['sent0'], data['sent1']):
        if count + 1 > num_samples:
            break
        
        formatted_input = model_pipeline.format_example_comve(sent0, sent1)
        gold_answer = gold_answers[gold_answers['id'] == idx]['answer'].values[0]
        correct_answer = 'A' if gold_answer == 0 else 'B'
        wrong_answer = 'A' if gold_answer == 1 else 'B'

        formatted_inputs.append(formatted_input)
        correct_answers.append(correct_answer)
        wrong_answers.append(wrong_answer)

        count += 1

###### bbh tests
elif c_task in ['causal_judgment', 'disambiguation_qa', 'logical_deduction_five_objects']:
    with open(f'cot-unfaithfulness/data/bbh/{c_task}/val_data.json','r') as f:
        data = json.load(f)['data']
        random.shuffle(data)

    for row in data:
        if count + 1 > num_samples:
            break
        
        formatted_input = row['parsed_inputs'] + '.'
        gold_answer = row['multiple_choice_scores'].index(1)
        correct_answer = LABELS[c_task][gold_answer]
        wrong_answer = random.choice([x for x in LABELS[c_task] if x != correct_answer])

        formatted_inputs.append(formatted_input)
        correct_answers.append(correct_answer)
        wrong_answers.append(wrong_answer)

        count += 1

######### e-SNLI tests
elif c_task == 'esnli':
    # read in the e-SNLI data from the csv file
    data = pd.read_csv('e-SNLI/esnli_test.csv')
    data = data.sample(frac=1, random_state=42) # shuffle the data

    for gold_answer, sent0, sent1 in zip(data['gold_label'], data['Sentence1'], data['Sentence2']):
        if count + 1 > num_samples:
            break
        
        formatted_input = model_pipeline.format_example_esnli(sent0, sent1)
        if gold_answer == 'entailment':
            correct_answer = 'A'
        elif gold_answer == 'contradiction':
            correct_answer = 'B'
        elif gold_answer == 'neutral':
            correct_answer = 'C'
        wrong_answer = random.choice([x for x in LABELS[c_task] if x != correct_answer])

        formatted_inputs.append(formatted_input)
        correct_answers.append(correct_answer)
        wrong_answers.append(wrong_answer)

        count += 1

logger.info("Done preparing data. Running test...")
start_test = time.time()
for k, formatted_input, correct_answer, wrong_answer in zip(range(len(formatted_inputs)), formatted_inputs, correct_answers, wrong_answers):
    logger.info(f"Example {k}")
    # compute model accuracy
    ask_input = model_pipeline.get_prompt_answer_ata(formatted_input, c_task)
    prediction = model_pipeline.lm_classify(ask_input, LABELS[c_task])
    correct_predictions += 1 if prediction == correct_answer else 0

    # for accuracy with CoT: first let the model generate the cot, then the answer.
    cot_prompt = model_pipeline.get_cot_prompt(formatted_input)
    generated_cot = model_pipeline.lm_generate(
        cot_prompt,
        max_new_tokens,
        repeat_input=True
    )
    ask_for_final_answer = model_pipeline.get_final_answer(generated_cot, c_task)
    prediction_cot = model_pipeline.lm_classify(ask_for_final_answer,LABELS[c_task])
    correct_predictions_cot += 1 if prediction_cot == correct_answer else 0

    # # post-hoc tests
    if 'atanasova_counterfactual' in TESTS:
        atanasova_counterfact = faithfulness_test_atanasova_etal_counterfact(
            model_pipeline,
            formatted_input,
            prediction,
            LABELS[c_task],
            c_task,
        )
    else:
        atanasova_counterfact = 0

    if 'atanasova_input_from_expl' in TESTS and c_task == 'comve':
        atanasova_input_from_expl = faithfulness_test_atanasova_etal_input_from_expl(
            model_pipeline,
            sent0,
            sent1,
            prediction,
            correct_answer,
            LABELS[c_task],
            c_task,
            max_new_tokens,
        )
    else:
        atanasova_input_from_expl = 0

    if 'cc_shap-posthoc' in TESTS:
        cc_shap_measures = cc_shap.cc_shap_measure(
            formatted_input,
            LABELS[c_task],
            'post_hoc',
            c_task,
            model_pipeline,
            explainer,
            max_new_tokens,
            max_evaluations = explainer_max_evaluations,
            use_separate_classify_prediction = use_separate_classify_prediction
        )
        score_post_hoc, dist_correl_ph, mse_ph, var_ph, kl_div_ph, js_div_ph, shap_plot_info_ph = cc_shap_measures
    else:
        score_post_hoc, dist_correl_ph, mse_ph, var_ph, kl_div_ph, js_div_ph, shap_plot_info_ph = 0, 0, 0, 0, 0, 0, 0

    # # CoT tests
    if 'turpin' in TESTS:
        turpin = faithfulness_test_turpin_etal(
            model_pipeline,
            formatted_input,
            prediction_cot,
            correct_answer,
            wrong_answer,
            LABELS[c_task],
            max_new_tokens,
            c_task,
        )
    else:
        turpin = 0

    if 'lanham' in TESTS:
        lanham_early, lanham_mistake, lanham_paraphrase, lanham_filler = faithfulness_test_lanham_etal(
            model_pipeline,
            prediction_cot,
            generated_cot,
            cot_prompt,
            LABELS[c_task],
            c_task,
            max_new_tokens,
        )
    else:
        lanham_early, lanham_mistake, lanham_paraphrase, lanham_filler = 0, 0, 0, 0

    if 'cc_shap-cot' in TESTS:
        cc_shap_measures = cc_shap.cc_shap_measure(
            formatted_input,
            LABELS[c_task],
            "cot",
            c_task,
            model_pipeline,
            explainer,
            max_new_tokens,
            max_evaluations = explainer_max_evaluations,
        )
        score_cot, dist_correl_cot, mse_cot, var_cot, kl_div_cot, js_div_cot, shap_plot_info_cot = cc_shap_measures
    else:
        score_cot, dist_correl_cot, mse_cot, var_cot, kl_div_cot, js_div_cot, shap_plot_info_cot = 0, 0, 0, 0, 0, 0, 0

    # aggregate results
    atanasova_counterfact_count += atanasova_counterfact
    atanasova_input_from_expl_test_count += atanasova_input_from_expl
    cc_shap_post_hoc_sum += score_post_hoc
    turpin_test_count += turpin
    lanham_early_count += lanham_early
    lanham_mistake_count += lanham_mistake
    lanham_paraphrase_count += lanham_paraphrase
    lanham_filler_count += lanham_filler
    cc_shap_cot_sum += score_cot

    res_dict[f"{c_task}_{model_name}_{k}"] = {
        "input": formatted_input,
        "correct_answer": correct_answer,
        "model_input": ask_input,
        "model_prediction": prediction,
        "model_input_cot": ask_for_final_answer,
        "model_prediction_cot": prediction_cot,
        "accuracy": correct_predictions,
        "accuracy_cot": correct_predictions_cot,
        "atanasova_counterfact": atanasova_counterfact,
        "atanasova_input_from_expl": atanasova_input_from_expl_test_count,
        "cc_shap-posthoc": f"{score_post_hoc:.2f}",
        "turpin": turpin,
        "lanham_early": lanham_early,
        "lanham_mistake": lanham_mistake,
        "lanham_paraphrase": lanham_paraphrase,
        "lanham_filler": lanham_filler,
        "cc_shap-cot": f"{score_cot:.2f}",
        "other_measures_post_hoc": {
            "dist_correl": f"{dist_correl_ph:.2f}",
            "mse": f"{mse_ph:.2f}",
            "var": f"{var_ph:.2f}",
            "kl_div": f"{kl_div_ph:.2f}",
            "js_div": f"{js_div_ph:.2f}"
        },
        "other_measures_cot": {
            "dist_correl": f"{dist_correl_cot:.2f}",
            "mse": f"{mse_cot:.2f}",
            "var": f"{var_cot:.2f}",
            "kl_div": f"{kl_div_cot:.2f}",
            "js_div": f"{js_div_cot:.2f}"
        },
        "shap_plot_info_post_hoc": shap_plot_info_ph,
        "shap_plot_info_cot": shap_plot_info_cot,
    }

end_test = time.time()
time_elapsed = datetime.timedelta(seconds = end_test - start_test)

results_json = {
    "args": str(args),
    "model": {
        "full_model_name": full_model_name,
        "dtype": str(dtype),
    },
    "explainer": {
        "type": str(explainer),
        "max_evaluations": explainer_max_evaluations,
    },
    "tests": TESTS,
    "prediction_accuracy": {
        "accuracy": correct_predictions / num_samples,
        "accuracy_cot": correct_predictions_cot / num_samples,
    },
    "time_elapsed": str(time_elapsed),
    "samples": res_dict
}

# save results to a json file, make results_json directory if it does not exist
results_dir = Path("results_json")
if not results_dir.exists():
    results_dir.mkdir()

results_file_name = f"{c_task}_{model_name}_{count}_{explainer_type}.json"
results_file_path = results_dir.joinpath(results_file_name)

with results_file_path.open('w') as file:
    json.dump(results_json, file)


print(f"Ran {TESTS} on {c_task} data with model {model_name}. Reporting accuracy and faithfulness percentage.\n")
print(f"Accuracy %                  : {correct_predictions*100/count:.2f}  ")
print(f"Atanasova Counterfact %     : {atanasova_counterfact_count*100/count:.2f}  ")
print(f"Atanasova Input from Expl % : {atanasova_input_from_expl_test_count*100/count:.2f}  ")
print(f"CC-SHAP post-hoc mean score : {cc_shap_post_hoc_sum/count:.2f}  ")
print(f"Accuracy CoT %              : {correct_predictions_cot*100/count:.2f}  ")
print(f"Turpin %                    : {turpin_test_count*100/count:.2f}  ")
print(f"Lanham Early Answering %    : {lanham_early_count*100/count:.2f}  ")
print(f"Lanham Filler %             : {lanham_filler_count*100/count:.2f}  ")
print(f"Lanham Mistake %            : {lanham_mistake_count*100/count:.2f}  ")
print(f"Lanham Paraphrase %         : {lanham_paraphrase_count*100/count:.2f}  ")
print(f"CC-SHAP CoT mean score      : {cc_shap_cot_sum/count:.2f}  ")

logger.info(f"Tests are done. Time elapsed {time_elapsed}")
