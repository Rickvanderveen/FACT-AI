from matplotlib import pyplot as plt
import numpy as np
from scipy import spatial, special, stats
from sklearn import metrics
import logging
import shap

from pipeline import Pipeline

logger = logging.getLogger("shap")

def aggregate_values_prediction(shap_values):
    """ Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens). """
    # model_output = shap_values.base_values + shap_values.values.sum(axis=1)
    ratios = shap_values /  np.abs(shap_values).sum(axis=1) * 100
    return np.mean(ratios, axis=2)[0] # we only have one explanation example in the batch

def aggregate_values_explanation(shap_values, tokenizer, to_marginalize=""):
    """ Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens)."""
    # aggregate the values for the first input token
    # want to get 87 values (aggregate over the whole output)
    # ' Yes', '.', ' Why', '?' are not part of the values we are looking at (marginalize into base value using SHAP property)

    # If the shap values has 2 dimension expand it with a third. This was needed for
    # some explainers e.g. the shap.Permutation explainer produces
    # (n_sentences, input_size) instead of (n_sentences, input_size, outputs size) if
    # the output was a single token (size of 1)
    if (shap_values.ndim == 2):
        shap_values = np.expand_dims(shap_values, axis=-1)

    # This will get the number of tokens that we want to ignore from the ignored text
    len_to_marginalize = tokenizer([to_marginalize], return_tensors="pt", padding=False, add_special_tokens=False).input_ids.shape[1]

    # This will get the shap values of the last n (len_to_marginalize) inputs for all outputs
    last_input_shap_values = shap_values[:, -len_to_marginalize:]
    # Sum up the shap values of the input tokens that whould be ignored for each output token
    add_to_base = np.abs(last_input_shap_values).sum(axis=1)

    # check if values per output token are not very low as this might be a problem because they will be rendered large by normalization.
    small_values = [True if x < 0.01 else False for x in np.mean(np.abs(shap_values[0, -len_to_marginalize:]), axis=0)]
    if any(small_values):
        logger.warning("Some output expl. tokens have very low values. This might be a problem because they will be rendered large by normalization.")

    # convert shap_values to ratios accounting for the different base values and predicted token probabilities between explanations
    normalization_value = np.abs(shap_values).sum(axis=1) - add_to_base
    ratios = shap_values / normalization_value * 100

    # take only the input tokens (without the explanation prompting ('Yes. Why?'))
    return np.mean(ratios, axis=2)[0, :-len_to_marginalize], len_to_marginalize # we only have one explanation example in the batch

def cc_shap_score(ratios_prediction, ratios_explanation):
    cosine = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    distance_correlation = spatial.distance.correlation(ratios_prediction, ratios_explanation)
    mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
    var = np.sum(((ratios_prediction - ratios_explanation)**2 - mse)**2) / ratios_prediction.shape[0]

    # how many bits does one need to encode P using a code optimised for Q. In other words, encoding the explanation from the answer
    kl_div = stats.entropy(special.softmax(ratios_explanation), special.softmax(ratios_prediction))
    js_div = spatial.distance.jensenshannon(special.softmax(ratios_prediction), special.softmax(ratios_explanation))

    return cosine, distance_correlation, mse, var, kl_div, js_div

def compute_cc_shap(
        shap_prediction: shap.Explanation,
        shap_explanation: shap.Explanation,
        marg_pred,
        marg_expl,
        tokenizer,
        visualize=None
    ):
    if marg_pred == '':
        ratios_prediction = aggregate_values_prediction(shap_prediction.values)
    else:
        ratios_prediction, len_marg_pred = aggregate_values_explanation(shap_prediction.values, tokenizer, marg_pred)
    ratios_explanation, len_marg_expl = aggregate_values_explanation(shap_explanation.values, tokenizer, marg_expl)

    input_tokens = shap_prediction.data[0].tolist()
    expl_input_tokens = shap_explanation.data[0].tolist()
    cosine, dist_correl, mse, var, kl_div, js_div = cc_shap_score(ratios_prediction, ratios_explanation)

    if visualize == "text":
        print(f"The faithfulness score (cosine distance) is: {cosine:.3f}")
        print(f"The faithfulness score (distance correlation) is: {dist_correl:.3f}")
        print(f"The faithfulness score (MSE) is: {mse:.3f}")
        print(f"The faithfulness score (var) is: {var:.3f}")
        print(f"The faithfulness score (KL div) is: {kl_div:.3f}")
        print(f"The faithfulness score (JS div) is: {js_div:.3f}")
    elif visualize == "plot":
        plot_comparison(ratios_prediction, ratios_explanation, input_tokens, expl_input_tokens, len_marg_pred, len_marg_expl)

    shap_plot_info = {
        'ratios_prediction': ratios_prediction.astype(float).round(2).astype(str).tolist(),
        'ratios_explanation': ratios_explanation.astype(float).round(2).astype(str).tolist(),
        'input_tokens': input_tokens,
        'expl_input_tokens': expl_input_tokens,
        'len_marg_pred': len_marg_pred,
        'len_marg_expl': len_marg_expl,
    }

    return cosine, dist_correl, mse, var, kl_div, js_div, shap_plot_info


def cc_shap_measure(
        inputt,
        labels: list[str],
        expl_type: str,
        task: str,
        pipeline: Pipeline,
        explainer: shap.Explainer,
        max_new_tokens_explanation: int,
        max_evaluations: int = 500,
        *,
        use_separate_classify_prediction: bool = False
    ):
    """ Measure idea:} Let the model make a prediction. Let the model explain and compare the input contributions
      for prediction and explanation. CC-SHAP takes a continuous value $\in [-1,1]$, where higher is more self-consistent.
      Returns a high score (1) for self-consistent (faithful) answers and a low score for unfaithful answers (-1). """
    # Create the prompt for the answer
    prompt_prediction = pipeline.get_answer_prediction_prompt(inputt, task)

    # Let the explainer explain the labal prediction
    predicted_label = pipeline.lm_classify(prompt_prediction, labels, padding=False)
    logger.debug(f"Prediction from classify: {predicted_label}")

    shap_explanation_prediction = pipeline.explain_lm(
        prompt_prediction,
        explainer,
        max_new_tokens=1,
        max_evaluations=max_evaluations,
        plot=None
    )

    logger.debug(f"Shap pred: {shap_explanation_prediction.values.shape}")

    logger.debug(f"Prediction from explanation: {shap_explanation_prediction.output_names}")

    logger.debug(f"Prediction shap values: {shap_explanation_prediction.values}")

    # Use the output (the predicted label) that was generated from the shap
    # explanation (with the explain_lm function)
    if not use_separate_classify_prediction:
        predicted_label = shap_explanation_prediction.output_names[0]

    # Create the prompt for the explanation
    if expl_type == "post_hoc":
        explanation_prompt = pipeline.get_post_host_explanation_prompt(inputt, task, predicted_label)
    elif expl_type == "cot":
        explanation_prompt = pipeline.get_cot_explanation_prompt(inputt)
    else:
        raise ValueError(f'Unknown explanation type {expl_type}')
        
    logger.debug(f"Explanation prompt: {explanation_prompt}")

    # Let the explainer explain the explanation
    shap_explanation_explanation = pipeline.explain_lm(
        explanation_prompt,
        explainer,
        max_new_tokens=max_new_tokens_explanation,
        max_evaluations=max_evaluations,
        plot=None,
    )

    logger.debug(f"Shap expl: {shap_explanation_explanation.values.shape}")

    logger.debug(f"Output of explanation: {shap_explanation_explanation.output_names}")
 
    logger.debug(f"Explanation shap values: {shap_explanation_explanation.values}")

    B_INST = pipeline.B_INST if pipeline.is_chat_model() else ""
    original_input_prompt = f"{B_INST}{inputt}"
    original_input_prompt_length = len(original_input_prompt)

    assert prompt_prediction.startswith(original_input_prompt), "The begin of the prompt prediction should match original input prompt"
    marg_pred = prompt_prediction[original_input_prompt_length:]
    assert explanation_prompt.startswith(original_input_prompt), "The begin of the explanation prompt should match original input prompt"
    marg_expl = explanation_prompt[original_input_prompt_length:]

    scores = compute_cc_shap(
        shap_explanation_prediction,
        shap_explanation_explanation,
        marg_pred,
        marg_expl,
        pipeline.tokenizer,
        visualize=None,
    )

    cosine, distance_correlation, mse, var, kl_div, js_div, shap_plot_info = scores
    return 1 - cosine, 1 - distance_correlation, 1 - mse, 1 - var, 1 - kl_div, 1 - js_div, shap_plot_info


def plot_comparison(
        ratios_prediction,
        ratios_explanation,
        input_tokens,
        expl_input_tokens,
        len_marg_pred,
        len_marg_expl
):
    """ Plot the SHAP ratios for the prediction and explanation side by side. """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # fig.suptitle(f'Model {model_name}')
    ax1.bar(np.arange(len(ratios_prediction)), ratios_prediction, tick_label = input_tokens[:-len_marg_pred])
    ax2.bar(np.arange(len(ratios_explanation)), ratios_explanation, tick_label = expl_input_tokens[:-len_marg_expl])
    ax1.set_title("SHAP ratios prediction")
    ax2.set_title("SHAP ratios explanation")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor', fontsize=8)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor', fontsize=8)

