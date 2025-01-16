import numpy as np
from scipy import spatial, special, stats
from sklearn import metrics

# TODO: Check if this method is neccessary or that it is almost the same as
# aggregate_values_explanation with an to_marginalize="" (empty string) or maybe None
def aggregate_values_prediction(shap_values):
    """ Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens). """
    # model_output = shap_values.base_values + shap_values.values.sum(axis=1)
    ratios = shap_values.values /  np.abs(shap_values.values).sum(axis=1) * 100
    return np.mean(ratios, axis=2)[0] # we only have one explanation example in the batch

def aggregate_values_explanation(shap_values, tokenizer, to_marginalize=""):
    """ Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens)."""
    # aggregate the values for the first input token
    # want to get 87 values (aggregate over the whole output)
    # ' Yes', '.', ' Why', '?' are not part of the values we are looking at (marginalize into base value using SHAP property)

    # This will get the number of tokens that we want to ignore from the ignored text
    len_to_marginalize = tokenizer([to_marginalize], return_tensors="pt", padding=False, add_special_tokens=False).input_ids.shape[1]

    # This will get the shap values of the last n (len_to_marginalize) inputs for all outputs
    last_input_shap_values = shap_values.values[:, -len_to_marginalize:]
    # Sum up the shap values of the input tokens that whould be ignored for each output token
    add_to_base = np.abs(last_input_shap_values).sum(axis=1)

    # check if values per output token are not very low as this might be a problem because they will be rendered large by normalization.
    small_values = [True if x < 0.01 else False for x in np.mean(np.abs(shap_values.values[0, -len_to_marginalize:]), axis=0)]
    if any(small_values):
        print("Warning: Some output expl. tokens have very low values. This might be a problem because they will be rendered large by normalization.")

    # convert shap_values to ratios accounting for the different base values and predicted token probabilities between explanations
    normalization_value = np.abs(shap_values.values).sum(axis=1) - add_to_base
    ratios = shap_values.values / normalization_value * 100

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

def compute_cc_shap(values_prediction, values_explanation, marg_pred='', marg_expl=' Yes. Why?', visualize=None):
    if marg_pred == '':
        ratios_prediction = aggregate_values_prediction(values_prediction)
    else:
        ratios_prediction, len_marg_pred = aggregate_values_explanation(values_prediction, marg_pred)
    ratios_explanation, len_marg_expl = aggregate_values_explanation(values_explanation, marg_expl)

    input_tokens = values_prediction.data[0].tolist()
    expl_input_tokens = values_explanation.data[0].tolist()
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