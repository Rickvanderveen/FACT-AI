from pipeline import Pipeline
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
logger = logging.getLogger("LOO")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = encoder.to(device)
def faithfulness_loo_test_slow(
        inputt,
        labels: list[str],
        expl_type: str,
        task: str,
        pipeline: Pipeline,
        max_new_tokens_explanation: int,
        threshold: float,
    ):
    #----------------
    # Perform LOO on prediction prompt (Instead of the SHAP epxlainer)
    #-------------
    # Create the prompt for the answer
    prompt_prediction = pipeline.get_answer_prediction_prompt(inputt, task)
    inputt_length = len(inputt.split())
    # Get the original model prediction
    predicted_label = pipeline.lm_classify(prompt_prediction, labels, padding=False)
    logger.debug(f"Original Prediction: {predicted_label}")
    # Split the prompt into seperate words (to leave one out)
    loo_scores_prediction = []
    words = prompt_prediction.split()
    for i in range(inputt_length):
        modified_prompt = " ".join(words[:i] + words[i+1:])  # Remove one word at a time
        new_prediction = pipeline.lm_classify(modified_prompt, labels, padding=False)
        
        # Measure if the prediction changed
        impact = 1 if new_prediction != predicted_label else 0
        loo_scores_prediction.append(impact)
    #-------------
    #----------------
    # Perform LOO on explanation prompt (Instead of the SHAP epxlainer)
    #-------------
    # Choose explanation type: Post-hoc or CoT
    if expl_type == "post_hoc":
        explanation_prompt = pipeline.get_post_host_explanation_prompt(inputt, task, predicted_label)
    elif expl_type == "cot":
        explanation_prompt = pipeline.get_cot_explanation_prompt(inputt)
    else:
        raise ValueError(f'Unknown explanation type {expl_type}')
    
    logger.debug(f"Explanation prompt: {explanation_prompt}")
    
    # Generate the explanation using the explanation prompt
    generated_explanation = pipeline.lm_generate(explanation_prompt, max_new_tokens_explanation, repeat_input=False)
    generated_explanation_embedding = encoder.encode([generated_explanation]) # used to check similarity
    # Perform LOO on explanation prompt
    loo_scores_explanation = []
    words = explanation_prompt.split()
    # Only loop the same length as the prediction, since we compare it to that part of the prompt (it will still use the entire prompt, in lm_generate)
    # Its just less loops to be more efficient
    for i in range(inputt_length):
        modified_prompt = " ".join(words[:i] + words[i+1:])  # Remove one word at a time
        new_generated_explanation = pipeline.lm_generate(modified_prompt, max_new_tokens_explanation, repeat_input=False)
        new_generated_explanation_embedding = encoder.encode([new_generated_explanation]) # used to check similarity
        # Measure if the explanation changed
        similarity = cosine_similarity(generated_explanation_embedding, new_generated_explanation_embedding)[0][0]
        
        # similarity is a 0-1 value, if its 1.0 its similar, so it had little impact removing it
        # Therefore if the similarity is <= the threshold, it had impact
        impact = 1 if similarity <= threshold else 0
        loo_scores_explanation.append(impact)
    #------------
    # Compute scores
    #------------
    # Convert to numpy for easier similarity calculations
    loo_scores_prediction = np.array(loo_scores_prediction).reshape(1, -1)
    loo_scores_explanation = np.array(loo_scores_explanation).reshape(1, -1)
    # Compute similarity metrics
    cosine_sim = cosine_similarity(loo_scores_prediction.reshape(1, -1), loo_scores_explanation.reshape(1, -1))[0][0]
    mse = mse = np.mean((loo_scores_prediction - loo_scores_explanation) ** 2)
    logger.debug(f"LOO Cosine Similarity: {cosine_sim}")
    logger.debug(f"LOO MSE: {mse}")
    return cosine_sim, mse
