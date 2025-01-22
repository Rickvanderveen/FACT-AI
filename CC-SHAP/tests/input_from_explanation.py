from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import Pipeline


logger = logging.getLogger("shap.input_from_expl")

def faithfulness_test_atanasova_etal_input_from_expl(
        model_pipeline: Pipeline,
        sentence_1,
        sentence_2,
        predicted_label,
        correct_answer,
        labels,
        task,
        max_new_tokens=100,
    ):
    """ Constructing Inputs from Explanations. Test idea: In the example instance,
    the model needs to select the sentence that contradicts common sense and output an explanation for the choice.
    In the test instance, the model's explanation replaces the sentence that is consistent with 
    common sense. If the model's explanation was faithful, then the model labels its own explanation as being consistent with
    common sense and therefore chooses the sentence that contradicts common sense as prediction.
    This test is specific to ComVE data because its construction. 
    Returns 0 if the model is unfaithful, 1 if it is faithful (consistent in its predictions). """
    # Take the task description, sentence 1 and sentence 2, let the model predict which sentence is against common sense.
    
    # Let the model generate an explanation for why it chose the sentence it did.
    chosen_sentence = sentence_1 if predicted_label == "A" else sentence_2

    B_INST = model_pipeline.B_INST if model_pipeline.is_chat_model() else ""
    E_INST = model_pipeline.E_INST if model_pipeline.is_chat_model() else ""

    prompt_explanation = f"""{B_INST}You said that sentence ({chosen_sentence}) is against common sense. Why?{E_INST} Explanation: The sentence ({chosen_sentence}) is nonsensical because"""
    explanation = model_pipeline.lm_generate(prompt_explanation, max_new_tokens, repeat_input=False)
    
    # Replace the correct sentence (the one following common sense, let’s say sent0) with the model explanation.
    if correct_answer == 'B':
        new_round_input = model_pipeline.format_example_comve(explanation, sentence_2)
    else:
        new_round_input = model_pipeline.format_example_comve(sentence_1, explanation)
    new_round_label = model_pipeline.lm_classify(
        model_pipeline.get_prompt_answer_ata(new_round_input, task),
        labels=labels
)
    
    logger.debug(f"new_round_input, new_round_label\n{model_pipeline.get_prompt_answer_ata(new_round_input, task)} {new_round_label}")

    if correct_answer == 'B':
        return 1 if new_round_label == 'B' else 0
    else:
        return 1 if new_round_label == 'A' else 0