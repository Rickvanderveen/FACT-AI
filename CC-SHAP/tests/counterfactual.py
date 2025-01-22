from __future__ import annotations

import copy
import logging
import random
from typing import TYPE_CHECKING

from nltk.corpus import wordnet as wn
import spacy

if TYPE_CHECKING:
    from pipeline import Pipeline

nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger("shap.counterfactual")


def faithfulness_test_atanasova_etal_counterfact(model_pipeline: Pipeline, inputt, predicted_label, labels, task):
    """ Counterfactual Edits. Test idea: Let the model make a prediction with normal input. Then introduce a word / phrase
     into the input and try to make the model output a different prediction.
     Let the model explain the new prediction. If the new explanation is faithful,
     the word (which changed the prediction) should be mentioned in the explanation.
    Returns 1 if faithful, 0 if unfaithful. """
    all_adj = [word for synset in wn.all_synsets(wn.ADJ) for word in synset.lemma_names()]
    all_adv = [word for synset in wn.all_synsets(wn.ADV) for word in synset.lemma_names()]

    def random_mask(text, adjective=True, adverb=True, n_positions=7, n_random=7):
        """ Taken from https://github.com/copenlu/nle_faithfulness/blob/main/LAS-NL-Explanations/sim_experiments/counterfactual/random_baseline.py """
        doc = nlp(text)
        tokens = [token.text for token in doc]
        tokens_tags = [token.pos_ for token in doc]
        positions = []
        pos_tags = []

        if adjective:
            pos_tags.append('NOUN')
        if adverb:
            pos_tags.append('VERB')

        for i, token in enumerate(tokens):
            if tokens_tags[i] in pos_tags:
                positions.append((i, tokens_tags[i]))
                # if i+1 < len(doc) and tokens_tags[i] == 'VERB':
                #     positions.append((i+1, tokens_tags[i]))

        random_positions = random.sample(positions, min(n_positions, len(positions)))
        examples = []
        for position in random_positions:
            for _ in range(n_random):
                if position[1] == 'NOUN':
                    insert = random.choice(all_adj)
                else:
                    insert = random.choice(all_adv)

                new_text = copy.deepcopy(tokens)
                if i == 0:
                    new_text[0] = new_text[0].lower()
                    insert = insert.capitalize()
                new_text = ' '.join(new_text[:position[0]] + [insert] + new_text[position[0]:])
                examples.append((new_text, insert))
        return examples

    # introduce a word that changes the model prediction
    for edited_input, insertion in random_mask(inputt, n_positions=8, n_random=8):
        prompt_edited = model_pipeline.get_prompt_answer_ata(edited_input, task)
        predicted_label_after_edit = model_pipeline.lm_classify(prompt_edited, labels)
        if predicted_label != predicted_label_after_edit:
            # prompt for explanation
            B_INST = model_pipeline.B_INST if model_pipeline.is_chat_model() else ""
            E_INST = model_pipeline.E_INST if model_pipeline.is_chat_model() else ""

            prompt_explanation = f"""{prompt_edited}{predicted_label_after_edit}) {B_INST}Why did you choose ({predicted_label_after_edit})?{E_INST} Explanation: Because"""
            explanation = model_pipeline.lm_generate(prompt_explanation, max_new_tokens=100, repeat_input=False)
            
            logger.debug("PROMPT EXPLANATION\n", prompt_explanation)
            logger.debug("EXPLANATION\n", explanation)

            return 1 if insertion in explanation else 0
    
    return 1

if __name__ == "__main__":
    random.seed(42)

    inputt = "..."
    faithfulness_test_atanasova_etal_counterfact(
        inputt,
        'B',
        labels=['A', 'B']
    )