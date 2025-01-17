import datetime
import logging
import time
from typing import Literal
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.core.display import HTML

import shap

MODELS = {
    'bloom-7b1': 'bigscience/bloom-7b1',
    'opt-30b': 'facebook/opt-30b',
    'llama30b': '/workspace/mitarb/parcalabescu/llama30b_hf',
    'oasst-sft-6-llama-30b': '/workspace/mitarb/parcalabescu/transformers-xor_env/oasst-sft-6-llama-30b-xor/oasst-sft-6-llama-30b',
    'gpt2': 'gpt2',
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
    'mistral-7b-chat': 'mistralai/Mistral-7B-Instruct-v0.1',
    'falcon-7b': 'tiiuae/falcon-7b',
    'falcon-7b-chat': 'tiiuae/falcon-7b-instruct',
    'falcon-40b': 'tiiuae/falcon-40b',
    'falcon-40b-chat': 'tiiuae/falcon-40b-instruct',
}

logger = logging.getLogger("shap")


def full_model_name(short_name: str):
    return MODELS[short_name]

class Pipeline:
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name, dtype, max_new_tokens):
        start_loading_time = time.time()

        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                token=True
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            padding_side='left'
        )

        end_loading_time = time.time()
        loading_time = end_loading_time - start_loading_time
        timedelta = datetime.timedelta(seconds=loading_time)

        logger.info(f"Done loading model and tokenizer. Time elapsed: {timedelta}")

        model.generation_config.is_decoder = True
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.min_new_tokens = 1

        model.config.is_decoder = True # for older models, such as gpt2
        model.config.max_new_tokens = max_new_tokens
        model.config.min_new_tokens = 1

        return cls(model_name, model, tokenizer)

    def lm_generate(self, prompt, max_new_tokens, padding=False, repeat_input=True):
        """ Generate text from a huggingface language model (LM).
        Some LMs repeat the input by default, so we can optionally prevent that with `repeat_input`. """

        # Tokenizes the prompt to token ids
        input_ids = self.tokenizer([prompt], return_tensors="pt", padding=padding).input_ids.cuda()
        # Generate text
        generated_ids = self, self.model.generate(input_ids, max_new_tokens)

        # prevent the model from repeating the input
        if not repeat_input:
            generated_ids = generated_ids[:, input_ids.shape[1]:]

        # Decode output from token ids to text
        decoded_batch = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # There is only 1 sample in the batch
        generated_text = decoded_batch[0]
        return generated_text

    def lm_classify(self, prompt, padding=False, labels=['A', 'B']):
        """ Choose the token from a list of `labels` to which the LM asigns highest probability.
        https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/15"""

        # Tokenize the prompt to token ids
        input_ids = self.tokenizer([prompt], padding=padding, return_tensors="pt").input_ids.cuda()
        # Generate the next token based on the tokenized prompt.
        # We set max_new_tokens at 1 since we only want a single label e.g. A or B
        generated_ids = self.model.generate(
            input_ids,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            min_new_tokens=1
        )

        # find out which ids the labels have
        label_scores = np.zeros(len(labels))

        for i, label in enumerate(labels):
            single_token_return_models = ["gpt", "bloom", "falcon"]
            is_single_token_return_model = any(model in self.model_name for model in single_token_return_models)

            idx = 0 if is_single_token_return_model else 1 # the gpt2 model returns only one token
            label_id = self.tokenizer.encode(label)[idx] # TODO: check this for all new models: print(tokenizer.encode(label))

            label_scores[i] = generated_ids.scores[0][0, label_id]

        # Choose as label the one with the highest score
        label = labels[np.argmax(label_scores)]
        return label

    def explain_lm(self, prompt, explainer, max_new_tokens, plot: Literal["html", "display", "text"] | None = None):
        """ Compute Shapley Values for a certain model and tokenizer initialized in explainer."""

        if len(prompt) < 0:
            logger.error("Cant generate explanation for an empty prompt")
            raise Exception

        # Set the max new tokens amount
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.model.config.max_new_tokens = max_new_tokens
        # Use the explainer to create a shap explenation. This creates a
        # shap.Explanation object.
        # (Incorrect version) https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html#shap-explanation
        batch_prompts = [prompt]
        shap_vals = explainer(batch_prompts)

        if plot == 'html':
            HTML(shap.plots.text(shap_vals, display=False))
            with open(f"results_cluster/prompting_{self.model_name}.html", 'w') as file:
                file.write(shap.plots.text(shap_vals, display=False))
        elif plot == 'display':
            shap.plots.text(shap_vals)
        elif plot == 'text':
            print(' '.join(shap_vals.output_names))
        return shap_vals


# print(lm_generate(
#     'I enjoy walking with my cute dog.',
#     model,
#     tokenizer,
#     max_new_tokens=max_new_tokens)
# )

# lm_classify(
#     'When do I enjoy walking with my cute dog? On (A): a rainy day, or (B): a sunny day. The answer is: (',
#     model,
#     tokenizer,
#     labels=['Y', 'X', 'A', 'B', 'var' ,'Y']
# ) # somehow the model has two ',', ',' with different ids

# explain_lm(
#     'I enjoy walking with my cute dog', 
#     explainer, 
#     model_name, 
#     plot='display'
# )

