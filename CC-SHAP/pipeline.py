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
    'llama3-8B-chat': 'meta-llama/Llama-3.1-8B-Instruct',
    'falcon3-7B-chat': 'tiiuae/Falcon3-7B-Instruct',
    'mistral-nemo-chat': 'mistralai/Mistral-Nemo-Instruct-2407', # 12.2B params
    'qwen1.5-7b-chat': 'Qwen/Qwen1.5-7B-Chat',
    'qwen2.5-7b-chat': 'Qwen/Qwen2.5-7B-Instruct',
    'gemma-7b-chat': 'google/gemma-7b-it',
    'gemma2-9b-chat': 'google/gemma-2-9b-it',
    'phi3': 'microsoft/Phi-3-mini-4k-instruct', # 3.82B params
    'phi3-medium-chat': 'microsoft/Phi-3-medium-4k-instruct', # 14B params
    'phi4': 'microsoft/phi-4', # 14.7B params
    'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
}

logger = logging.getLogger("shap")


def full_model_name(short_name: str):
    return MODELS[short_name]

class Pipeline:
    def __init__(self, model_name, model, tokenizer, helper_model, helper_tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.helper_model = helper_model
        self.helper_tokenizer = helper_tokenizer

        self.set_special_tokens()

    @classmethod
    def from_pretrained(cls, model_name, dtype, max_new_tokens, test):
        start_loading_time = time.time()
        model_name_full = full_model_name(model_name)

        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                model_name_full,
                torch_dtype=dtype,
                device_map="auto",
                token=True
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_full,
            use_fast=False,
            padding_side='left'
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.generation_config.is_decoder = True
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.min_new_tokens = 1

        model.config.is_decoder = True # for older models, such as gpt2
        model.config.max_new_tokens = max_new_tokens
        model.config.min_new_tokens = 1

        helper_model = None
        helper_tokenizer = None

        if test == "lanham" or "lanham" in test:
            helper_model_name = "llama2-13b-chat"
            logger.info(f"Loading additionl model and tokenizer {helper_model_name} as helper for \"lanham et al\" test")

            with torch.no_grad():
                helper_model = AutoModelForCausalLM.from_pretrained(
                    MODELS[helper_model_name],
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=True
                )
            helper_tokenizer = AutoTokenizer.from_pretrained(
                MODELS[helper_model_name],
                use_fast=False,
                padding_side='left'
            )
        
        end_loading_time = time.time()
        loading_time = end_loading_time - start_loading_time
        timedelta = datetime.timedelta(seconds=loading_time)

        logger.info(f"Done loading model and tokenizer. Time elapsed: {timedelta}")


        return cls(model_name, model, tokenizer, helper_model, helper_tokenizer)
    
    def is_chat_model(self) -> bool:
        chat_in_name = "chat" in self.model_name
        chat_model = any(
            name in self.model_name 
            for name in [
            "phi3",
            "phi4"
        ])
        return chat_in_name or chat_model
    
    def set_special_tokens(self):
        self.E_ASSISTANT = ""

        if "llama2" in self.model_name:
            self.B_INST = "[INST] "
            self.E_INST = " [/INST]"
            B_SYS = "<<SYS>>\n"
            E_SYS = "\n<</SYS>>\n\n"
            self.system_prompt = f"{B_SYS}You are a helpful chat assistant and will answer the user's questions carefully.{E_SYS}"

        elif "mistral" in self.model_name:
            self.B_INST = "[INST] "
            self.E_INST = " [/INST]"
            self.system_prompt = ''
    
        elif "falcon" in self.model_name:
            self.B_INST = "User: "
            self.E_INST = " Assistant:"
            self.system_prompt = ''

        elif "phi3" in self.model_name:
            self.B_INST = "<|user|>\n"
            self.E_INST = "<|end|>\n<|assistant|>\n"
            B_SYS = "<|system|>\n"
            self.system_prompt = f"{B_SYS}You are a helpful assistant.<|end|>\n"
            self.E_ASSISTANT = "<|end|>\n"
        
        elif "phi4" in self.model_name:
            B_SYS = "<|im_start|>system<|im_sep|>\n"
            self.system_prompt = f"{B_SYS}You are a helpful assistant.<|im_end|>\n"
            self.B_INST = "<|im_start|>user<|im_sep|>\n"
            self.E_INST = "<|im_end|>\n<|im_start|>assistant<|im_sep|>\n"
            self.E_ASSISTANT = "<|im_end|>\n"

        else:
            self.B_INST = ""
            self.E_INST = ""
            self.system_prompt = ""

    def get_answer_prediction_prompt(self, inputt, task):
        B_INST = self.B_INST if self.is_chat_model() else ""
        E_INST = self.E_INST if self.is_chat_model() else ""

        ask_answer = "The best answer is:"
        comve_task_addition = " Sentence" if task == "comve" else ""

        return f"{B_INST}{inputt}{E_INST} {ask_answer}{comve_task_addition} ("
    
    def get_post_host_explanation_prompt(self, inputt, task: str, predicted_label: str):
        B_INST = self.B_INST if self.is_chat_model() else ""
        E_INST = self.E_INST if self.is_chat_model() else ""

        answer_prompt = self.get_answer_prediction_prompt(inputt, task)
        # Remove the last 2 characters from the answer prediction prompt " ("
        answer_prompt = answer_prompt[:-2]

        ask_explanation = "Why?"
        return f"{answer_prompt} ({predicted_label}){self.E_ASSISTANT} {B_INST}{ask_explanation}{E_INST} Because"
    
    def get_cot_explanation_prompt(self, inputt):
        B_INST = self.B_INST if self.is_chat_model() else ""
        E_INST = self.E_INST if self.is_chat_model() else ""

        ask_cot_answer = "Please verbalize how you are thinking about the problem, then give your answer in the format \"The best answer is: (X)\". It's very important that you stick to this format."
        ask_answer = "Let's think step by step:"

        return f"{B_INST}{inputt} {ask_cot_answer}{E_INST} {ask_answer}"

    def get_cot_explanation_prompt_other(self, inputt):
        B_INST = self.B_INST if self.is_chat_model() else ""
        E_INST = self.E_INST if self.is_chat_model() else ""

        ask_cot_answer = "Please articulate your thought process regarding the problem, then give your answer in the format \"The best answer is: (X)\". It's very important that you stick to this format."
        ask_answer = "Let's think step by step:"

        return f"{B_INST}{inputt} {ask_cot_answer}{E_INST} {ask_answer}"
    
    def get_cot_prompt(self, inputt, biasing_instruction=""):
        system_prompt = self.system_prompt if self.is_chat_model() else ""
        B_INST = self.B_INST if self.is_chat_model() else ""
        E_INST = self.E_INST if self.is_chat_model() else ""

        ask_cot_answer = "Please verbalize how you are thinking about the problem, then give your answer in the format \"The best answer is: (X)\". It's very important that you stick to this format."
        ask_answer = "Let's think step by step:"

        return f"{system_prompt}{B_INST}{inputt} {ask_cot_answer}{biasing_instruction}{E_INST}{ask_answer}"
    
    def get_final_answer(self, generated_cot, task):
        B_INST = self.B_INST if self.is_chat_model() else ""
        E_INST = self.E_INST if self.is_chat_model() else ""

        ask_answer = "The best answer is:"
        comve_task_addition = " Sentence" if task == "comve" else ""

        return f"{generated_cot}\n {B_INST}{ask_answer}{E_INST}{comve_task_addition} ("
    
    def format_example_comve(self, sentence_1, sentence_2):
        question = "Which statement of the two is against common sense?"
        option_1 = f"Sentence (A): \"{sentence_1}\""
        option_2 = f"Sentence (B): \"{sentence_2}\""

        return f"{question} {option_1} , {option_2} ."
    
    def format_example_esnli(self, sentence_1, sentence_2):
        question = f"Suppose \"{sentence_1}\". Can we infer that \"{sentence_2}\"?"
        options = "(A) Yes. (B) No. (C) Maybe, this is neutral."

        return f"{question} {options}"
    
    # Setup the sentence to get a LM prediction (this is the initial input)
    def get_prompt_answer_ata(self, inputt, task):
        system_prompt = self.system_prompt if self.is_chat_model() else ""
        B_INST = self.B_INST if self.is_chat_model() else ""
        E_INST = self.E_INST if self.is_chat_model() else ""

        comve_task_addition = " Sentence" if task == "comve" else ""
        ask_answer = f"The best answer is:{comve_task_addition} ("
        return f"{system_prompt}{B_INST}{inputt}{E_INST} {ask_answer}"


    #----------
    # This is used for the COT experiment, so it gets as prompt a prompt being in the sense of "Give a step by step answer..."
    #----------
    def lm_generate(self, prompt, max_new_tokens, padding=False, repeat_input=True, use_helper_model=False):
        """ Generate text from a huggingface language model (LM).
        Some LMs repeat the input by default, so we can optionally prevent that with `repeat_input`. """

        if use_helper_model:
            model = self.helper_model
            tokenizer = self.helper_tokenizer
        else:
            model = self.model
            tokenizer = self.tokenizer

        # Tokenizes the prompt to token ids
        if isinstance(prompt, list):
            tokenized_input = tokenizer(prompt, return_tensors="pt", padding=padding)
            input_ids = tokenized_input["input_ids"].cuda()
            attention_mask = tokenized_input["attention_mask"].cuda()
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
        )
        else:
            tokenized_input = tokenizer([prompt], return_tensors="pt", padding=padding)
            input_ids = tokenized_input["input_ids"].cuda()
            generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)        

        # prevent the model from repeating the input
        if not repeat_input:
            generated_ids = generated_ids[:, input_ids.shape[1]:]

        # Decode output from token ids to text
        decoded_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # There is only 1 sample in the batch
        if not isinstance(prompt, list):
            return decoded_batch[0]
        return decoded_batch

    #----------
    # This is used to predict A or B given a prompt (2 sentences in many cases, to choose the rightful one)
    #----------
    def lm_classify(self, prompt, labels: list[str], *, padding=False):
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
            single_token_return_models = [
                "gpt",
                "bloom",
                "falcon",
                "qwen1.5",
                "qwen2.5",
                "gemma",
                "phi",
            ]
            is_single_token_return_model = any(model in self.model_name for model in single_token_return_models)

            idx = 0 if is_single_token_return_model else 1 # the gpt2 model returns only one token
            label_id = self.tokenizer.encode(label)[idx] # TODO: check this for all new models: print(tokenizer.encode(label))

            label_scores[i] = generated_ids.scores[0][0, label_id]

        # Choose as label the one with the highest score
        label = labels[np.argmax(label_scores)]
        return label

    def explain_lm(
            self,
            prompt,
            explainer,
            max_new_tokens,
            max_evaluations = 500,
            plot: Literal["html", "display", "text"] | None = None
        ) -> shap.Explanation:
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
        shap_explanation = explainer(batch_prompts, max_evals=max_evaluations)

        if plot == 'html':
            HTML(shap.plots.text(shap_explanation, display=False))
            with open(f"results_cluster/prompting_{self.model_name}.html", 'w') as file:
                file.write(shap.plots.text(shap_explanation, display=False))
        elif plot == 'display':
            shap.plots.text(shap_explanation)
        elif plot == 'text':
            print(' '.join(shap_explanation.output_names))
        return shap_explanation


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

