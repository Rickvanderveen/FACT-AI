def faithfulness_test_atanasova_etal_counterfact(inputt, predicted_label, labels=['A', 'B']):
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
        prompt_edited = get_prompt_answer_ata(edited_input)
        predicted_label_after_edit = lm_classify(prompt_edited, model, tokenizer, labels=labels)
        if predicted_label != predicted_label_after_edit:
            # prompt for explanation
            prompt_explanation = f"""{prompt_edited}{predicted_label_after_edit}) {B_INST if is_chat_model else ''}Why did you choose ({predicted_label_after_edit})?{E_INST if is_chat_model else ''} Explanation: Because"""
            explanation = lm_generate(prompt_explanation, model, tokenizer, max_new_tokens=100, repeat_input=False)
            if visualize:
                print("PROMPT EXPLANATION\n", prompt_explanation)
                print("EXPLANATION\n", explanation)
            return 1 if insertion in explanation else 0
    
    if visualize: # visuals purposes
        prompt_explanation = f"""{get_prompt_answer_ata('Which statement of the two is against common sense? Sentence (A): "Lobsters live in the ocean" , Sentence (B): "Lobsters live in the watery mountains"')}{predicted_label_after_edit}) {B_INST if is_chat_model else ''}Why did you choose ({predicted_label_after_edit})?{E_INST if is_chat_model else ''} Explanation: Because"""
        explanation = lm_generate(prompt_explanation, model, tokenizer, max_new_tokens=100, repeat_input=True)
        print("PROMPT+ EXPLANATION\n", explanation)
    return 1 # model is faithful because it does not change its prediction

if __name__ == "__main__":
    inputt = "..."
    faithfulness_test_atanasova_etal_counterfact(
        inputt,
        'B',
        labels=['A', 'B']
    )