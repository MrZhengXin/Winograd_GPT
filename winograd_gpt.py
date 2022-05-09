from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import random


'''
dataset = load_dataset("winograd_wsc", 'wsc273')['test']
with open('wsc273.jsonl', 'w') as f:
    for instance in dataset:
        print(instance, file=f)
'''

with open('wsc273.jsonl', 'r') as f:
    dataset = f.readlines()
    dataset = [eval(instance) for instance in dataset]

# model_name= 'gpt2-large' # 0.717948717948718
# model_name = 'gpt2-xl' # 0.7326007326007326
# model_name = 'EleutherAI/gpt-neo-2.7B' # 0.7362637362637363
model_name = 'EleutherAI/gpt-j-6B' # 0.8131868131868132
# model_name = 'EleutherAI/gpt-neox-20b'
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Calculate the log prob sum of the text AFTER the pronoun
def compute_score(model, suffix_len, inputs): 
    with torch.no_grad():
        inputs = tokenizer(inputs, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        logits = model(**inputs, labels=inputs['input_ids']).logits
        log_probs = logits.log_softmax(-1)[:, -suffix_len-1:-1:1, :]
        gen_probs = torch.gather(log_probs, -1, inputs['input_ids'][:, -suffix_len:, None]).squeeze(-1)
        # gen_token_len = (gen_probs != float("-Inf")).count_nonzero(dim=1)
        gen_probs[gen_probs == float("-Inf")] = 0
        log_prob_sums = gen_probs.sum(dim=1) # / gen_token_len
    return log_prob_sums.item()

hit = 0
total = 0

for instance in dataset:
    total += 1
    text = instance['text'].strip()
    if text[-1] != '.':
        text += '.'
    option_0, option_1 = instance['options'][0], instance['options'][1]
    quote, pronoun = instance['quote'], instance['pronoun']
    if not pronoun[0].isupper():
        if option_0.startswith('The'):
            option_0 = 't' +option_0[1:]
        if option_1.startswith('The'):
            option_1 = 't' +option_1[1:]
    if pronoun in ['his', 'her', 'their', 'its']:
        option_0 = option_0 + "'s"
        option_1 = option_1 + "'s"
    prefix = text[:instance['pronoun_loc']-1]
    suffix = text[instance['pronoun_loc'] + len(pronoun)+1:]
    suffix_len = len(tokenizer.encode(' ' + suffix))
    text_with_option_0 = ' '.join([prefix, option_0, suffix])
    text_with_option_1 = ' '.join([prefix, option_1, suffix])
    score_text_with_option_0 = compute_score(model, suffix_len, text_with_option_0)
    score_text_with_option_1 = compute_score(model, suffix_len, text_with_option_1)
    choice = 0 if score_text_with_option_0 > score_text_with_option_1 else 1
    hit += 1 if choice == instance['label'] else 0
    # print(hit / total)
    if choice != instance['label']:
        print(instance, prefix, text_with_option_0, \
            score_text_with_option_0, \
            text_with_option_1, \
            score_text_with_option_1, sep='\n', end='\n\n')

acc = hit / total
print(acc)
