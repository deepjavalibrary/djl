import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, torchscript=True)

# %% model_inputs
output_attentions = False
output_hidden_states = False
model_inputs = {}

model_inputs['past_key_values'] = torch.load(
    "../data/nested_tuple_" + model_name + ".pt")
past_seq = model_inputs['past_key_values'][0][0].shape[-2]
model_inputs['input_ids'] = torch.tensor([[404]])
model_inputs['position_ids'] = torch.tensor([[past_seq]])
# |attention_mask| = `len(past_key_values) + len(input_ids)`
model_inputs['attention_mask'] = torch.ones(past_seq + 1, dtype=torch.int64)

model_inputs['use_cache'] = True
model_inputs['token_type_ids'] = None

model_inputs['return_dict'] = False
model_inputs['output_attentions'] = False
model_inputs['output_hidden_states'] = False

# This is a testing of text generation
outputs = model(**model_inputs)

# %% Wrapper class of GPT2LMHeadModel
from typing import Tuple

class Tracable(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, torchscript=True)
        self.config = {'use_cache': config.get('use_cache', True),
                       'token_type_ids': config.get('token_type_ids', None),
                       'return_dict': config.get('return_dict', False),
                       'output_attentions': config.get('output_attentions', False),
                       'output_hidden_states': config.get('output_hidden_states', True)}

    def forward(self, my_input_ids, position_ids, attention_mask, past_key_values):
        return self.model(input_ids=my_input_ids,
                          position_ids=position_ids,
                          attention_mask=attention_mask,
                          past_key_values=past_key_values,
                          **self.config)  # return_tensor = True

# %% create class
config = {}
tracable = Tracable(config)
input = (model_inputs['input_ids'],
         model_inputs['position_ids'],
         model_inputs['attention_mask'],
         model_inputs['past_key_values'])

output = tracable(*input)

# %% trace
tracable.eval()

traced_model = torch.jit.trace(tracable, input)
torch.jit.save(traced_model, "../traced_GPT2_hidden.pt")

out1 = traced_model(*input)

# %% load back
loaded_model = torch.jit.load("../traced_GPT2_hidden.pt")
out2 = loaded_model(*input)
