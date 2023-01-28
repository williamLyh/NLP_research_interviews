# Coding Practicals
## Huggingface
### Is it possible to use batch generation for Huggingface models?
It is possible to make huggingface model to do batch generation. However, it requires tokenizer to initilize with pad_side='left'. Below is an example:

Greedy Decoding:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
>>> tokenizer.pad_token = tokenizer.eos_token
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> prompt = ["Today I believe we can finally", "Tomorrow I believe we can finally"] 
>>> input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids

>>> # generate up to 30 tokens
>>> outputs = model.generate(input_ids, max_length=30)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Line breaker in GPT2 tokenizer
In the pretrain of GPT2, "\n" is used as a line breaker. Therefore we could continue using it during fine-tuning. 
It has token id of 198. However, we shouldn't check it with tokenizer.convert_id_to_token(198), which will return "Ċ". 
The tokenizer.convert_token_to_id("\n") will give 50256, which is the id of "<|endoftext|>".
The right way to check to id is by encode and decode functions.
