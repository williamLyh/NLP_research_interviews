# Coding Practicals
## Techniques
**Einstein Summation**
Einstein Summation (Einsum) is a handy tool for matrix operation. It can do everything, including inner product, batch product, summation or elementwise product. Please refer to the [Blog](https://www.cnblogs.com/qftie/p/16245124.html)

## Huggingface
### Is it possible to use batch generation for Huggingface models?
As described in this [link](https://github.com/huggingface/transformers/pull/7552#issue-497255933), it is possible to make huggingface model to do batch generation. However, it requires tokenizer to initilize with pad_side='left'. 
Below is an example:

Greedy Decoding:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
>>> tokenizer.pad_token = tokenizer.eos_token
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> prompt = ["Today I believe we can finally", "Tomorrow I believe we can finally"] 
>>> batch_input = tokenizer(prompt, padding=True, return_tensors="pt")

>>> # generate up to 30 tokens
>>> outputs = model.generate(batch_input.input_ids, attention_mask=batch_input.attention_mask, max_length=30)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Line breaker in GPT2 tokenizer
In the pretrain of GPT2, "\n" is used as a line breaker. Therefore we could continue using it during fine-tuning. 
It has token id of 198. However, we shouldn't check it with tokenizer.convert_id_to_token(198), which will return "Ċ". 
The tokenizer.convert_token_to_id("\n") will give 50256, which is the id of "<|endoftext|>". The "\n\n" is also another predefined token, whose id is 628.
The right way to check to id is by encode and decode functions. 

### Difference between attention_mask and token_id of -100 (in GPT2):
The attention_mask will mask input such that it won't be passed into the model in the first place. 
The token_id of -100 is usually set for padding tokens in labels. It won't be calculated by the loss function. As stated in this link.
