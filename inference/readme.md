# Loading Amber 

```python
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber", revision="ckpt_356")
model = LlamaForCausalLM.from_pretrained("LLM360/Amber", revision="ckpt_356")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```
