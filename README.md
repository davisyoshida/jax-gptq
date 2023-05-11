# JAX-GPTQ
This is a JAX implementation of the pGPT-Q quantization algorithm](https://arxiv.org/abs/2210.17323).
It's currently a rough draft of the idea, but it makes using GPT-Q way easier than having to custom write the quantization loop for each model you want to apply it to.

I've tested it on my own Haiku models, and it also worked out of the box on the GPT-2 and T5 models from HuggingFace. 

I'll add documentation soon but for now [this notebook](https://github.com/davisyoshida/easy-lora-and-gptq) has a usage example.
