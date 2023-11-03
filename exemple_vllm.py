from vllm import LLM, SamplingParams
import torch
import os

model_path = os.environ['DSDIR']+'/HuggingFace_Models/' + 'bloom-7b1'

prompts = [
    "Dear citizens,",
    "The capital of France is",
]
print(prompts)
sampling_params = SamplingParams(temperature=0.8)

llm = LLM(model=model_path, dtype="bfloat16", tensor_parallel_size=8) # tensor parallel size = number of gpus
print("model loaded")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
