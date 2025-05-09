import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Dream-org/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
device = "cuda:1"
model = model.to(device).eval()

messages = [
    {
        "role": "user",
        "content": "Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset.",
    }
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device=device)
attention_mask = inputs.attention_mask.to(device=device)


print("#" * 80)
print("`alg_temp`: 0.0")
print("#" * 80)
try:
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        output_history=True,
        return_dict_in_generate=True,
        steps=10,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
    )
    generations = [
        tokenizer.decode(g[len(p) :].tolist())
        for p, g in zip(input_ids, output.sequences)
    ]

    print(generations[0].split(tokenizer.eos_token)[0])
except Exception as e:
    print(f"Error during generation, {e}")

print("#" * 80)
print("`alg_temp`: 0.5")
print("#" * 80)
try:
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        output_history=True,
        return_dict_in_generate=True,
        steps=10,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.5,
    )
    generations = [
        tokenizer.decode(g[len(p) :].tolist())
        for p, g in zip(input_ids, output.sequences)
    ]

    print("#" * 80)
    print(generations[0].split(tokenizer.eos_token)[0])
    print("#" * 80)
except Exception as e:
    print(f"Error during generation, {e}")

print("#" * 80)
print("`alg_temp`: 0.0")
print("#" * 80)
try:
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        output_history=True,
        return_dict_in_generate=True,
        steps=50,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
    )
    generations = [
        tokenizer.decode(g[len(p) :].tolist())
        for p, g in zip(input_ids, output.sequences)
    ]

    print(generations[0].split(tokenizer.eos_token)[0])
except Exception as e:
    print(f"Error during generation, {e}")
print("#" * 80)
