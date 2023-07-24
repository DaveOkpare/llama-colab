from utils import extract_clean_text


PROMPT_TEMPLATE = """
### system: You are an helpful assistant who returns correct and concise answers.
### human: {}
### response:
"""


def get_completion(prompt, tokenizer, model, max_new_tokens=4000, do_sample=False):
    """Generate a completion for the given prompt using a pre-trained language model."""
    prompt = PROMPT_TEMPLATE.format(prompt)
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    gen_out = model.generate(**model_inputs, **generate_kwargs)
    response = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    clean_response = extract_clean_text(response)
    return clean_response
