import os

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import extract_clean_text

load_dotenv()

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

PROMPT_TEMPLATE = """
### system: You are an helpful assistant who returns correct and concise answers.
### human: {}
### response:
"""

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_HUB_TOKEN")


def init_checkpoints(model_name=MODEL_NAME):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        rope_scaling={"type": "dynamic", "factor": 2.0},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer


def get_completion(prompt, tokenizer, model, max_new_tokens=4000, do_sample=False):
    """Generate a completion for the given prompt using a pre-trained language model."""
    prompt = PROMPT_TEMPLATE.format(prompt)
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    gen_out = model.generate(**model_inputs, **generate_kwargs)
    response = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    clean_response = extract_clean_text(response)
    return clean_response
