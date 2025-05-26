import logging
import re
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
device = 0 if torch.cuda.is_available() else -1
if torch.cuda.is_available():
    logger.info("CUDA available, using GPU")
else:
    logger.warning("CUDA not available, using CPU; this will be slower")

tokenizer = None
model = None
_generator = None

PROMPT_TEMPLATE = (
    "Uzdevums: Klasificē šo virsrakstu – \"{headline}\"\n"
    "Izvēlies vienu no šīm klasēm:\n"
    "1 – nav klikšķēsma\n"
    "2 – daļēja klikšķēsma\n"
    "3 – ir klikšķēsma\n"
    "Atbildi tikai ar skaitli."
)

def _load_model():
    """Lazy load the model and tokenizer."""
    global tokenizer, model, _generator
    if _generator is not None:
        return _generator

    logger.info("Loading Mistral 7B model...")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    _generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    logger.info("Model loaded successfully")
    return _generator

def classify_mistral(
    headline: str,
    max_new_tokens: int = 10,
    temperature: float = 0.1
) -> int:
    """Classify a single Latvian headline using Mistral 7B Instruct."""
    generator = _load_model()
    prompt = PROMPT_TEMPLATE.format(headline=headline)
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )[0]["generated_text"]
    response = output[len(prompt):].strip()

    match = re.search(r"[123]", response)
    if match:
        return int(match.group(0))

    label_str = response.strip().split()[-1] if response.strip() else ""
    if label_str not in {"1", "2", "3"}:
        raise ValueError(f"Unexpected output: {response!r}")
    return int(label_str)

def classify_mistral_batch(
    headlines: List[str],
    max_new_tokens: int = 10,
    temperature: float = 0.1
) -> List[Optional[int]]:
    """Batch classify a list of headlines."""
    results: List[Optional[int]] = []
    total = len(headlines)

    for i, h in enumerate(headlines, start=1):
        logger.info("Mistral classifying %d/%d", i, total)
        try:
            results.append(
                classify_mistral(h, max_new_tokens=max_new_tokens, temperature=temperature)
            )
        except Exception as e:
            logger.error("Error on %r: %s", h, e)
            results.append(None)

    successful = sum(r is not None for r in results)
    logger.info("Batch complete: %d/%d successful", successful, total)
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-shot Mistral 7B Instruct clickbait classifier"
    )
    parser.add_argument(
        "headline", nargs="+",
        help="Headline(s) to classify"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature"
    )
    args = parser.parse_args()

    for h in args.headline:
        try:
            label = classify_mistral(
                h,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            print(f"{h!r} → {label}")
        except Exception as e:
            print(f"Error classifying {h!r}: {e}")
