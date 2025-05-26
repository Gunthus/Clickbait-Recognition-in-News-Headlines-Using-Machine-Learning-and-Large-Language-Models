import os
import logging
import time
from typing import List, Optional

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY is not set; zero-shot classification will fail.")

_PROMPT_TEMPLATE = """Uzdevums: Klasificē šo virsrakstu – "{headline}"
Izvēlies vienu no šīm klasēm:
1 – nav klikšķēsma
2 – daļēja klikšķēsma
3 – ir klikšķēsma
Atbildi tikai ar skaitli."""


def classify_zero_shot(
    headline: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    timeout: int = 30
) -> int:
    """
    Classify a single Latvian headline into 1/2/3, per the thesis specifications.
    """
    prompt = _PROMPT_TEMPLATE.format(headline=headline)

    try:
        if not model.startswith("gpt-"):
            raise ValueError(f"Unsupported model: {model}")

        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            request_timeout=timeout
        )
        output = response.choices[0].message.content.strip()

        if output not in {"1", "2", "3"}:
            raise ValueError(f"Expected '1','2', or '3', got: {output!r}")

        return int(output)

    except Exception as e:
        logger.error(f"Error classifying headline {headline!r}: {e}")
        raise


def classify_zero_shot_batch(
    headlines: List[str],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    timeout: int = 30,
    delay_between_requests: float = 0.1
) -> List[Optional[int]]:
    """
    Batch classification: returns a list of ints in the same order as `headlines`.
    """
    results = []
    total = len(headlines)

    for idx, h in enumerate(headlines, start=1):
        logger.info("Classifying %d/%d", idx, total)
        try:
            results.append(classify_zero_shot(h, model=model, temperature=temperature, timeout=timeout))
        except Exception:
            results.append(None)

        if idx < total:
            time.sleep(delay_between_requests)

    successful = sum(r is not None for r in results)
    logger.info("Batch complete: %d/%d successful", successful, total)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-shot Latvian clickbait classifier (3-class)"
    )
    parser.add_argument("headline", nargs="+", help="Headline(s) to classify")
    parser.add_argument(
        "--model", default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4-1106-preview"],
        help="OpenAI model"
    )
    parser.add_argument("--api-key", help="OpenAI API key")

    args = parser.parse_args()
    if args.api_key:
        openai.api_key = args.api_key

    for h in args.headline:
        try:
            label = classify_zero_shot(h, model=args.model)
            print(f"{h!r} → {label}")
        except Exception as e:
            print(f"Error classifying {h!r}: {e}")
