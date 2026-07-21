#!/usr/bin/env python
"""Cache known Hugging Face models in a Transformers 3.x-compatible layout."""

import argparse
import contextlib
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


MODEL_FILES = {
    "roberta-base": [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    "gpt2": [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
}

DEFAULT_MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def download_from_resolve(repo_id: str, filename: str, destination: Path) -> None:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    response = requests.get(url, stream=True)
    if response.status_code == 404:
        raise FileNotFoundError(filename)
    response.raise_for_status()

    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    with open(tmp_destination, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    tmp_destination.replace(destination)


def cache_model(model_name: str, cache_dir: Path) -> Path:
    model_dir = cache_dir / model_name.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)

    required_files = {"config.json", "pytorch_model.bin"}
    filenames = MODEL_FILES.get(model_name, DEFAULT_MODEL_FILES)

    for filename in filenames:
        destination = model_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue

        try:
            with contextlib.redirect_stdout(sys.stderr):
                hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    cache_dir=str(model_dir),
                    force_filename=filename,
                )
        except Exception as exc:
            try:
                download_from_resolve(model_name, filename, destination)
            except Exception as fallback_exc:
                if filename not in required_files:
                    continue
                raise RuntimeError(
                    f"Could not cache {model_name}/{filename}: {fallback_exc}"
                ) from exc

    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name or local path")
    parser.add_argument("--cache-dir", default="model_files", help="Directory for cached model files")
    args = parser.parse_args()

    if Path(args.model).exists():
        print(args.model)
        return 0

    model_dir = cache_model(args.model, Path(args.cache_dir))
    print(model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
