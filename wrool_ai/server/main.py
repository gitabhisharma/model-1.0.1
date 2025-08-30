"""
Minimal Text-Only AI Module
---------------------------------
Features:
- Pure text input -> text output (no images/audio)
- Single-file module with:
  - A pluggable TextGenerator class
  - Default local backend using Hugging Face Transformers (distilgpt2)
  - Simple safety/guardrails hook (very basic)
  - FastAPI endpoint /generate for HTTP use
  - CLI usage for quick testing

Requirements:
  pip install fastapi uvicorn pydantic transformers torch --upgrade

Run API server:
  uvicorn text_ai_module:app --reload --host 0.0.0.0 --port 8000

CLI usage:
  python text_ai_module.py --prompt "Write a haiku about monsoon in Mumbai."

Notes:
- The default model is small (distilgpt2) for easy local testing. Swap with a better model (e.g., "gpt2-medium" or an instruction-tuned model) if you have the compute.
- If you want to plug in a provider API (OpenAI, Groq, Together, etc.), implement the ProviderBackend class below using their official SDK.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from transformers import pipeline

# --------------------------
# Basic guardrails (toy)
# --------------------------

DEFAULT_BLOCKLIST = [
    r"\b(?i)(credit\s*card|ssn|password)\b",
]


def violates_basic_policy(text: str, patterns: Optional[List[str]] = None) -> Optional[str]:
    patterns = patterns or DEFAULT_BLOCKLIST
    for pat in patterns:
        if re.search(pat, text):
            return pat
    return None


# --------------------------
# Generation Config
# --------------------------

@dataclass
class GenConfig:
    model_name: str = os.getenv("LOCAL_MODEL", "distilgpt2")
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    device: str = os.getenv("DEVICE", "cpu")  # set to "cuda" if you have a GPU


# --------------------------
# Backends
# --------------------------

class LocalHFBackend:
    """Local text-generation backend using Hugging Face Transformers pipeline."""

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.pipe = pipeline(
            task="text-generation",
            model=cfg.model_name,
            device_map=None,
        )

    def generate(self, prompt: str, **overrides) -> str:
        cfg = self._merged_config(overrides)
        outputs = self.pipe(
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        # Transformers pipeline returns a list of dicts with 'generated_text'
        text = outputs[0]["generated_text"]
        # Return only the completion after the prompt for a cleaner result
        if text.startswith(prompt):
            return text[len(prompt):].lstrip("\n")
        return text

    def _merged_config(self, overrides: Dict[str, Any]):
        cfg = GenConfig(**{**self.cfg.__dict__, **overrides})
        return cfg


class ProviderBackend:
    """Template for plugging in a hosted LLM provider.
    Implement generate(self, prompt: str, **overrides) -> str using the provider SDK.
    """

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        # Initialize provider client here (e.g., OpenAI, Together, etc.)

    def generate(self, prompt: str, **overrides) -> str:
        raise NotImplementedError("Implement provider-backed generation here")


# --------------------------
# Text Generator (pluggable)
# --------------------------

class TextGenerator:
    def __init__(self, backend: Optional[str] = None, cfg: Optional[GenConfig] = None):
        self.cfg = cfg or GenConfig()
        backend = backend or os.getenv("BACKEND", "local")
        if backend == "local":
            self.impl = LocalHFBackend(self.cfg)
        elif backend == "provider":
            self.impl = ProviderBackend(self.cfg)  # fill in your provider
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate(self, prompt: str, **overrides) -> str:
        # Guardrails: block-listed prompts
        violated = violates_basic_policy(prompt)
        if violated:
            return (
                "Sorry, I can't help with that request. "
                "(Matched a restricted pattern.)"
            )

        # Simple system-style prefix (optional)
        system_prefix = (
            "You are a helpful, concise assistant. "
            "Answer directly without extra preamble.\n\n"
        )
        full_prompt = system_prefix + prompt.strip()
        return self.impl.generate(full_prompt, **overrides).strip()


# --------------------------
# FastAPI App
# --------------------------

app = FastAPI(title="Text-Only AI Module", version="1.0.0")

generator = TextGenerator()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=200)
    repetition_penalty: Optional[float] = Field(None, ge=0.5, le=2.0)


class GenerateResponse(BaseModel):
    completion: str
    usage: Dict[str, Any] = {}


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(payload: GenerateRequest):
    try:
        overrides = {k: v for k, v in payload.dict().items() if v is not None and k != "prompt"}
        text = generator.generate(payload.prompt, **overrides)
        return GenerateResponse(completion=text, usage={"backend": os.getenv("BACKEND", "local")})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
# CLI Entry
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal text-only AI module (local HF backend)")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()

    gen = TextGenerator()
    out = gen.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    print(out)


if __name__ == "__main__":
    main()
