from __future__ import annotations

"""Minimal model selector for real-world samples.

Defaults to Gemini via GOOGLE_API_KEY and a fast model.
Swap or extend as needed (e.g., Ollama, LangChain) mirroring examples/models.py.
"""

import os
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import genai_model
from loguru import logger
from google.genai import types as genai_types


def build_turn_model(
    system_instruction: content_api.ProcessorContentTypes | None = None,
    *,
    model_name: str | None = None,
) -> processor.Processor:
    """Returns a turn-based LLM processor.

    Args:
      system_instruction: Optional system instruction content (string or Parts).
      model_name: Optional Gemini model name override.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Export GOOGLE_API_KEY to use Gemini."
        )
    if not model_name:
        model_name = "gemini-2.0-flash-lite"

    generate_cfg = genai_types.GenerateContentConfig(
        system_instruction=[
            p.text for p in content_api.ProcessorContent(system_instruction or [])
        ],
        response_modalities=["TEXT"],
    )

    return genai_model.GenaiModel(
        api_key=api_key,
        model_name=model_name,
        generate_content_config=generate_cfg,
        http_options=genai_types.HttpOptions(api_version="v1alpha"),
    )


def build_caption_model(
    *,
    model_name: str | None = None,
    language: str | None = "ja",
    allow_inference: bool = True,
    temperature: float = 0.2,
    max_output_tokens: int = 48,
    max_words: int | None = 18,
) -> processor.Processor:
    """Returns a turn-based model configured for factual media captioning.

    The model is instructed to output concise, factual alt-text only (TEXT).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Export GOOGLE_API_KEY to use Gemini."
        )
    if not model_name:
        # Prefer a multimodal-capable model for images/audio/video facts.
        model_name = "gemini-2.0-flash"

    # System instruction tuned for high‑quality alt-text.
    # – Single concise sentence, noun-phrase lead, no fluff.
    # – Only verifiable visual facts. Include key subject(s), count, color, action, salient context.
    # – If clear readable text is present, transcribe only essential words exactly as seen (no paraphrase).
    # – Do not guess brand/model names unless explicitly readable in the image.
    # – Avoid ‘A/An/The’ at the start; begin with the subject noun phrase.
    # – Output plain text only.
    lang_line = "回答は日本語で。一文、簡潔に。" if (language or "").lower().startswith("ja") else (
        f"Respond in {language}. One sentence, concise." if language else "One sentence, concise."
    )
    si_core = (
        "You are an alt-text captioner for images, audio, and video. "
        "Write one concise sentence with verifiable visual facts. "
        "Start with a noun phrase (no leading 'A', 'An', or 'The'). "
        "Mention main subject, count, colors, actions, and salient context. "
        "If large, clear on-screen text is visible, transcribe essential words exactly as seen. "
    )

    if allow_inference:
      si_spec = (
          "You may include inferred details when you are highly confident based on the asset and general world knowledge. "
          "For medium confidence, use qualifiers like 'likely' or 'probably'. If confidence is low, omit the inference. "
          "Only include brand/model names if unambiguous from distinctive features or readable markings; otherwise qualify or omit. "
          "Avoid opinions and avoid inventing text or numbers. "
      )
    else:
      si_spec = (
          "Do not speculate beyond what is visibly verifiable. "
          "Do not include brand/model names unless directly readable. "
          "Avoid opinions and any invented text or numbers. "
      )

    length_hint = (
        f" Limit to at most {max_words} words." if (max_words and (language or "en").lower().startswith("en")) else ""
    )
    si_text = si_core + si_spec + (" " + lang_line if lang_line else "") + length_hint
    si = [si_text]

    cfg = genai_types.GenerateContentConfig(
        system_instruction=[p.text for p in content_api.ProcessorContent(si)],
        response_modalities=["TEXT"],
        response_mime_type="text/plain",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    logger.info(
        "Caption model configured: model_name={} language={} allow_inference={} temp={} max_tokens={} si_len={}",
        model_name, language, allow_inference, temperature, max_output_tokens, len(si_text),
    )

    return genai_model.GenaiModel(
        api_key=api_key,
        model_name=model_name,
        generate_content_config=cfg,
        http_options=genai_types.HttpOptions(api_version="v1alpha"),
    )
