"""Local machine translation helper (no external API, no rate limits).

Heavy dependencies (torch, transformers, sentencepiece) are only imported
inside `translate_texts`, so importing this module stays cheap and the rest
of the codebase doesn't need them installed. See requirements-nlp.txt.
"""


def translate_texts(
    texts: list[str],
    model_name: str = "Helsinki-NLP/opus-mt-en-es",
    batch_size: int = 64,
) -> list[str]:
    """Translate a list of texts using a local MarianMT model, in batches."""
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translations: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=60
        )
        generated = model.generate(**encoded, max_length=60)
        translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return translations
