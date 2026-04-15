from __future__ import annotations

"""
Utilities for retrieving seed survey items and generating new self-report items.

This module provides:
- retrieve_seeds: strict seed retrieval for target dimensions
- seed_texts_from_df: extract seed text strings from a dataframe
- GenerationResult: structured generation output
- SeededQuestionGenerator: prompt construction, generation, filtering, and export
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import gc
import re

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROMPT_TEMPLATE = """You are generating high-quality self-report mental health survey items.

Task:
Create ONE new self-report statement that reflects the following dimensions:
{target_dimensions}

STRICT REQUIREMENTS:
- Must be ONE clear declarative sentence (NOT a question)
- Must be concise, specific, and easy to understand
- Must use clear subject reference (avoid ambiguous or unclear pronouns)
- Must describe personal feelings, behaviors, or tendencies
- Must sound like a psychological self-report item
- Must NOT copy or closely rephrase any example
- Must NOT include contradictions, vague wording, or nonsensical phrases

STYLE GUIDELINES:
- Prefer simple and direct sentence structure
- Avoid unnecessary complexity or overly abstract wording
- Avoid phrases like:
  - "self-report"
  - "I have a strong immune system"
  - overly generic statements like "I am good"
- Avoid repeating the same structure as examples
- Aim for diversity in wording and meaning

GOOD EXAMPLE STYLE:
- "I am able to manage my emotions effectively during stressful situations."
- "I seek support from others when I feel overwhelmed."

BAD EXAMPLES (DO NOT DO):
- Questions (e.g., "How often do you feel sad?")
- Rewriting the same sentence structure
- Vague or unclear statements
- Ambiguous pronouns (e.g., "this", "that", "it" without clear meaning)
- Contradictory statements

Original examples:
{seed_examples}

New statement:
"""


def retrieve_seeds(
    df: pd.DataFrame,
    dimensions: list[str],
    target_dims: Sequence[str],
    *,
    k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve high-quality seed examples for a target dimension set.

    Strict filtering rule:
        min(target_dims) > mean(other_dims)

    Intuition:
    - All target dimensions must be consistently strong
    - Non-target dimensions should be comparatively weaker

    Ranking rule:
        score = target_mean - other_mean

    Args:
        df: Input dataframe containing one row per survey item/question.
        dimensions: Full list of dimension column names.
        target_dims: Target dimensions to retrieve seeds for.
        k: Number of top seed rows to return.

    Returns:
        A dataframe containing up to `k` seed rows.
        Returns an empty dataframe with the same columns as `df` if no valid seeds exist.
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    cleaned_target_dims = [dim.strip() for dim in target_dims if str(dim).strip()]
    if not cleaned_target_dims:
        raise ValueError("target_dims must contain at least one non-empty dimension.")

    missing_targets = [dim for dim in cleaned_target_dims if dim not in df.columns]
    if missing_targets:
        raise ValueError(
            f"Missing target dimension columns in dataframe: {missing_targets}"
        )

    missing_dimensions = [dim for dim in dimensions if dim not in df.columns]
    if missing_dimensions:
        raise ValueError(
            f"Missing dimension columns in dataframe: {missing_dimensions}"
        )

    other_dims = [dim for dim in dimensions if dim not in cleaned_target_dims]

    out = df.copy()

    # Aggregate target and non-target scores.
    out["target_mean"] = out[cleaned_target_dims].mean(axis=1)
    out["target_min"] = out[cleaned_target_dims].min(axis=1)

    if other_dims:
        out["other_mean"] = out[other_dims].mean(axis=1)
    else:
        out["other_mean"] = 0.0

    # Strict filter: every target dimension must exceed the average of non-target dimensions.
    mask = out["target_min"] > out["other_mean"]
    out = out.loc[mask].copy()

    if out.empty:
        return pd.DataFrame(columns=df.columns)

    # Rank rows by how strongly they favor the target dimensions.
    out["score"] = out["target_mean"] - out["other_mean"]
    out = out.sort_values("score", ascending=False)

    # Return only original columns to keep the result clean.
    return out.head(k)[df.columns].reset_index(drop=True)


def seed_texts_from_df(seed_df: pd.DataFrame, text_col: str = "text") -> list[str]:
    """
    Extract non-empty seed texts from a retrieved seed dataframe.

    Args:
        seed_df: Dataframe returned by `retrieve_seeds` or a compatible dataframe.
        text_col: Column containing the seed question text.

    Returns:
        A list of cleaned, non-empty text strings.
    """
    if text_col not in seed_df.columns:
        raise ValueError(f"Column '{text_col}' not found in seed dataframe.")

    return [
        str(text).strip()
        for text in seed_df[text_col].dropna().tolist()
        if str(text).strip()
    ]


@dataclass
class GenerationResult:
    """Structured output for collected survey item generations."""

    target_dims: list[str]
    prompt: str
    outputs: list[str]


class SeededQuestionGenerator:
    """
    Generate new self-report survey items conditioned on latent dimensions and seed examples.

    The generator:
    1. Builds a prompt from target dimensions and seed examples
    2. Produces raw generations from a seq2seq model
    3. Filters generations by length and n-gram similarity
    4. Optionally returns results as a pandas DataFrame
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the generator and load the tokenizer/model.

        Args:
            model_name: Hugging Face model name.
            device: Explicit device string, e.g. "cuda" or "cpu".
                If None, automatically selects CUDA when available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _clear_memory(self) -> None:
        """Run Python and CUDA memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_target_dims(self, target_dims: Sequence[str]) -> str:
        """Format a list of target dimensions into natural-language text."""
        dims = [dim.strip() for dim in target_dims if str(dim).strip()]

        if not dims:
            raise ValueError("target_dims must contain at least one non-empty dimension.")

        if len(dims) == 1:
            return dims[0]
        if len(dims) == 2:
            return f"{dims[0]} and {dims[1]}"
        return ", ".join(dims[:-1]) + f", and {dims[-1]}"

    def _build_prompt(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
    ) -> str:
        """
        Build the generation prompt from target dimensions and seed examples.

        Args:
            target_dims: Target latent dimensions.
            seed_texts: Example survey items used as seed references.

        Returns:
            A formatted prompt string.
        """
        if not seed_texts:
            raise ValueError("seed_texts must contain at least one example.")

        target_str = self._format_target_dims(target_dims)
        seed_block = "\n".join(
            f"- {text.strip()}" for text in seed_texts if str(text).strip()
        )

        return PROMPT_TEMPLATE.format(
            target_dimensions=target_str,
            seed_examples=seed_block,
        )

    def _clean_output(self, text: str) -> str:
        """
        Clean raw model output and keep only the first sentence.

        Args:
            text: Raw decoded model output.

        Returns:
            Cleaned single-sentence text.
        """
        text = text.strip()
        text = re.sub(r"^\s*[-•0-9.)]+\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return text

        parts = re.split(r"(?<=[.!?])\s+", text)
        return parts[0].strip()

    def _tokenize_words(self, text: str) -> list[str]:
        """Tokenize text into lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def _is_valid_length(self, text: str, *, min_words: int = 12) -> bool:
        """Check whether a candidate contains at least `min_words` tokens."""
        return len(self._tokenize_words(text)) >= min_words

    def _get_ngrams(self, text: str, n: int = 3) -> set[tuple[str, ...]]:
        """Extract word n-grams from text."""
        tokens = self._tokenize_words(text)
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def _ngram_jaccard(self, a: str, b: str, n: int = 3) -> float:
        """
        Compute Jaccard similarity between n-gram sets of two strings.

        Returns:
            Float in [0, 1].
        """
        a_ngrams = self._get_ngrams(a, n=n)
        b_ngrams = self._get_ngrams(b, n=n)

        if not a_ngrams or not b_ngrams:
            return 0.0

        return len(a_ngrams & b_ngrams) / len(a_ngrams | b_ngrams)

    def _is_too_similar(
        self,
        text: str,
        reference_texts: Sequence[str],
        *,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
    ) -> bool:
        """
        Check whether a candidate is too similar to any reference text.
        """
        for ref in reference_texts:
            if self._ngram_jaccard(text, ref, n=ngram_n) >= jaccard_threshold:
                return True
        return False

    def _accept_candidate(
        self,
        text: str,
        *,
        base_references: Sequence[str],
        collected: Sequence[str],
        min_words: int = 12,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
    ) -> bool:
        """
        Determine whether a generated candidate should be accepted.
        """
        if not text.strip():
            return False

        if not self._is_valid_length(text, min_words=min_words):
            return False

        if self._is_too_similar(
            text,
            reference_texts=base_references,
            ngram_n=ngram_n,
            jaccard_threshold=jaccard_threshold,
        ):
            return False

        if self._is_too_similar(
            text,
            reference_texts=collected,
            ngram_n=ngram_n,
            jaccard_threshold=jaccard_threshold,
        ):
            return False

        return True

    def generate(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
        *,
        batch_size: int = 20,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> list[str]:
        """
        Generate one raw batch of survey item candidates.

        This method performs generation only. It does not retry or filter outputs.

        Args:
            target_dims: Target latent dimensions.
            seed_texts: Seed example texts.
            batch_size: Number of returned sequences.
            max_new_tokens: Maximum number of newly generated tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            A list of cleaned generated strings.
        """
        prompt = self._build_prompt(target_dims, seed_texts)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=batch_size,
            )

        decoded = [
            self._clean_output(self.tokenizer.decode(output, skip_special_tokens=True))
            for output in outputs
        ]

        del outputs
        del inputs
        self._clear_memory()

        return decoded

    def collect(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
        *,
        n_questions: int = 30,
        existing_texts: Optional[Sequence[str]] = None,
        min_words: int = 12,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
        batch_size: int = 20,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_total_batches: int = 50,
        require_exact_count: bool = True,
    ) -> GenerationResult:
        """
        Collect valid generations across multiple batches until enough items are accepted.

        Args:
            target_dims: Target latent dimensions.
            seed_texts: Seed example texts.
            n_questions: Number of accepted outputs to collect.
            existing_texts: Existing reference texts to avoid duplicating.
            min_words: Minimum word count for accepted outputs.
            ngram_n: N-gram size for similarity checking.
            jaccard_threshold: Similarity threshold for rejection.
            batch_size: Number of samples per generation call.
            max_new_tokens: Maximum generated tokens per sample.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_total_batches: Maximum number of generation rounds.
            require_exact_count: Whether to raise an error if the target count is not reached.

        Returns:
            A GenerationResult containing the prompt and accepted outputs.
        """
        sorted_dims = sorted(target_dims)
        prompt = self._build_prompt(sorted_dims, seed_texts)

        if existing_texts is None:
            existing_texts = []

        collected: list[str] = []
        seen: set[str] = set()
        base_references = list(seed_texts) + list(existing_texts)

        batch_idx = 0
        while len(collected) < n_questions:
            batch_idx += 1

            if batch_idx > max_total_batches:
                if require_exact_count:
                    raise RuntimeError(
                        f"Could not collect {n_questions} valid outputs for {sorted_dims}. "
                        f"Collected {len(collected)} after {max_total_batches} batches."
                    )
                break

            raw_outputs = self.generate(
                target_dims=sorted_dims,
                seed_texts=seed_texts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            for text in raw_outputs:
                key = text.lower().strip()

                if not key or key in seen:
                    continue

                if not self._accept_candidate(
                    text,
                    base_references=base_references,
                    collected=collected,
                    min_words=min_words,
                    ngram_n=ngram_n,
                    jaccard_threshold=jaccard_threshold,
                ):
                    continue

                seen.add(key)
                collected.append(text)

                if len(collected) >= n_questions:
                    break

            del raw_outputs
            self._clear_memory()

        return GenerationResult(
            target_dims=list(sorted_dims),
            prompt=prompt,
            outputs=collected[:n_questions],
        )

    def collect_stream(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
        *,
        n_questions: int = 30,
        existing_texts: Optional[Sequence[str]] = None,
        min_words: int = 12,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
        batch_size: int = 20,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_total_batches: int = 50,
        require_exact_count: bool = True,
        progress_callback=None,
    ) -> GenerationResult:
        sorted_dims = sorted(target_dims)
        prompt = self._build_prompt(sorted_dims, seed_texts)

        if existing_texts is None:
            existing_texts = []

        collected: list[str] = []
        seen: set[str] = set()
        base_references = list(seed_texts) + list(existing_texts)

        batch_idx = 0
        while len(collected) < n_questions:
            batch_idx += 1

            if batch_idx > max_total_batches:
                if require_exact_count:
                    raise RuntimeError(
                        f"Could not collect {n_questions} valid outputs for {sorted_dims}. "
                        f"Collected {len(collected)} after {max_total_batches} batches."
                    )
                break

            raw_outputs = self.generate(
                target_dims=sorted_dims,
                seed_texts=seed_texts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            for text in raw_outputs:
                key = text.lower().strip()

                if not key or key in seen:
                    continue

                if not self._accept_candidate(
                    text,
                    base_references=base_references,
                    collected=collected,
                    min_words=min_words,
                    ngram_n=ngram_n,
                    jaccard_threshold=jaccard_threshold,
                ):
                    continue

                seen.add(key)
                collected.append(text)

                if progress_callback:
                    progress_callback(len(collected))

                if len(collected) >= n_questions:
                    break

            del raw_outputs
            self._clear_memory()

        return GenerationResult(
            target_dims=list(sorted_dims),
            prompt=prompt,
            outputs=collected[:n_questions],
        )

    def generate_to_df(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
        *,
        n_questions: int = 30,
        existing_texts: Optional[Sequence[str]] = None,
        min_words: int = 12,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
        batch_size: int = 20,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_total_batches: int = 50,
        require_exact_count: bool = True,
    ) -> pd.DataFrame:
        """
        Generate accepted outputs and return them as a DataFrame.
        """
        result = self.collect(
            target_dims=target_dims,
            seed_texts=seed_texts,
            n_questions=n_questions,
            existing_texts=existing_texts,
            min_words=min_words,
            ngram_n=ngram_n,
            jaccard_threshold=jaccard_threshold,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_total_batches=max_total_batches,
            require_exact_count=require_exact_count,
        )

        return pd.DataFrame(
            {
                "target_dims": [", ".join(result.target_dims)] * len(result.outputs),
                "generated_text": result.outputs,
            }
        )