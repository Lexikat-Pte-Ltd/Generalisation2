#!/usr/bin/env python
from __future__ import annotations
import itertools, json, math, os, re, time
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Callable
from loguru import logger

from src.config import DreamConfig
from src.genner.Dream import DreamGenner
from src.types import Message, PList

try:
    import google.genai as genai

    _SDK = "genai"
except ImportError:
    import google.generativeai as genai

    _SDK = "generativeai"

DEFAULT_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

Score = float
ParamSet = Dict[str, Any]
ResultEntry = Tuple[ParamSet, Score, float]

_SYSTEM_PROMPT = r"""
You are an automated grader.

Return only JSON like {"score": 0.0}  
with a float 0–1 (two decimals ok).

Grading rubric for a Python answer to *invert a binary tree*:

1.00 – Perfect  
  - Defines `class TreeNode:` or uses LeetCode signature.  
  - Implements `def invert_tree(root)` (any valid name is ok).  
  - Handles `None`.  
  - Swaps left/right either recursively or with a stack/queue.  
  - Returns the (new) root.  
  - No logical errors.

0.70 – 0.99 – Nearly correct  
  - Algorithm right but minor issues (e.g. missing `return root`, no base-case docstring, etc.)

0.40 – 0.69 – Shows correct idea but code incomplete / wrong return or bad recursion depth check.

0.10 – 0.39 – Mentions tree inversion but gives only pseudocode or commentary, not runnable.

0.00 – 0.09 – Irrelevant or wrong.

Do not output anything except the JSON.
"""


@lru_cache
def _send(system_prompt: str,prompt: str) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY env var missing")

    if _SDK == "genai":  # new SDK
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=DEFAULT_MODEL_ID, contents=[prompt])
        return resp.text
    else:  # legacy SDK
        genai.configure(api_key=key)
        model = genai.GenerativeModel(DEFAULT_MODEL_ID)
        return model.generate_content(prompt).text


def grade_invert_tree(reply: str) -> float:
    prompt = (
        f"{_SYSTEM_PROMPT}\n\nUSER REQUEST:\n"
        "Write a Python function `invert_tree(root)` that inverts a binary tree "
        "and returns the new root. Include any helper `TreeNode` class if needed.\n\n"
        f"ASSISTANT REPLY:\n{reply}\n"
    )
    raw = _send(prompt)
    m = re.search(r"{.*}", raw, re.S)
    if not m:
        raise ValueError("Gemini returned unexpected payload: " + raw)
    return float(json.loads(m.group(0))["score"])


def _cfg(base: DreamConfig, over: ParamSet) -> DreamConfig:
    return base._replace(**{k: v for k, v in over.items() if hasattr(base, k)})


def _frange(a, b, s):
    return [round(a + i * s, 10) for i in range(int(math.floor((b - a) / s)) + 1)]


def make_space(g="medium"):
    if g not in {"coarse", "medium", "fine"}:
        raise ValueError
    return {
        "max_new_tokens": (
            [64, 128, 256] if g == "coarse" else [32, 64, 96, 128, 160, 192, 224, 256]
        ),
        "temperature": (
            _frange(0, 0.5, 0.25)
            if g == "coarse"
            else _frange(0, 1, 0.25 if g == "medium" else 0.05)
        ),
        "top_p": (
            _frange(0.8, 1, 0.2)
            if g == "coarse"
            else _frange(0.5, 1, 0.15 if g == "medium" else 0.05)
        ),
        "steps": [32, 64, 128] if g == "coarse" else [16, 32, 64, 96, 128, 160, 192],
        "top_k": [0, 20, 40] if g == "coarse" else [0, 10, 20, 30, 40, 50],
        "alg": ["entropy", "origin"],
        "alg_temp": (
            _frange(0, 0.3, 0.3)
            if g == "coarse"
            else _frange(0, 0.5, 0.25 if g == "medium" else 0.05)
        ),
        "delay": [0.0],
    }


def grid_search(
    msgs: PList,
    space: Dict[str, Iterable[Any]],
    obj: Callable[[str], Score],
    *,
    base: DreamConfig | None = None,
    k: int = 5,
    verbose: bool = False,
) -> List[ResultEntry]:
    base = base or DreamConfig()
    names = list(space)
    combos = [
        dict(zip(names, c)) for c in itertools.product(*(space[n] for n in names))
    ]
    logger.info("Evaluating {} combos …", len(combos))
    ranked = []
    for idx, ov in enumerate(combos, 1):
        cfg = _cfg(base, ov)
        gen = DreamGenner(cfg)
        t0 = time.perf_counter()
        try:
            reply = gen.plist_completion(msgs).unwrap()
            score = obj(reply)
            dt = time.perf_counter() - t0
            ranked.append((ov, score, dt))
            if verbose:
                logger.debug("[{}/{}] score={:.3f} {}", idx, len(combos), score, ov)
        except Exception as e:
            logger.exception("combo {} failed: {}", ov, e)
    ranked.sort(key=lambda x: (-x[1], x[2]))
    return ranked[:k]


if __name__ == "__main__":
    convo = PList(
        [
            Message(
                role="user",
                content="Write a Python function `invert_tree(root)` that inverts a binary tree and returns the new root. Include any helper TreeNode class if needed.",
            )
        ]
    )
    best = grid_search(
        convo, make_space("coarse"), grade_invert_tree, k=10, verbose=True
    )
    print("\nTop results (Gemini continuous scoring – invert binary tree):")
    for p, s, d in best:
        print(f"{p}\n  → score={s:.2f}   {d:.1f}s")
