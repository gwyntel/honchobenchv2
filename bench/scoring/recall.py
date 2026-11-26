"""Stage 1 — recall scorer: honcho conclusions vs labels.json explicit_facts.

For each labeled conversation, match every ``explicit_fact`` against the
conclusions honcho derived from it and verdict each fact **covered /
partial / missing** (CoverageBench template, upstream
``tests/bench/coverage.py`` @ v3.7.0).

Two matching paths, in order:

1. **Decision-model matcher (the verdict path).** One Score question per
   (conclusion, fact) pair — ``covers_fact`` — composed per the
   ``building-with-decision-models`` skill: atomic, literal, state as a
   named JSON object, score levels as *situations* not degrees. A second
   Noul, ``overclaims``, flags conclusions that claim more than the source
   supports. The judge is an injectable callable so the scorer stays
   backend-free: tests inject a stub, Stage 2 wires the real decision
   judge (Plexus alias or local MLX) through the same interface.
2. **Embedding-similarity pre-filter (optional).** Cosine similarity
   between fact and conclusion text, keeping the top-k candidates per
   fact. Pure-Python, embedder injectable; with no embedder the matcher
   sees every pair.

What ``Conclusion.source_message_ids`` (new in v3.7.0) buys here: it does
*not* decide the verdict — ``labels.json`` facts carry no message refs,
so a fact cannot be linked to messages directly. It is recorded as a
**grounding check** on every match (``linked``: the matched conclusion
cites real source messages) and reported as ``linked_share`` among
covered facts. Full fact→message linkage needs hand labels per fact.

Importance weighting is CoverageBench-style (critical / important /
minor / trivial) but **starts unweighted**: labels carry no importance
yet, so every fact defaults to weight 1.0. Pass per-fact weights in when
hand labels exist — no code change needed.

Thresholds are starting points, not gospel (the Jev calibration lesson
from the decision-judge work: calibrate against hand labels before
trusting them). They live in the constants below, next to the questions.

No live compute in this module: no honcho, no Plexus, no model calls.
It scores conclusion objects you hand it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (skill rule 11: questions, thresholds, weights in one module)
# ---------------------------------------------------------------------------

RECALL_QUESTIONS = {
    "covers_fact": {
        "type": "score",
        "instructions": (
            "Does `conclusion.content` state the fact in `fact.text`? "
            "Judge only what the conclusion says, not what you know about "
            "the conversation. A rephrased fact still counts if it keeps "
            "the fact's core claim, names, numbers, and conditions."
        ),
        "levels": [
            "The conclusion does not state the fact at all.",
            "The conclusion states the core of the fact but leaves out a "
            "key part: a name, a number, a condition, or a qualifier the "
            "fact includes.",
            "The conclusion states the fact fully, possibly rephrased.",
        ],
    },
    "overclaims": {
        "type": "noul",
        "instructions": (
            "Does `conclusion.content` claim something that the fact in "
            "`fact.text` and the conclusion's cited source messages do not "
            "support? Answer yes only if the conclusion adds a claim of "
            "its own beyond the fact."
        ),
    },
}

# Score (0..2, may be fractional) -> verdict. Calibrate against hand labels.
COVERED_THRESHOLD = 1.5
PARTIAL_THRESHOLD = 0.5
OVERCLAIM_THRESHOLD = 0.5  # Noul P(yes) >= this -> overclaimed

# CoverageBench-style importance weights. Labels carry no importance yet,
# so every fact defaults to 1.0 (unweighted) until hand labels exist.
IMPORTANCE_WEIGHTS = {
    "critical": 3.0,
    "important": 2.0,
    "minor": 1.0,
    "trivial": 0.5,
}
DEFAULT_WEIGHT = 1.0

# Pre-filter defaults.
PREFILTER_TOP_K = 5


# ---------------------------------------------------------------------------
# Loading / normalizing
# ---------------------------------------------------------------------------

def load_labels(path: str | Path) -> dict:
    """Load labels.json: {conversation_id: {explicit_facts: [...], ...}}."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def normalize_conclusions(raw) -> list[dict]:
    """Accept a conclusions/list response or a plain list; return flat dicts.

    Keeps ``content`` plus the v3.7.0 evidence fields (``source_message_ids``,
    ``source_ids``) and ``id`` when present. Anything else passes through.
    """
    if isinstance(raw, dict):
        raw = raw.get("conclusions") or raw.get("items") or []
    out = []
    for c in raw:
        if isinstance(c, str):
            c = {"content": c}
        out.append({
            "id": c.get("id"),
            "content": c.get("content", ""),
            "source_message_ids": c.get("source_message_ids") or [],
            "source_ids": c.get("source_ids") or [],
        })
    return out


# ---------------------------------------------------------------------------
# Embedding pre-filter (pure Python; embedder injectable)
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def prefilter_pairs(facts: list[str], conclusions: list[dict], embed,
                    top_k: int = PREFILTER_TOP_K) -> dict[int, list[int]]:
    """Per fact index, the top-k conclusion indices by cosine similarity.

    ``embed(texts)`` returns one vector per text. With no embedder (None),
    every fact keeps every conclusion (no filtering).
    """
    if embed is None:
        return {i: list(range(len(conclusions))) for i in range(len(facts))}
    vectors = embed(facts + [c["content"] for c in conclusions])
    fact_vecs, concl_vecs = vectors[:len(facts)], vectors[len(facts):]
    out: dict[int, list[int]] = {}
    for i, fv in enumerate(fact_vecs):
        ranked = sorted(
            range(len(concl_vecs)),
            key=lambda j: cosine(fv, concl_vecs[j]),
            reverse=True,
        )
        out[i] = ranked[:top_k]
    return out


# ---------------------------------------------------------------------------
# Matcher — one decision-model call per (conclusion, fact) pair
# ---------------------------------------------------------------------------

def verdict_for_score(score: float) -> str:
    if score >= COVERED_THRESHOLD:
        return "covered"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "missing"


def judge_fact(fact: str, conclusion: dict, judge_fn) -> dict:
    """Score one (conclusion, fact) pair through the injected judge.

    ``judge_fn(state, questions)`` follows the decision-model contract:
    state is a named JSON object, questions are ``RECALL_QUESTIONS``, and
    it returns ``{"covers_fact": {"score": 0..2},
    "overclaims": {"probability": 0..1}}``. Out-of-range answers raise
    instead of being silently clamped.
    """
    state = {"conclusion": conclusion, "fact": {"text": fact}}
    answers = judge_fn(state, RECALL_QUESTIONS)
    score = float(answers["covers_fact"]["score"])
    over_prob = float(answers["overclaims"]["probability"])
    if not 0.0 <= score <= 2.0:
        raise ValueError(f"covers_fact score out of range: {score}")
    if not 0.0 <= over_prob <= 1.0:
        raise ValueError(f"overclaims probability out of range: {over_prob}")
    return {
        "score": score,
        "verdict": verdict_for_score(score),
        "overclaims": over_prob >= OVERCLAIM_THRESHOLD,
    }


def score_conversation(convo_id: str, facts: list[str],
                       conclusions: list[dict], judge_fn, embed=None,
                       weights: dict | None = None) -> dict:
    """Score one conversation. Returns per-fact verdicts + aggregates.

    ``weights`` maps fact text -> importance label (critical / important /
    minor / trivial); omitted facts default to weight 1.0.
    """
    conclusions = normalize_conclusions(conclusions)
    candidates = prefilter_pairs(facts, conclusions, embed)
    fact_results = []
    overclaimed: list[dict] = []
    for i, fact in enumerate(facts):
        best = None
        for j in candidates.get(i, []):
            judged = judge_fact(fact, conclusions[j], judge_fn)
            judged["conclusion_id"] = conclusions[j].get("id")
            judged["linked"] = bool(conclusions[j].get("source_message_ids"))
            if best is None or judged["score"] > best["score"]:
                best = judged
            if judged["overclaims"] and not any(
                    o["conclusion_id"] == conclusions[j].get("id")
                    for o in overclaimed):
                overclaimed.append({
                    "conclusion_id": conclusions[j].get("id"),
                    "content": conclusions[j]["content"],
                    "fact": fact,
                })
        if best is None:
            best = {"score": 0.0, "verdict": "missing",
                    "overclaims": False, "conclusion_id": None,
                    "linked": False}
        elif best["score"] < PARTIAL_THRESHOLD:
            # No conclusion actually matches: keep the best score for
            # calibration diagnostics, but there is no match to ground.
            best = {**best, "conclusion_id": None, "linked": False,
                    "overclaims": False}
        weight = IMPORTANCE_WEIGHTS.get(
            (weights or {}).get(fact, ""), DEFAULT_WEIGHT)
        fact_results.append({
            "fact": fact, "verdict": best["verdict"], "score": best["score"],
            "weight": weight, "best_conclusion_id": best["conclusion_id"],
            "linked": best["linked"], "overclaims": best["overclaims"],
        })
    return {
        "conversation_id": convo_id,
        "facts": fact_results,
        **_aggregate(fact_results),
        "overclaimed_conclusions": overclaimed,
    }


def _aggregate(fact_results: list[dict]) -> dict:
    n = len(fact_results)
    covered = sum(1 for f in fact_results if f["verdict"] == "covered")
    partial = sum(1 for f in fact_results if f["verdict"] == "partial")
    missing = sum(1 for f in fact_results if f["verdict"] == "missing")
    total_w = sum(f["weight"] for f in fact_results) or 1.0
    weighted = sum(
        f["weight"] * (1.0 if f["verdict"] == "covered"
                       else 0.5 if f["verdict"] == "partial" else 0.0)
        for f in fact_results) / total_w
    linked_covered = [f for f in fact_results
                      if f["verdict"] == "covered" and f["linked"]]
    return {
        "n_facts": n,
        "covered": covered,
        "partial": partial,
        "missing": missing,
        "recall": covered / n if n else 0.0,
        "partial_recall": (covered + 0.5 * partial) / n if n else 0.0,
        "weighted_recall": weighted,
        "linked_share": len(linked_covered) / covered if covered else 0.0,
        "missing_facts": [f["fact"] for f in fact_results
                          if f["verdict"] == "missing"],
    }


def score_alias(per_convo: list[dict]) -> dict:
    """Roll up per-conversation results into one per-alias report dict.

    The returned dict is JSON-serializable and drops straight into
    ``bench.reports.write_results`` under the ``recall`` key.
    """
    all_facts = [f for c in per_convo for f in c["facts"]]
    rollup = _aggregate(all_facts)
    return {
        "recall": rollup["recall"],
        "partial_recall": rollup["partial_recall"],
        "weighted_recall": rollup["weighted_recall"],
        "linked_share": rollup["linked_share"],
        "n_facts": rollup["n_facts"],
        "covered": rollup["covered"],
        "partial": rollup["partial"],
        "missing": rollup["missing"],
        "missing_facts": rollup["missing_facts"],
        "per_conversation": [
            {k: c[k] for k in (
                "conversation_id", "n_facts", "covered", "partial",
                "missing", "recall", "partial_recall", "linked_share")}
            for c in per_convo
        ],
        "overclaimed_conclusions": [
            o for c in per_convo for o in c["overclaimed_conclusions"]
        ],
    }


def render_md(score: dict) -> str:
    """Markdown fragment for the per-alias report."""
    lines = [
        "## Recall (explicit facts)",
        "",
        f"- **recall:** {score['recall']:.3f} "
        f"({score['covered']}/{score['n_facts']} covered)",
        f"- **partial recall:** {score['partial_recall']:.3f} "
        f"({score['partial']} partial, {score['missing']} missing)",
        f"- **weighted recall:** {score['weighted_recall']:.3f}",
        f"- **linked share (covered facts with source linkage):** "
        f"{score['linked_share']:.3f}",
        "",
        "| conversation | facts | covered | partial | missing | recall |",
        "|---|---|---|---|---|---|",
    ]
    for c in score["per_conversation"]:
        lines.append(
            f"| {c['conversation_id']} | {c['n_facts']} | {c['covered']} | "
            f"{c['partial']} | {c['missing']} | {c['recall']:.3f} |")
    if score["missing_facts"]:
        lines += ["", "### Missing facts (first 10)", ""]
        lines += [f"- {f}" for f in score["missing_facts"][:10]]
    if score["overclaimed_conclusions"]:
        lines += ["", "### Overclaimed conclusions",
                  f"({len(score['overclaimed_conclusions'])} flagged)", ""]
        lines += [f"- {o['content']}" for o
                  in score["overclaimed_conclusions"][:10]]
    return "\n".join(lines) + "\n"
