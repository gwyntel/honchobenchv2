"""Tests for bench.scoring.recall — stdlib unittest only (no pytest here).

The stub judge is a TEST DOUBLE, not the production matcher: it returns
2.0 for a verbatim conclusion, 1.0 for a strict first-half prefix, 0.0
otherwise, and flags overclaim on the UNSUPPORTED marker. The production
matcher is a decision model behind the same judge_fn interface.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.scoring import recall  # noqa: E402

LABELS = Path(__file__).resolve().parents[1] / "data" / "dataset" / "labels.json"


def stub_judge(state, questions):
    """Deterministic test double for the decision-model judge."""
    fact = state["fact"]["text"]
    content = state["conclusion"]["content"]
    if content == fact:
        score = 2.0
    elif fact.startswith(content) and content:
        score = 1.0
    else:
        score = 0.0
    return {
        "covers_fact": {"score": score},
        "overclaims": {"probability": 1.0 if "UNSUPPORTED" in content else 0.0},
    }


def bow_embed(texts):
    """Tiny bag-of-words embedder for the pre-filter test."""
    vocab = sorted({w.lower() for t in texts for w in t.split()})
    idx = {w: i for i, w in enumerate(vocab)}
    vecs = []
    for t in texts:
        v = [0.0] * len(vocab)
        for w in t.split():
            v[idx[w.lower()]] += 1.0
        vecs.append(v)
    return vecs


class TestVerdicts(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(recall.verdict_for_score(2.0), "covered")
        self.assertEqual(recall.verdict_for_score(1.5), "covered")
        self.assertEqual(recall.verdict_for_score(1.49), "partial")
        self.assertEqual(recall.verdict_for_score(1.0), "partial")
        self.assertEqual(recall.verdict_for_score(0.5), "partial")
        self.assertEqual(recall.verdict_for_score(0.49), "missing")
        self.assertEqual(recall.verdict_for_score(0.0), "missing")

    def test_judge_fact_validation(self):
        def bad_judge(state, questions):
            return {"covers_fact": {"score": 3.0},
                    "overclaims": {"probability": 0.0}}
        with self.assertRaises(ValueError):
            recall.judge_fact("f", {"content": "c"}, bad_judge)


class TestNormalize(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(len(recall.normalize_conclusions(
            {"conclusions": [{"content": "a"}, {"content": "b"}]})), 2)
        self.assertEqual(len(recall.normalize_conclusions(
            [{"content": "a"}])), 1)
        self.assertEqual(recall.normalize_conclusions(["plain"])[0]["content"],
                         "plain")
        self.assertEqual(recall.normalize_conclusions(
            [{"content": "a", "source_message_ids": ["m1"]}])[0]
            ["source_message_ids"], ["m1"])


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.facts = [
            "Mark Taylor is EnronOnline counsel.",
            "Mark Taylor requested Revised Gas Option Language from Taffy Milligan.",
            "The index describes contracts and was discussed with Mark Dilworth.",
        ]
        self.conclusions = [
            {"id": "c1", "content": self.facts[0],
             "source_message_ids": ["m1", "m2"]},
            {"id": "c2",
             "content": "Mark Taylor requested Revised Gas Option Language",
             "source_message_ids": ["m3"]},
            {"id": "c3", "content": "UNSUPPORTED: Mark Taylor owns Enron.",
             "source_message_ids": []},
        ]

    def test_per_fact_verdicts(self):
        res = recall.score_conversation(
            "t", self.facts, self.conclusions, stub_judge)
        verdicts = [f["verdict"] for f in res["facts"]]
        self.assertEqual(verdicts, ["covered", "partial", "missing"])
        self.assertEqual(res["covered"], 1)
        self.assertEqual(res["partial"], 1)
        self.assertEqual(res["missing"], 1)
        self.assertAlmostEqual(res["recall"], 1 / 3)
        self.assertAlmostEqual(res["partial_recall"], 0.5)

    def test_linkage_recorded(self):
        res = recall.score_conversation(
            "t", self.facts, self.conclusions, stub_judge)
        self.assertTrue(res["facts"][0]["linked"])
        self.assertTrue(res["facts"][1]["linked"])
        self.assertFalse(res["facts"][2]["linked"])
        self.assertAlmostEqual(res["linked_share"], 1.0)

    def test_overclaim_flagged_once(self):
        res = recall.score_conversation(
            "t", self.facts, self.conclusions, stub_judge)
        flagged = res["overclaimed_conclusions"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["conclusion_id"], "c3")

    def test_importance_weights(self):
        weights = {self.facts[0]: "critical", self.facts[1]: "trivial"}
        res = recall.score_conversation(
            "t", self.facts, self.conclusions, stub_judge, weights=weights)
        # covered critical (3.0) + partial trivial (0.5*0.5) over 3+0.5+1
        self.assertAlmostEqual(res["weighted_recall"], 3.25 / 4.5)
        res_plain = recall.score_conversation(
            "t", self.facts, self.conclusions, stub_judge)
        self.assertAlmostEqual(res_plain["weighted_recall"], 0.5)

    def test_prefilter_ranking(self):
        pairs = recall.prefilter_pairs(
            self.facts, self.conclusions, bow_embed, top_k=1)
        self.assertEqual(pairs[0], [0])  # fact 0 -> conclusion 0
        self.assertEqual(pairs[1], [1])  # fact 1 -> conclusion 1

    def test_prefilter_none_keeps_all(self):
        pairs = recall.prefilter_pairs(self.facts, self.conclusions, None)
        self.assertEqual(pairs[0], [0, 1, 2])


class TestRealLabels(unittest.TestCase):
    """Hand-checked results on all 8 labeled conversations.

    Fixture conclusions per conversation: fact[0] verbatim (linked) ->
    covered, fact[1] first-half prefix (linked) -> partial, everything
    else unmatched -> missing. Expected counts are hand-computed, not
    derived from the scorer.
    """

    def test_all_eight_conversations(self):
        labels = recall.load_labels(LABELS)
        self.assertEqual(len(labels), 8)
        per_convo = []
        for convo_id, lab in labels.items():
            facts = lab["explicit_facts"]
            first_half = " ".join(facts[1].split()[:len(facts[1].split()) // 2])
            conclusions = [
                {"id": f"{convo_id}-c1", "content": facts[0],
                 "source_message_ids": ["m1"]},
                {"id": f"{convo_id}-c2", "content": first_half,
                 "source_message_ids": ["m2"]},
            ]
            res = recall.score_conversation(
                convo_id, facts, conclusions, stub_judge)
            n = len(facts)
            self.assertEqual(res["covered"], 1, convo_id)
            self.assertEqual(res["partial"], 1, convo_id)
            self.assertEqual(res["missing"], n - 2, convo_id)
            self.assertAlmostEqual(res["recall"], 1 / n, places=6)
            self.assertAlmostEqual(res["partial_recall"], 1.5 / n, places=6)
            self.assertAlmostEqual(res["linked_share"], 1.0)
            per_convo.append(res)
        rollup = recall.score_alias(per_convo)
        total = sum(len(lab["explicit_facts"]) for lab in labels.values())
        self.assertEqual(rollup["n_facts"], total)
        self.assertEqual(rollup["covered"], 8)
        self.assertEqual(rollup["partial"], 8)
        self.assertEqual(rollup["missing"], total - 16)
        self.assertAlmostEqual(rollup["recall"], 8 / total, places=6)
        md = recall.render_md(rollup)
        self.assertIn(f"recall:** {8 / total:.3f}", md)
        self.assertIn("enron_taylor_forster_intl", md)


if __name__ == "__main__":
    unittest.main()
