"""
test_pruner.py

Stdlib-only unit tests for the Prompt-Pruning Layer:
  - message_model.py  (Message, token approximation, REF/DEFINE parsing)
  - corpus_generator.py (workload-driven synthetic corpus, ground truth)
  - prompt_pruner.py  (three-pass pruner + PromptBuilder)

Run with:
    python -m unittest test_pruner -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_pruning.message_model import Message, ROLE_SYSTEM, ROLE_USER, ROLE_TOOL_OUTPUT, ROLE_RETRIEVED_DOC
from prompt_pruning.corpus_generator import (
    generate_corpus,
    WORKLOAD_CHAT,
    WORKLOAD_RAG,
    WORKLOAD_TOOL_AGENT,
    WorkloadConfig,
)
from prompt_pruning.prompt_pruner import PromptPruner, PromptBuilder


# ---------------------------------------------------------------------------
# Message model tests
# ---------------------------------------------------------------------------
class TestMessageModel(unittest.TestCase):

    def test_valid_role_accepted(self):
        m = Message(id="1", role=ROLE_USER, content="hello", turn=1)
        self.assertEqual(m.role, ROLE_USER)

    def test_invalid_role_raises(self):
        with self.assertRaises(ValueError):
            Message(id="1", role="bogus_role", content="hello", turn=1)

    def test_empty_content_token_count_is_zero(self):
        m = Message(id="1", role=ROLE_USER, content="", turn=1)
        self.assertEqual(m.approx_token_count(), 0)

    def test_token_count_scales_with_length(self):
        short = Message(id="1", role=ROLE_USER, content="hello world", turn=1)
        long = Message(id="2", role=ROLE_USER, content="hello world " * 20, turn=1)
        self.assertGreater(long.approx_token_count(), short.approx_token_count())

    def test_punctuation_increases_token_count(self):
        plain = Message(id="1", role=ROLE_USER, content="hello world", turn=1)
        punct = Message(id="2", role=ROLE_USER, content="hello, world!", turn=1)
        self.assertGreater(punct.approx_token_count(), plain.approx_token_count())

    def test_references_extracts_ref_keys(self):
        m = Message(id="1", role=ROLE_USER, content="Export this. REF:pref1 REF:pref2", turn=1)
        self.assertEqual(set(m.references()), {"pref1", "pref2"})

    def test_no_references_returns_empty_list(self):
        m = Message(id="1", role=ROLE_USER, content="just a normal message", turn=1)
        self.assertEqual(m.references(), [])

    def test_defines_keys_autodetected(self):
        m = Message(id="1", role=ROLE_USER, content="My format is CSV. DEFINE:pref1", turn=1)
        self.assertEqual(m.defines_keys, ["pref1"])

    def test_defines_keys_explicit_override_respected(self):
        m = Message(
            id="1", role=ROLE_USER, content="no define marker here", turn=1,
            defines_keys=["manual_key"],
        )
        self.assertEqual(m.defines_keys, ["manual_key"])


# ---------------------------------------------------------------------------
# Corpus generator tests
# ---------------------------------------------------------------------------
class TestCorpusGenerator(unittest.TestCase):

    def test_generation_is_deterministic_for_fixed_seed(self):
        c1 = generate_corpus(num_turns=50, workload=WORKLOAD_RAG, seed=7)
        c2 = generate_corpus(num_turns=50, workload=WORKLOAD_RAG, seed=7)
        ids1 = [m.id for m in c1.messages]
        ids2 = [m.id for m in c2.messages]
        self.assertEqual(ids1, ids2)
        contents1 = [m.content for m in c1.messages]
        contents2 = [m.content for m in c2.messages]
        self.assertEqual(contents1, contents2)

    def test_different_seed_can_change_output(self):
        c1 = generate_corpus(num_turns=50, workload=WORKLOAD_RAG, seed=1)
        c2 = generate_corpus(num_turns=50, workload=WORKLOAD_RAG, seed=2)
        self.assertNotEqual(
            [m.content for m in c1.messages],
            [m.content for m in c2.messages],
        )

    def test_chat_workload_has_no_retrieved_docs(self):
        corpus = generate_corpus(num_turns=100, workload=WORKLOAD_CHAT, seed=3)
        docs = [m for m in corpus.messages if m.role == ROLE_RETRIEVED_DOC]
        self.assertEqual(len(docs), 0)

    def test_rag_workload_has_retrieved_docs(self):
        corpus = generate_corpus(num_turns=50, workload=WORKLOAD_RAG, seed=3)
        docs = [m for m in corpus.messages if m.role == ROLE_RETRIEVED_DOC]
        self.assertGreater(len(docs), 0)

    def test_tool_agent_workload_has_more_tool_calls_than_chat(self):
        chat = generate_corpus(num_turns=200, workload=WORKLOAD_CHAT, seed=5)
        agent = generate_corpus(num_turns=200, workload=WORKLOAD_TOOL_AGENT, seed=5)
        chat_tools = [m for m in chat.messages if m.role == ROLE_TOOL_OUTPUT]
        agent_tools = [m for m in agent.messages if m.role == ROLE_TOOL_OUTPUT]
        self.assertGreater(len(agent_tools), len(chat_tools))

    def test_system_message_always_required(self):
        corpus = generate_corpus(num_turns=20, workload=WORKLOAD_CHAT, seed=1)
        sys_msgs = [m for m in corpus.messages if m.role == ROLE_SYSTEM]
        self.assertEqual(len(sys_msgs), 1)
        self.assertIn(sys_msgs[0].id, corpus.required_ids)

    def test_duplicate_ids_have_matching_normalized_content(self):
        corpus = generate_corpus(num_turns=100, workload=WORKLOAD_RAG, seed=9)
        by_id = {m.id: m for m in corpus.messages}
        norm_to_ids = {}
        for m in corpus.messages:
            if m.role == ROLE_RETRIEVED_DOC:
                norm = " ".join(m.content.lower().split())
                norm_to_ids.setdefault(norm, []).append(m.id)
        for dup_id in corpus.duplicate_ids:
            norm = " ".join(by_id[dup_id].content.lower().split())
            self.assertGreater(len(norm_to_ids[norm]), 1)

    def test_expired_and_required_tool_ids_are_disjoint_per_key(self):
        # For a given tool_call_key, the required (latest) id must not
        # also appear in expired_ids.
        corpus = generate_corpus(num_turns=300, workload=WORKLOAD_TOOL_AGENT, seed=11)
        by_id = {m.id: m for m in corpus.messages}
        last_occurrence = {}
        for m in corpus.messages:
            if m.role == ROLE_TOOL_OUTPUT and m.tool_call_key:
                last_occurrence[m.tool_call_key] = m.id
        for key, latest_id in last_occurrence.items():
            self.assertNotIn(latest_id, corpus.expired_ids)

    def test_zero_turns_still_produces_system_message(self):
        corpus = generate_corpus(num_turns=0, workload=WORKLOAD_CHAT, seed=1)
        self.assertEqual(len(corpus.messages), 1)
        self.assertEqual(corpus.messages[0].role, ROLE_SYSTEM)


# ---------------------------------------------------------------------------
# Prompt pruner tests
# ---------------------------------------------------------------------------
class TestPromptPruner(unittest.TestCase):

    def setUp(self):
        self.pruner = PromptPruner()

    def test_empty_message_list_prunes_to_empty(self):
        pruned, report = self.pruner.prune([])
        self.assertEqual(pruned, [])
        self.assertEqual(report.input_count, 0)
        self.assertEqual(report.output_count, 0)

    def test_expired_tool_output_is_removed(self):
        m1 = Message(id="t1", role=ROLE_TOOL_OUTPUT, content="old result", turn=1, tool_call_key="search")
        m2 = Message(id="t2", role=ROLE_TOOL_OUTPUT, content="new result", turn=2, tool_call_key="search")
        pruned, report = self.pruner.prune([m1, m2])
        pruned_ids = {m.id for m in pruned}
        self.assertNotIn("t1", pruned_ids)
        self.assertIn("t2", pruned_ids)
        self.assertEqual(report.expired_removed, 1)

    def test_duplicate_retrieved_doc_is_collapsed(self):
        m1 = Message(id="d1", role=ROLE_RETRIEVED_DOC, content="Same passage here.", turn=1)
        m2 = Message(id="d2", role=ROLE_RETRIEVED_DOC, content="same PASSAGE   here.", turn=2)
        pruned, report = self.pruner.prune([m1, m2])
        pruned_ids = {m.id for m in pruned}
        self.assertIn("d1", pruned_ids)
        self.assertNotIn("d2", pruned_ids)
        self.assertEqual(report.duplicates_removed, 1)

    def test_non_duplicate_docs_both_survive(self):
        m1 = Message(id="d1", role=ROLE_RETRIEVED_DOC, content="First passage.", turn=1)
        m2 = Message(id="d2", role=ROLE_RETRIEVED_DOC, content="Totally different passage.", turn=2)
        pruned, report = self.pruner.prune([m1, m2])
        pruned_ids = {m.id for m in pruned}
        self.assertIn("d1", pruned_ids)
        self.assertIn("d2", pruned_ids)
        self.assertEqual(report.duplicates_removed, 0)

    def test_expired_message_restored_if_still_referenced(self):
        # t1 is superseded by t2 (same tool_call_key) so Pass 1 would
        # normally drop it -- but it also DEFINEs a key that a later
        # user message REFs, so Pass 3 must bring it back.
        t1 = Message(
            id="t1", role=ROLE_TOOL_OUTPUT,
            content="settings lookup DEFINE:fmt1", turn=1, tool_call_key="get_settings",
        )
        t2 = Message(
            id="t2", role=ROLE_TOOL_OUTPUT,
            content="settings lookup again", turn=2, tool_call_key="get_settings",
        )
        ref = Message(id="u1", role=ROLE_USER, content="Export now. REF:fmt1", turn=3)
        pruned, report = self.pruner.prune([t1, t2, ref])
        pruned_ids = {m.id for m in pruned}
        self.assertIn("t1", pruned_ids, "expired-but-referenced message must be restored")
        self.assertIn("t2", pruned_ids)
        self.assertIn("u1", pruned_ids)
        self.assertEqual(report.restored_for_dependency, 1)

    def test_duplicate_message_restored_if_it_defines_a_referenced_key(self):
        # d1 is the canonical kept copy; d2 is a near-duplicate that
        # would normally be dropped by Pass 2 -- but d2 (not d1) is the
        # one that happens to carry the DEFINE marker referenced later.
        d1 = Message(id="d1", role=ROLE_RETRIEVED_DOC, content="Shared passage text.", turn=1)
        d2 = Message(
            id="d2", role=ROLE_RETRIEVED_DOC,
            content="Shared passage text. DEFINE:notek",
            turn=2,
        )
        ref = Message(id="u1", role=ROLE_USER, content="Use that note. REF:notek", turn=3)
        pruned, report = self.pruner.prune([d1, d2, ref])
        pruned_ids = {m.id for m in pruned}
        self.assertIn("d2", pruned_ids, "duplicate carrying a referenced DEFINE must be restored")

    def test_unreferenced_defines_do_not_force_retention(self):
        t1 = Message(id="t1", role=ROLE_TOOL_OUTPUT, content="DEFINE:unused_key", turn=1, tool_call_key="k")
        t2 = Message(id="t2", role=ROLE_TOOL_OUTPUT, content="newer", turn=2, tool_call_key="k")
        pruned, report = self.pruner.prune([t1, t2])
        pruned_ids = {m.id for m in pruned}
        self.assertNotIn("t1", pruned_ids)
        self.assertEqual(report.restored_for_dependency, 0)

    def test_chained_dependency_across_multiple_removed_messages(self):
        # Two separate expired tool outputs, each defining a distinct
        # key, both referenced later -- both must be restored.
        t1 = Message(id="t1", role=ROLE_TOOL_OUTPUT, content="a DEFINE:k1", turn=1, tool_call_key="x")
        t1b = Message(id="t1b", role=ROLE_TOOL_OUTPUT, content="a2", turn=2, tool_call_key="x")
        t2 = Message(id="t2", role=ROLE_TOOL_OUTPUT, content="b DEFINE:k2", turn=1, tool_call_key="y")
        t2b = Message(id="t2b", role=ROLE_TOOL_OUTPUT, content="b2", turn=2, tool_call_key="y")
        ref = Message(id="u1", role=ROLE_USER, content="REF:k1 REF:k2", turn=3)
        pruned, report = self.pruner.prune([t1, t1b, t2, t2b, ref])
        pruned_ids = {m.id for m in pruned}
        self.assertIn("t1", pruned_ids)
        self.assertIn("t2", pruned_ids)
        self.assertEqual(report.restored_for_dependency, 2)

    def test_output_never_exceeds_input_count(self):
        corpus = generate_corpus(num_turns=200, workload=WORKLOAD_TOOL_AGENT, seed=13)
        pruned, report = self.pruner.prune(corpus.messages)
        self.assertLessEqual(len(pruned), len(corpus.messages))

    def test_all_required_ids_survive_across_workloads_and_sizes(self):
        for workload in (WORKLOAD_CHAT, WORKLOAD_RAG, WORKLOAD_TOOL_AGENT):
            for n in (30, 150, 600):
                corpus = generate_corpus(num_turns=n, workload=workload, seed=17)
                pruned, report = self.pruner.prune(corpus.messages)
                surviving_ids = {m.id for m in pruned}
                missing = corpus.required_ids - surviving_ids
                self.assertEqual(
                    missing, set(),
                    f"missing required ids for workload={workload.name}, n={n}: {missing}",
                )

    def test_pruning_never_introduces_new_message_ids(self):
        corpus = generate_corpus(num_turns=150, workload=WORKLOAD_RAG, seed=21)
        input_ids = {m.id for m in corpus.messages}
        pruned, report = self.pruner.prune(corpus.messages)
        pruned_ids = {m.id for m in pruned}
        self.assertTrue(pruned_ids.issubset(input_ids))

    def test_output_is_chronologically_sorted(self):
        corpus = generate_corpus(num_turns=100, workload=WORKLOAD_TOOL_AGENT, seed=23)
        pruned, report = self.pruner.prune(corpus.messages)
        turns = [m.turn for m in pruned]
        self.assertEqual(turns, sorted(turns))

    def test_idempotent_on_hand_built_case(self):
        # prune(prune(x)) must equal prune(x) -- pruning an
        # already-pruned prompt should be a no-op fixed point.
        t1 = Message(id="t1", role=ROLE_TOOL_OUTPUT, content="old", turn=1, tool_call_key="k")
        t2 = Message(id="t2", role=ROLE_TOOL_OUTPUT, content="new", turn=2, tool_call_key="k")
        d1 = Message(id="d1", role=ROLE_RETRIEVED_DOC, content="Same passage.", turn=1)
        d2 = Message(id="d2", role=ROLE_RETRIEVED_DOC, content="same PASSAGE.", turn=2)
        messages = [t1, t2, d1, d2]

        once, _ = self.pruner.prune(messages)
        twice, _ = self.pruner.prune(once)

        once_ids = {m.id for m in once}
        twice_ids = {m.id for m in twice}
        self.assertEqual(once_ids, twice_ids)

    def test_idempotent_across_workloads_and_sizes(self):
        for workload in (WORKLOAD_CHAT, WORKLOAD_RAG, WORKLOAD_TOOL_AGENT):
            for n in (30, 150, 600):
                corpus = generate_corpus(num_turns=n, workload=workload, seed=29)
                once, _ = self.pruner.prune(corpus.messages)
                twice, _ = self.pruner.prune(once)
                once_ids = {m.id for m in once}
                twice_ids = {m.id for m in twice}
                self.assertEqual(
                    once_ids, twice_ids,
                    f"not idempotent for workload={workload.name}, n={n}",
                )


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------
class TestPromptBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = PromptBuilder()

    def test_empty_list_builds_empty_string(self):
        self.assertEqual(self.builder.build([]), "")

    def test_build_includes_role_labels(self):
        m = Message(id="1", role=ROLE_USER, content="hi there", turn=1)
        out = self.builder.build([m])
        self.assertIn("[USER]", out)
        self.assertIn("hi there", out)

    def test_build_orders_by_turn(self):
        m1 = Message(id="1", role=ROLE_USER, content="second", turn=2)
        m2 = Message(id="2", role=ROLE_USER, content="first", turn=1)
        out = self.builder.build([m1, m2])
        self.assertLess(out.index("first"), out.index("second"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
