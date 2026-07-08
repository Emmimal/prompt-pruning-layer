"""
prompt_pruner.py

The Prompt-Pruning Layer: three deterministic passes over a list of
Message objects, run before the final prompt is assembled and sent to
the model. Named in the style of compiler optimization passes, since
each is a deterministic, order-independent-within-itself transformation
with a provable fixed point (see test_idempotent_* in test_pruner.py
and the idempotence check in benchmark.py):

    Pass 1 -- Expired Context Elimination
        Removes tool outputs / retrieved content that has been
        superseded by a later call with the same tool_call_key.

    Pass 2 -- Duplicate Context Elimination
        Collapses near-identical passages (same content after
        whitespace/casing normalization) down to a single canonical
        copy, keeping the first occurrence.

    Pass 3 -- Dependency Restoration
        Restores any message that a later, still-surviving message
        depends on via the REF:<key> / DEFINE:<key> convention, even
        if an earlier pass would otherwise have dropped it.

Honest limitation, stated up front (and repeated in the article): the
dependency detector keys on literal REF:/DEFINE: identifier matching.
It will not catch a paraphrased reference ("what format did I ask for
earlier?") that doesn't use the identifier convention. This is a
deliberate, disclosed simplification -- see message_model.py docstring.

Zero external dependencies (stdlib only).
"""

from dataclasses import dataclass, field
from .message_model import Message, ROLE_TOOL_OUTPUT, ROLE_RETRIEVED_DOC


@dataclass
class PruneReport:
    """Diagnostics describing what each pass did, for the benchmark."""
    input_count: int
    expired_removed: int
    duplicates_removed: int
    restored_for_dependency: int
    output_count: int
    removed_ids: list = field(default_factory=list)


class PromptPruner:
    def __init__(self):
        pass

    def _pass1_expired_context_elimination(self, messages):
        """Pass 1: Expired Context Elimination. Keep only the LATEST message for each tool_call_key."""
        last_occurrence = {}
        for m in messages:
            if m.tool_call_key:
                last_occurrence[m.tool_call_key] = m.id

        kept = []
        removed = []
        for m in messages:
            if m.tool_call_key and last_occurrence[m.tool_call_key] != m.id:
                removed.append(m)
            else:
                kept.append(m)
        return kept, removed

    def _pass2_duplicate_context_elimination(self, messages):
        """Pass 2: Duplicate Context Elimination. Collapse near-identical retrieved/tool content to first occurrence."""
        seen = {}
        kept = []
        removed = []
        for m in messages:
            if m.role in (ROLE_RETRIEVED_DOC,):
                norm = " ".join(m.content.lower().split())
                if norm in seen:
                    removed.append(m)
                    continue
                seen[norm] = m.id
            kept.append(m)
        return kept, removed

    def _pass3_dependency_restoration(self, all_messages, kept_messages, removed_messages):
        """
        Pass 3: Dependency Restoration. Restore any removed message
        whose id is a DEFINE-target for a key referenced by a message
        that is still present in kept_messages.
        """
        kept_ids = {m.id for m in kept_messages}
        by_id = {m.id: m for m in all_messages}

        # Build key -> defining message id map across the FULL original
        # message set (a message could define a key even if it was
        # dropped by an earlier pass).
        key_definer = {}
        for m in all_messages:
            for key in m.defines_keys:
                key_definer[key] = m.id

        # Find keys referenced by any message still present in kept set.
        referenced_keys = set()
        for m in kept_messages:
            referenced_keys.update(m.references())

        restored = []
        for key in referenced_keys:
            definer_id = key_definer.get(key)
            if definer_id and definer_id not in kept_ids:
                restored_msg = by_id[definer_id]
                kept_messages.append(restored_msg)
                kept_ids.add(definer_id)
                restored.append(restored_msg)

        # Keep chronological order after restoration.
        kept_messages.sort(key=lambda m: (m.turn, m.id))
        return kept_messages, restored

    def prune(self, messages):
        """
        Run all three passes in sequence and return (pruned_messages, report).
        """
        input_count = len(messages)

        after_p1, removed_p1 = self._pass1_expired_context_elimination(messages)
        after_p2, removed_p2 = self._pass2_duplicate_context_elimination(after_p1)

        all_removed_so_far = removed_p1 + removed_p2
        after_p3, restored = self._pass3_dependency_restoration(
            messages, after_p2, all_removed_so_far
        )

        removed_ids = [
            m.id for m in all_removed_so_far if m.id not in {r.id for r in restored}
        ]

        report = PruneReport(
            input_count=input_count,
            expired_removed=len(removed_p1),
            duplicates_removed=len(removed_p2),
            restored_for_dependency=len(restored),
            output_count=len(after_p3),
            removed_ids=removed_ids,
        )
        return after_p3, report


class PromptBuilder:
    """
    Assembles a final prompt string from a (pruned) list of Messages,
    in chronological order, with a role-labeled format.
    """

    def build(self, messages) -> str:
        ordered = sorted(messages, key=lambda m: (m.turn, m.id))
        lines = []
        for m in ordered:
            lines.append(f"[{m.role.upper()}] {m.content}")
        return "\n".join(lines)
