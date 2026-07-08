"""
message_model.py

Core data model for the Prompt-Pruning Layer.

A "Message" represents one unit of prompt state: a system prompt, a user
turn, an assistant turn, a tool output, or a retrieved document chunk.
Everything the Prompt Builder would normally concatenate into one giant
prompt is represented as a list of Message objects, which the Prompt
Pruner then filters before final assembly.

This module has zero external dependencies (stdlib only).
"""

from dataclasses import dataclass, field
from typing import Optional
import re


# Message roles / types that can appear in a conversation state.
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_OUTPUT = "tool_output"
ROLE_RETRIEVED_DOC = "retrieved_doc"

VALID_ROLES = {
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL_OUTPUT,
    ROLE_RETRIEVED_DOC,
}

# Identifiers referenced by later turns are written as REF:<key> inside
# message content. This is a deliberately simple, deterministic
# convention -- see the article's "Honest Limitations" section for why
# this is a heuristic and not a full coreference resolver.
REF_PATTERN = re.compile(r"REF:([A-Za-z0-9_\-]+)")


@dataclass
class Message:
    """A single unit of prompt state."""

    id: str
    role: str
    content: str
    turn: int
    # Optional bookkeeping fields used by the pruning passes below.
    tool_call_key: Optional[str] = None   # groups repeated calls to same tool/query
    expires_after_turn: Optional[int] = None  # turn number after which this is stale
    defines_keys: list = field(default_factory=list)   # REF keys this message defines
    # internal, set by pruner for diagnostics
    _dropped_by: Optional[str] = None

    def __post_init__(self):
        if self.role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {self.role}")
        if not self.defines_keys:
            # auto-detect defined keys via a simple "DEFINE:<key>" convention
            self.defines_keys = re.findall(r"DEFINE:([A-Za-z0-9_\-]+)", self.content)

    def references(self) -> list:
        """Return the REF keys this message's content depends on."""
        return REF_PATTERN.findall(self.content)

    def approx_token_count(self) -> int:
        """
        Approximate token count using a word/punctuation based heuristic.

        This is NOT a real tokenizer. It splits on whitespace and common
        punctuation boundaries as a stand-in for subword tokenization,
        matching the pure-Python, zero-dependency constraint of this
        project. It is consistently applied "before" and "after" pruning,
        so relative reduction percentages are meaningful even though the
        absolute counts will not exactly match a real BPE tokenizer count.
        """
        if not self.content:
            return 0
        # Split into words, then split words further on punctuation,
        # which approximates how subword tokenizers often split at
        # punctuation boundaries.
        words = self.content.split()
        token_count = 0
        for w in words:
            pieces = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", w)
            token_count += max(1, len(pieces))
        return token_count
