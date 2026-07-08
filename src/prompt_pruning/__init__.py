"""
prompt_pruning

A deterministic, zero-dependency prompt-pruning layer for long-running
LLM conversation state.

Public API:
    Message, ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL_OUTPUT,
    ROLE_RETRIEVED_DOC

    generate_corpus, WorkloadConfig,
    WORKLOAD_CHAT, WORKLOAD_RAG, WORKLOAD_TOOL_AGENT, ALL_WORKLOADS

    PromptPruner, PromptBuilder, PruneReport
"""

from .message_model import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL_OUTPUT,
    ROLE_RETRIEVED_DOC,
)
from .corpus_generator import (
    generate_corpus,
    GeneratedCorpus,
    WorkloadConfig,
    WORKLOAD_CHAT,
    WORKLOAD_RAG,
    WORKLOAD_TOOL_AGENT,
    ALL_WORKLOADS,
)
from .prompt_pruner import PromptPruner, PromptBuilder, PruneReport

__all__ = [
    "Message",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "ROLE_ASSISTANT",
    "ROLE_TOOL_OUTPUT",
    "ROLE_RETRIEVED_DOC",
    "generate_corpus",
    "GeneratedCorpus",
    "WorkloadConfig",
    "WORKLOAD_CHAT",
    "WORKLOAD_RAG",
    "WORKLOAD_TOOL_AGENT",
    "ALL_WORKLOADS",
    "PromptPruner",
    "PromptBuilder",
    "PruneReport",
]

