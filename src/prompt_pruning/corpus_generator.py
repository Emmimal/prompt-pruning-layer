"""
corpus_generator.py

Generates synthetic conversation states from an explicit WORKLOAD MODEL,
rather than from hand-picked counts of duplicates/expired items. This
matters: if duplication and tool-call repetition are just constants we
pick, a reviewer can reasonably ask whether the benchmark was tuned to
produce a target number. Instead, duplication and expiry are EMERGENT
properties of workload parameters that are decided up front, before any
benchmark is run, based on plausible production characteristics:

  - retrieval_per_turn:     how many documents get retrieved each turn
                            (0 for a plain chat agent, >0 for RAG/tool
                            agents)
  - retrieval_overlap_rate: probability that a retrieved document is a
                            near-duplicate of one already retrieved in a
                            recent window (users revisit topics, RAG
                            re-surfaces the same passages)
  - tool_call_rate:         probability a given turn invokes a tool
  - tool_repeat_rate:       probability a tool call reuses a previously
                            used tool_call_key rather than a new one
                            (repeated planning / re-querying the same
                            resource -> creates expired snapshots)
  - dependency_rate:        probability a turn defines a REF-able key
                            (a stated user preference/setting)
  - dependency_reference_rate: probability a later turn references an
                            open dependency

Three workloads are defined below with parameters chosen BEFORE running
any benchmark:

  WORKLOAD_CHAT       -- plain conversational agent, no retrieval
  WORKLOAD_RAG        -- retrieval-augmented assistant, every turn
                         retrieves documents with realistic overlap
  WORKLOAD_TOOL_AGENT -- multi-tool agent (search, SQL, calculator,
                         filesystem, web) with frequent, often-repeated
                         tool invocations and retrieval

Ground truth for benchmarking (required_ids / expired_ids / duplicate_ids)
is recorded exactly as before, so "required facts preserved" remains a
deterministic check, not a subjective score.

Zero external dependencies (stdlib only).
"""

import random
from dataclasses import dataclass
from .message_model import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL_OUTPUT,
    ROLE_RETRIEVED_DOC,
)

FILLER_SENTENCES = [
    "The quarterly report covers regional sales performance.",
    "Please summarize the attached document in three bullet points.",
    "The API returned a 200 status code with an empty payload.",
    "Our team discussed the migration plan during the sync.",
    "The dataset contains transactions from the last twelve months.",
    "Customer feedback trended positive after the last release.",
    "The build pipeline finished in under four minutes.",
    "Recent logs show a spike in retry attempts around noon.",
    "The design doc was updated to reflect the new schema.",
    "We are still waiting on confirmation from the vendor.",
]


@dataclass
class WorkloadConfig:
    name: str
    retrieval_per_turn: int
    retrieval_overlap_rate: float
    tool_call_rate: float
    tool_repeat_rate: float
    tool_keys: list
    dependency_rate: float
    dependency_reference_rate: float
    overlap_window: int = 8  # how many recent unique docs form the "revisit" pool


# Parameters below are fixed BEFORE running any benchmark. They are not
# adjusted afterward based on which result "looks better" -- see the
# article's methodology section for the reasoning.

WORKLOAD_CHAT = WorkloadConfig(
    name="Normal chat",
    retrieval_per_turn=0,
    retrieval_overlap_rate=0.0,
    tool_call_rate=0.05,
    tool_repeat_rate=0.3,
    tool_keys=["lookup_faq"],
    dependency_rate=0.08,
    dependency_reference_rate=0.3,
)

WORKLOAD_RAG = WorkloadConfig(
    name="RAG assistant",
    retrieval_per_turn=3,
    retrieval_overlap_rate=0.45,
    tool_call_rate=0.10,
    tool_repeat_rate=0.4,
    tool_keys=["search_docs", "fetch_source"],
    dependency_rate=0.08,
    dependency_reference_rate=0.3,
)

WORKLOAD_TOOL_AGENT = WorkloadConfig(
    name="Tool agent",
    retrieval_per_turn=2,
    retrieval_overlap_rate=0.35,
    tool_call_rate=0.65,
    tool_repeat_rate=0.7,
    tool_keys=["search", "sql_query", "calculator", "filesystem", "web_fetch"],
    dependency_rate=0.08,
    dependency_reference_rate=0.3,
)

ALL_WORKLOADS = [WORKLOAD_CHAT, WORKLOAD_RAG, WORKLOAD_TOOL_AGENT]


@dataclass
class GeneratedCorpus:
    messages: list
    required_ids: set
    expired_ids: set
    duplicate_ids: set
    total_turns: int
    workload_name: str


def _filler(rng: random.Random) -> str:
    return rng.choice(FILLER_SENTENCES)


def generate_corpus(
    num_turns: int,
    workload: WorkloadConfig = WORKLOAD_CHAT,
    seed: int = 42,
) -> GeneratedCorpus:
    rng = random.Random(seed)
    messages = []
    required_ids = set()
    expired_ids = set()
    duplicate_ids = set()

    msg_counter = 0

    def next_id(prefix):
        nonlocal msg_counter
        msg_counter += 1
        return f"{prefix}{msg_counter}"

    sys_msg = Message(
        id=next_id("sys"),
        role=ROLE_SYSTEM,
        content="You are a helpful assistant for EmiTechLogic support.",
        turn=0,
    )
    messages.append(sys_msg)
    required_ids.add(sys_msg.id)

    recent_doc_pool = []  # rolling pool of (content, msg_id) for "revisit" overlap
    doc_counter = 0

    open_dependencies = {}  # key -> defining message id
    dependency_counter = 0

    for turn in range(1, num_turns + 1):
        user_msg = Message(
            id=next_id("user"),
            role=ROLE_USER,
            content=_filler(rng),
            turn=turn,
        )
        messages.append(user_msg)

        if rng.random() < workload.dependency_rate:
            dependency_counter += 1
            key = f"pref{dependency_counter}"
            define_msg = Message(
                id=next_id("define"),
                role=ROLE_USER,
                content=f"My preferred output format is CSV. DEFINE:{key}",
                turn=turn,
            )
            messages.append(define_msg)
            required_ids.add(define_msg.id)
            open_dependencies[key] = define_msg.id

        asst_msg = Message(
            id=next_id("asst"),
            role=ROLE_ASSISTANT,
            content=_filler(rng),
            turn=turn,
        )
        messages.append(asst_msg)

        if open_dependencies and rng.random() < workload.dependency_reference_rate:
            key = rng.choice(list(open_dependencies.keys()))
            ref_msg = Message(
                id=next_id("user"),
                role=ROLE_USER,
                content=f"Export the results. REF:{key}",
                turn=turn,
            )
            messages.append(ref_msg)
            required_ids.add(ref_msg.id)
            required_ids.add(open_dependencies[key])

        for _ in range(workload.retrieval_per_turn):
            is_overlap = recent_doc_pool and rng.random() < workload.retrieval_overlap_rate
            if is_overlap:
                content, _orig_id = rng.choice(recent_doc_pool)
            else:
                doc_counter += 1
                content = f"Retrieved passage #{doc_counter}: {_filler(rng)}"

            doc_msg = Message(
                id=next_id("doc"),
                role=ROLE_RETRIEVED_DOC,
                content=content,
                turn=turn,
            )
            messages.append(doc_msg)

            if not is_overlap:
                recent_doc_pool.append((content, doc_msg.id))
                if len(recent_doc_pool) > workload.overlap_window:
                    recent_doc_pool.pop(0)

        if workload.tool_keys and rng.random() < workload.tool_call_rate:
            used_keys = [m.tool_call_key for m in messages if m.tool_call_key]
            reuse = used_keys and rng.random() < workload.tool_repeat_rate
            if reuse:
                key = rng.choice(used_keys)
            else:
                key = rng.choice(workload.tool_keys)

            content = f"[{key}] result snapshot at turn {turn}: {_filler(rng)}"

            # Some tool calls (e.g. a "get_user_settings" style lookup)
            # surface a fact that a LATER turn depends on, even though
            # this exact tool message may later be superseded (expired)
            # by a newer call with the same tool_call_key. This is the
            # scenario Pass 3 exists for: an expired-looking message
            # that must be restored because something still depends on
            # it. Without this, expiry/duplicate removal never actually
            # touches a message a later turn references, and Pass 3
            # would never be exercised by the benchmark.
            defines_dependency = rng.random() < 0.15
            dep_key = None
            if defines_dependency:
                dependency_counter += 1
                dep_key = f"toolpref{dependency_counter}"
                content += f" DEFINE:{dep_key}"

            tool_msg = Message(
                id=next_id("tool"),
                role=ROLE_TOOL_OUTPUT,
                content=content,
                turn=turn,
                tool_call_key=key,
            )
            messages.append(tool_msg)

            if dep_key:
                open_dependencies[dep_key] = tool_msg.id

    last_occurrence = {}
    for m in messages:
        if m.role == ROLE_TOOL_OUTPUT and m.tool_call_key:
            last_occurrence[m.tool_call_key] = m.id
    for m in messages:
        if m.role == ROLE_TOOL_OUTPUT and m.tool_call_key:
            if last_occurrence[m.tool_call_key] != m.id:
                expired_ids.add(m.id)
            else:
                required_ids.add(m.id)

    seen_content = {}
    for m in messages:
        if m.role == ROLE_RETRIEVED_DOC:
            norm = " ".join(m.content.lower().split())
            if norm in seen_content:
                duplicate_ids.add(m.id)
            else:
                seen_content[norm] = m.id
                required_ids.add(m.id)

    return GeneratedCorpus(
        messages=messages,
        required_ids=required_ids,
        expired_ids=expired_ids,
        duplicate_ids=duplicate_ids,
        total_turns=num_turns,
        workload_name=workload.name,
    )
