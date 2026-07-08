# prompt-pruning-layer

A deterministic, zero-dependency prompt-pruning layer for long-running LLM conversations — expires stale tool state, collapses duplicate context, and proves it never drops a fact a later turn still depends on.

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)

Most agent frameworks are good at appending to conversation state. Almost none of them are good at removing from it. Every tool call, every retrieved chunk, every turn gets bolted onto the prompt and stays there forever, whether or not it's still needed. This library is the missing other half: a small, provable pipeline that decides what stops earning its place in the prompt.

Read the full write-up on Towards Data Science → [Long Context Isn't Free — I Built a Safe Prompt-Pruning Layer for Production LLMs](https://towardsdatascience.com/author/emmimalp-alexander/)

## What It Does

```
Conversation State → Prompt Builder → Prompt Pruner → Optimized Prompt → LLM
                                            │
                          ┌─────────────────┼─────────────────┐
                          ▼                 ▼                 ▼
                   Pass 1: Expired   Pass 2: Duplicate   Pass 3: Dependency
                   Context           Context             Restoration
                   Elimination       Elimination
```

Three deterministic passes, one `prune()` call:

| Pass | Job |
|---|---|
| Expired Context Elimination | Drops tool outputs superseded by a later call under the same key — keeps only the newest snapshot per resource |
| Duplicate Context Elimination | Collapses near-identical retrieved passages down to a single canonical copy |
| Dependency Restoration | Restores anything Pass 1 or 2 removed that a later, surviving message still depends on |

No embeddings, no LLM calls, no similarity thresholds anywhere in the pruning step itself. Same input always produces the same output, and `prune(prune(x)) == prune(x)` holds by construction — pruning an already-pruned prompt is a no-op, verified across every workload and conversation size in the test suite.

## Installation

```
git clone https://github.com/Emmimal/prompt-pruning-layer.git
cd prompt-pruning-layer
pip install -e .
```

No third-party dependencies for the core library. Nothing to configure, no API key, no model to download. `pip install matplotlib` is only needed if you want to regenerate the benchmark charts — that's article tooling, not part of the pruning pipeline.

## Quick Start

```python
from prompt_pruning import Message, PromptPruner, PromptBuilder, ROLE_USER, ROLE_TOOL_OUTPUT

messages = [
    Message(id="t1", role=ROLE_TOOL_OUTPUT, content="account lookup: tier=free",
            turn=1, tool_call_key="get_account"),
    Message(id="t2", role=ROLE_TOOL_OUTPUT, content="account lookup: tier=pro",
            turn=9, tool_call_key="get_account"),  # supersedes t1
    Message(id="u1", role=ROLE_USER, content="What's my current plan?", turn=10),
]

pruner = PromptPruner()
pruned, report = pruner.prune(messages)

print(report.expired_removed)   # 1  (t1 was superseded by t2)
print(len(pruned))               # 2  (t2 and u1 survive)

prompt = PromptBuilder().build(pruned)
```

Wire it into a turn loop by calling `prune()` on the full conversation state right before you serialize the final prompt:

```python
def handle_turn(conversation_state, new_user_message):
    conversation_state.append(new_user_message)
    pruned_messages, report = pruner.prune(conversation_state)
    prompt = PromptBuilder().build(pruned_messages)
    response = call_llm(prompt)
    conversation_state.append(make_assistant_message(response))
    return response
```

Because pruning is idempotent, calling it on every turn of a growing conversation is safe by default — there's no state to track between calls.

## Dependency Tagging

Pass 3 only restores what it's explicitly told to track. A message can mark itself as defining a fact with `DEFINE:<key>` in its content; a later message that still needs that fact references it with `REF:<key>`. If the defining message gets dropped by Pass 1 or 2 but something surviving still refs its key, Dependency Restoration puts it back.

```python
Message(id="d1", role=ROLE_USER, content="My preferred output format is CSV. DEFINE:fmt1", turn=3)
...
Message(id="u9", role=ROLE_USER, content="Export the results. REF:fmt1", turn=47)
```

This is a literal identifier match, not semantic understanding — it will not catch a paraphrased reference with no matching tag. See Known Limitations below. In a real deployment, these tags are better attached as structured metadata (tool call arguments, session variables) than as literal text markers; Pass 3's logic doesn't change either way.

## Running the Benchmark and Tests

```
python -m unittest discover -s tests -v      # 35 tests
python benchmarks/benchmark.py                # full benchmark, all 15 configurations
python charts/generate_charts.py               # regenerate figures (requires matplotlib)
```

The benchmark generates synthetic conversations from three explicit workload models — plain chat, a RAG assistant, and a multi-tool agent — with parameters fixed before any run, not tuned afterward to hit a target number. Every synthetic corpus ships with ground-truth labels for which messages a later turn depends on, so "did pruning keep everything it needed to" is a deterministic check against known labels, not an estimate.

## Project Structure

```
prompt-pruning-layer/
├── pyproject.toml
├── README.md
├── src/
│   └── prompt_pruning/
│       ├── __init__.py          # public API
│       ├── message_model.py     # Message, REF:/DEFINE: parsing, token approximation
│       ├── corpus_generator.py  # workload-driven synthetic corpus + ground truth
│       └── prompt_pruner.py     # 3-pass PromptPruner + PromptBuilder
├── benchmarks/
│   └── benchmark.py             # deterministic benchmark, 3 workloads x 5 sizes
├── charts/
│   └── generate_charts.py       # article figures (requires matplotlib)
└── tests/
    └── test_pruner.py           # 35 stdlib unittest tests
```

## Performance

Measured on Python 3.12, reproduced independently across two machines (Linux and Windows) with identical token and message counts on both — only wall-clock overhead varies with hardware.

| Workload | Token reduction (50–2,000 turns) | Overhead at 2,000 turns |
|---|---|---|
| Normal chat | 1.9% – 4.1% | 8.3 ms |
| RAG assistant | 27.0% – 32.5% | 29.0 ms |
| Tool agent | 33.1% – 33.7% | 43.0 ms |

Reduction is workload-dependent by design — the pruner removes whatever waste is actually present rather than a flat percentage. Plain chat has little duplicate or expired content to catch; a tool-heavy agent with frequent repeated calls and retrieval overlap has substantially more.

Across all 15 (workload × size) configurations tested: **15/15 preserved 100% of required facts, and 15/15 reached the idempotent fixed point** (`prune(prune(x)) == prune(x)`). Full per-configuration output is in the article linked above.

## When to Use This

Worth it when you have:
- Long-running conversations where tool outputs or retrieved context accumulate across turns
- Agent loops that repeat the same or similar tool calls / retrievals as they work through a task
- A need to reduce prompt size without risking a silently broken dependency chain

Skip it when you have:
- Short, single-turn interactions with nothing to accumulate
- A system where a semantic or embedding-based compressor is already handling this at a layer you don't control
- A need to catch paraphrased references, not just literal ones — this pipeline won't help there (see below)

## Known Limitations

- **Dependency detection is literal, not semantic.** Pass 3 matches exact `REF:`/`DEFINE:` identifiers. A user asking "what format did I mention earlier?" with no matching tag will not be caught.
- **Workload parameters are illustrative, not measured from production telemetry.** They were fixed based on plausible production characteristics before any benchmark was run. Regenerating them from real usage logs is the natural next step if you have that data.
- **Token counts are an approximation.** `approx_token_count()` is a whitespace/punctuation-boundary heuristic, not a real BPE tokenizer (e.g. tiktoken). Applied consistently before and after pruning, so relative reduction percentages are meaningful even though absolute counts won't match a real tokenizer exactly.
- **No semantic compression, embeddings, or LLM-scored pruning.** Deliberately out of scope — mixing a learned component into this deterministic pipeline would trade away the fixed-point and dependency-safety guarantees it's built to prove. A hybrid design (these three passes first, then an optional learned pass on what survives) is a natural extension, not something this library does today.
- **No direct LLM latency measurement.** This library measures token count and pruning overhead, both of which it fully controls. End-to-end latency depends on your provider, model, and serving setup — validate that separately against your own API calls before citing a latency number.

## License

MIT — see [LICENSE](LICENSE).
