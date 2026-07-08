"""
benchmark.py

Deterministic benchmark for the Prompt-Pruning Layer.

This measures ONLY what this system directly controls:

  - input tokens before pruning (approx_token_count sum)
  - input tokens after pruning
  - token reduction percentage
  - pruning pipeline wall-clock overhead (time.perf_counter)
  - prompt build time before/after
  - "required facts preserved": does every message id in the corpus's
    labeled required_ids set survive pruning? (deterministic pass/fail
    against the synthetic corpus's ground truth, not a subjective score)
  - missing dependencies: count of required ids that did NOT survive
    (should always be 0 for a correct pruner)

It deliberately does NOT claim an LLM latency number. See the article's
optional real-API validation section for that, run separately and
labeled as illustrative.

Run from the project root:
    python benchmarks/benchmark.py

Zero external dependencies (stdlib only).
"""

import os
import sys
import time
import statistics
from dataclasses import dataclass

# Allow running this script directly (python benchmarks/benchmark.py)
# without installing the package first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_pruning import (
    generate_corpus,
    ALL_WORKLOADS,
    WorkloadConfig,
    PromptPruner,
    PromptBuilder,
)


@dataclass
class BenchmarkResult:
    workload_name: str
    num_turns: int
    input_message_count: int
    output_message_count: int
    input_tokens: int
    output_tokens: int
    token_reduction_pct: float
    expired_removed: int
    duplicates_removed: int
    restored_for_dependency: int
    required_facts_preserved: bool
    missing_dependency_count: int
    prune_time_ms: float
    build_time_before_ms: float
    build_time_after_ms: float
    tokens_removed_per_ms: float
    idempotent: bool


def _time_ms(fn, repeats=30, warmup=2):
    """
    Run fn() once (discarded) to warm up, then `repeats` more times,
    returning (result_of_last_call, median_ms). The warm-up call rules
    out first-call cold-start effects (e.g. attribute lookups, small
    list allocations) as an explanation for occasional "after > before"
    readings on tiny sub-millisecond operations.
    """
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return result, statistics.median(times)


def run_benchmark(num_turns: int, workload: WorkloadConfig, seed: int = 42) -> BenchmarkResult:
    corpus = generate_corpus(num_turns=num_turns, workload=workload, seed=seed)
    messages = corpus.messages

    builder = PromptBuilder()
    pruner = PromptPruner()

    _, build_before_ms = _time_ms(lambda: builder.build(messages))

    input_tokens = sum(m.approx_token_count() for m in messages)

    pruned_messages, report = None, None

    def do_prune():
        nonlocal pruned_messages, report
        pruned_messages, report = pruner.prune(messages)
        return pruned_messages

    _, prune_time_ms = _time_ms(do_prune)
    pruned_messages, report = pruner.prune(messages)

    output_tokens = sum(m.approx_token_count() for m in pruned_messages)

    _, build_after_ms = _time_ms(lambda: builder.build(pruned_messages))

    reduction_pct = 0.0
    if input_tokens > 0:
        reduction_pct = 100.0 * (input_tokens - output_tokens) / input_tokens

    surviving_ids = {m.id for m in pruned_messages}
    missing_required = corpus.required_ids - surviving_ids
    required_facts_preserved = len(missing_required) == 0

    # Idempotence check: pruning an already-pruned prompt should be a
    # no-op. prune(prune(messages)) must yield exactly the same
    # surviving message ids as prune(messages) -- a fixed point after
    # one pass. This is checked on message ids (not object identity),
    # so it's a structural equality check, not just "didn't crash."
    twice_pruned_messages, _twice_report = pruner.prune(pruned_messages)
    twice_ids = {m.id for m in twice_pruned_messages}
    idempotent = twice_ids == surviving_ids

    tokens_removed = input_tokens - output_tokens
    tokens_removed_per_ms = (tokens_removed / prune_time_ms) if prune_time_ms > 0 else 0.0

    return BenchmarkResult(
        workload_name=workload.name,
        num_turns=num_turns,
        input_message_count=len(messages),
        output_message_count=len(pruned_messages),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_reduction_pct=round(reduction_pct, 2),
        expired_removed=report.expired_removed,
        duplicates_removed=report.duplicates_removed,
        restored_for_dependency=report.restored_for_dependency,
        required_facts_preserved=required_facts_preserved,
        missing_dependency_count=len(missing_required),
        prune_time_ms=round(prune_time_ms, 4),
        build_time_before_ms=round(build_before_ms, 4),
        build_time_after_ms=round(build_after_ms, 4),
        tokens_removed_per_ms=round(tokens_removed_per_ms, 1),
        idempotent=idempotent,
    )


def print_result(r: BenchmarkResult, show_internal_metrics: bool = False):
    print(f"--- Workload: {r.workload_name} | Conversation size: {r.num_turns} turns ---")
    print(f"Required facts kept: {r.required_facts_preserved}  "
          f"(missing: {r.missing_dependency_count})")
    print(f"Idempotent (prune(prune(x)) == prune(x)): {r.idempotent}")
    print(f"Messages:            {r.input_message_count} -> {r.output_message_count}")
    print(f"Tokens (approx):     {r.input_tokens:,} -> {r.output_tokens:,}  "
          f"({r.token_reduction_pct}% reduction)")
    print(f"Expired removed:     {r.expired_removed}")
    print(f"Duplicates removed:  {r.duplicates_removed}")
    print(f"Restored (deps):     {r.restored_for_dependency}")
    print(f"Prune overhead:      {r.prune_time_ms} ms  (median of 30 runs, 2 warm-up)")
    if show_internal_metrics:
        # tokens_removed_per_ms is kept as a field on BenchmarkResult for
        # internal analysis, but isn't part of the default published
        # output -- it isn't directly actionable for a reader the way
        # "tokens removed" / "overhead" / "reduction %" are.
        print(f"[internal] Tokens removed/ms: {r.tokens_removed_per_ms}")
    print(f"Build time before:   {r.build_time_before_ms} ms")
    print(f"Build time after:    {r.build_time_after_ms} ms")
    print()


STANDARD_SIZES = (50, 200, 500, 1000, 2000)


def run_all_benchmarks(sizes=STANDARD_SIZES, workloads=ALL_WORKLOADS, seed: int = 42):
    """
    Run the benchmark across every (workload, size) combination and
    return a flat list of BenchmarkResult objects. Used both by the
    __main__ block below and by generate_charts.py, so the charts are
    always built from the exact same run function as the printed
    benchmark output -- no separate code path that could drift.
    """
    results = []
    for workload in workloads:
        for n in sizes:
            results.append(run_benchmark(num_turns=n, workload=workload, seed=seed))
    return results


if __name__ == "__main__":
    for result in run_all_benchmarks():
        print_result(result)
