# Novelty Dedup Design for Parallel AutoResearch

## Background

`parallel_runner.py` uses independent `claude -p` calls to generate and analyze experiments. These calls do not share conversation state. Agents can only recover history through files such as `results.tsv`, `autorunner/candidates`, and `autorunner/artifacts`.

The current system often repeats experiment directions because completed history is visible, but active work is not reliably coordinated. In a parallel runner, queued, running, and still-generating ideas must also reserve their direction space.

Embedding-based semantic deduplication is useful as a retrieval aid, but it is not reliable enough to be the final decision maker. Experiment novelty depends on code symbols, parameter values, mechanisms, hypotheses, and outcome context, not just natural-language similarity.

## Design Principle

The runner owns coordination and hard gates. Agents provide intelligence, but the script decides whether a candidate is allowed to enter the queue.

Use:

1. Structured experiment memory.
2. Independent novelty judge agent.
3. Runner-enforced blocking of duplicate candidates.
4. Embedding only as a shortlist/retrieval tool, not as final judgment.

## Experiment Registry

Add a global registry file:

`autorunner/artifacts/experiment_registry.jsonl`

Every candidate should be registered from the proposal stage, not only after training finishes.

Each row should include fields like:

```json
{
  "candidate_id": "cand-000123",
  "status": "proposed|queued|running|keep|discard|blocked_duplicate|generation_failed",
  "created_at": "ISO8601 time",
  "updated_at": "ISO8601 time",
  "parent_ref": "best_train.py",
  "direction_key": "lr_schedule.warmup_length",
  "hypothesis": "Longer warmup may stabilize early training.",
  "mechanism": "Increase warmup steps while slightly lowering peak LR.",
  "touched_symbols": ["WARMUP_STEPS", "LR"],
  "changed_upper_keys": ["WARMUP_STEPS", "LR"],
  "description": "Increase warmup steps with slightly lower peak LR.",
  "novelty_claim": "Differs from prior LR experiments by changing warmup duration.",
  "result": {
    "status": "completed",
    "decision": "keep|discard",
    "val_bpb": 1.234567,
    "discard_reason": "metric_not_improved"
  }
}
```

The registry should include active states:

- `proposed`
- `queued`
- `running`

These active entries reserve experiment directions and must be shown to future agents.

## Generator Output Contract

The generator/refine agent should output structured fields, not only a free-form `DESCRIPTION`.

Minimum required fields:

```text
DIRECTION_KEY: <short hierarchical key>
HYPOTHESIS: <why this could improve val_bpb>
MECHANISM: <what code/parameter mechanism changes>
TOUCHED_SYMBOLS: <comma-separated symbols or concepts>
NOVELTY_CLAIM: <how this differs from similar prior work>
DESCRIPTION: <short human-readable description>
```

`DESCRIPTION` is for humans. The runner and novelty judge should primarily use `direction_key`, `hypothesis`, `mechanism`, `touched_symbols`, `changed_upper_keys`, and the code diff.

## Preferred Two-Stage Flow

The robust design is:

1. `propose`: agent proposes a structured experiment JSON without editing code.
2. Runner performs novelty review.
3. If approved, `implement`: agent edits `train.py` to implement the approved proposal.
4. Runner verifies and queues the candidate.

This gives the runner a chance to reject duplicate ideas before spending effort on code edits.

## Minimal Compatible Flow

If a full two-stage flow is too large for the first patch, keep the current refine flow but add a post-generation novelty gate:

1. Agent edits candidate `train.py`.
2. Runner extracts structured proposal fields and `DESCRIPTION`.
3. Runner computes `changed_upper_keys` and a concise diff summary.
4. Runner calls the novelty judge.
5. If duplicate, mark candidate as `blocked_duplicate` and do not queue it.
6. Optionally retry generation with the judge's `required_pivot`, capped at 2-3 attempts.

## Novelty Judge Agent

Use an independent judge agent, separate from the generator. The generator should not judge its own novelty.

The judge receives:

- Current candidate proposal fields.
- Current candidate diff summary.
- Active `proposed|queued|running` registry entries.
- Recent completed experiments.
- Same-bucket historical experiments.
- Embedding/keyword-retrieved top K similar historical items.

The judge must output strict JSON:

```json
{
  "is_duplicate": true,
  "duplicate_level": "exact|near|same_family|novel",
  "nearest_candidate_ids": ["cand-000012"],
  "reason": "Same hypothesis and same knobs as a prior warmup experiment.",
  "allow_run": false,
  "required_pivot": "Explore optimizer beta schedule instead of warmup length."
}
```

Runner policy:

- `allow_run=true`: register and queue the candidate.
- `allow_run=false`: block the candidate and record `candidate_blocked_duplicate`.
- `same_family` may be allowed only if the judge explains a concrete experimental difference and `allow_run=true`.

## Embedding Role

Keep embedding, but demote it to retrieval.

Good uses:

- Fetch top K similar prior descriptions.
- Surface likely duplicates for the judge.
- Detect obvious exact/near text repeats.

Bad use:

- Making the final queue/block decision by threshold alone.

Reason: embeddings are weak at numeric distinctions, code-level mechanisms, and small experimental differences.

## Active Direction Coordination

Before each generation, the runner should summarize occupied direction space:

```text
Active directions:
- cand-000120 queued lr_schedule.warmup_length: longer warmup + lower LR
- cand-000121 running architecture.width: increase ASPECT_RATIO
```

Pass this into the generator prompt. The generator must avoid those active directions unless it makes a clearly different mechanism-level change.

## Direction Buckets

Optionally assign each new candidate a bucket to improve parallel diversity:

- `architecture`
- `attention`
- `normalization`
- `activation`
- `optimizer`
- `lr_schedule`
- `batching`
- `regularization`
- `systems_speed`

The generation loop can lease a bucket to each candidate so multiple agents do not all explore the same area at once.

## Runner Integration Points

Likely modifications in `parallel_runner.py`:

1. Add `EXPERIMENT_REGISTRY_FILE`.
2. Add registry load/append/update helpers.
3. Extend `CandidateTask` with structured novelty fields.
4. Include active registry summary in `_create_candidate()`.
5. Parse structured fields from agent output.
6. Add a novelty judge call before queueing.
7. Block duplicate candidates before they enter `queued`.
8. Update registry on `queued`, `running`, `keep`, `discard`, and generation failures.
9. Log queue events such as `candidate_blocked_duplicate`.
10. Retry generation a small number of times when blocked for duplication.

Likely modifications in `claude_code_agent.py` and `subagent.yaml`:

1. Add or extend prompt templates for structured proposal fields.
2. Add a novelty judge template.
3. Add `ClaudeCodeAgent.judge_novelty(...)`.
4. Ensure judge output is strict JSON and does not edit files.

## Implementation Checklist

- [ ] Create registry schema and helpers.
- [ ] Add active direction summaries.
- [ ] Make generator output structured novelty fields.
- [ ] Add independent novelty judge prompt and agent method.
- [ ] Add hard queue gate before `candidate_queued`.
- [ ] Record duplicate blocks in registry and queue events.
- [ ] Keep embedding as top-K retrieval only.
- [ ] Add bounded retry with judge-provided pivot guidance.
- [ ] Verify that queued/running directions are visible to future candidates.
- [ ] Verify that duplicate candidates do not consume GPU time.

