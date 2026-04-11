# Minimal Patch: Adding a Fix Loop to `ExperimentRunner.run_loop()`

## Problem

Current flow in `run_loop()`:

```
LLM generates code → run_experiment() → if error: result.improved=False, kept=False → next iteration
```

When code **crashes or fails to run**, the loop treats it as "no improvement" and moves on,
never giving the LLM a chance to fix the error.

## Desired Flow (after patch)

```
run_experiment() → if error:
    fix_code(llm, failed_code, error)  ← NEW: call LLM with error context
    → run_experiment() again with fixed code
    → if still error after N retries: abandon this iteration
→ proceed to keep/discard + next improvement
```

This is exactly the same pattern as `experiment_repair._get_repaired_code()` and
`experiment_diagnosis` in this project — give the LLM the error, let it fix.

---

## Changes (minimal, only `experiment/runner.py`)

### 1. Add `_fix_code()` method to `ExperimentRunner`

Add this method to the `ExperimentRunner` class (after `_improve_code`):

```python
def _fix_code(
    self,
    llm: _ChatClient,
    failed_code: str,
    error: str,
) -> str:
    """Given a runtime error, ask the LLM to produce a fixed version of the code."""
    prompt = (
        "The following experiment code crashed. Fix the bug(s) that caused the error.\n\n"
        f"Error message:\n{error}\n\n"
        "Faulty code:\n"
        "```python\n"
        f"{failed_code}\n"
        "```\n\n"
        "Return only the corrected Python code."
    )
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            system="You are an expert ML experimentation assistant.",
        )
    except Exception:
        return failed_code  # fallback: keep the failing code

    candidate = getattr(response, "content", "")
    if not candidate or not isinstance(candidate, str):
        return failed_code

    return self._extract_python_code(candidate)
```

### 2. Modify `run_loop()` to use `_fix_code()` on error

In `run_loop()`, after running the experiment, check for errors and retry:

Find this section in `run_loop()`:

```python
for iteration in range(1, self.config.max_iterations + 1):
    next_code = self._improve_code(llm, current_code, self.history)
    result = self.run_experiment(next_code, run_id=run_id, iteration=iteration)
    current_code = next_code
```

Change it to:

```python
for iteration in range(1, self.config.max_iterations + 1):
    next_code = self._improve_code(llm, current_code, self.history)
    result = self.run_experiment(next_code, run_id=run_id, iteration=iteration)

    # NEW: if experiment crashed, attempt one repair round before keep/discard
    if result.error is not None and iteration > 0:  # skip baseline (iter 0)
        fixed_code = self._fix_code(llm, next_code, result.error)
        if fixed_code != next_code:
            result_fixed = self.run_experiment(fixed_code, run_id=run_id, iteration=iteration)
            if result_fixed.error is None:
                result = result_fixed  # use the fixed result
                next_code = fixed_code

    current_code = next_code
```

### 3. Optional: also handle error in baseline (iteration 0)

If the **initial** code also crashes, you may want to fix it before establishing baseline.
Currently `run_loop` does:

```python
baseline = self.run_experiment(current_code, run_id=run_id, iteration=0)
```

Add after it:

```python
if baseline.error is not None:
    fixed_baseline = self._fix_code(llm, current_code, baseline.error)
    if fixed_baseline != current_code:
        baseline_fixed = self.run_experiment(fixed_baseline, run_id=run_id, iteration=0)
        if baseline_fixed.error is None:
            current_code = fixed_baseline
            baseline = baseline_fixed
```

---

## Key Design Points

1. **`iteration > 0` guard**: Skip fix for the forced baseline run (iteration 0) unless you want auto-fix from the start. In this project, the baseline is "what you start with" — fixing it is your call.

2. **Re-run sandbox after fix**: Calling `_fix_code` only generates new text. You must `run_experiment()` the fixed code to verify it actually works.

3. **Single fix attempt per iteration**: If the first fix still crashes, abandon this iteration rather than entering an infinite fix loop. This matches the "max 1 re-run" pattern in `experiment_repair.py` cycle.

4. **Metrics vs. error**: If `result.error is None` but `result.primary_metric is None` (no metrics produced), that is a separate case — it may mean the experiment silently exited. Consider treating "no metrics + exit 0" as an error too.

---

## If You Want the Full Diagnosis → Fix Loop (bigger change)

The above patch only passes the raw `stderr`. For more targeted fixes, integrate
`experiment_diagnosis.py` (the `DeficiencyType` classifier) into the fix step:

```
error + failed_code → diagnose_experiment() → DeficiencyType
  → build_repair_prompt() with suggested_fix from Deficiency
  → llm.chat() with structured diagnosis
  → run_experiment() with fixed code
```

This is exactly what `experiment_repair.run_repair_loop()` does in this project.
See `experiment_diagnosis.py:DeficiencyType` and `experiment_repair.py:build_repair_prompt()`.
