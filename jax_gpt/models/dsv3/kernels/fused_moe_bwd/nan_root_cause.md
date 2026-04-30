# NaN Root Cause Investigation

> **Status (2026-04-01):** Initial max_tpe hypothesis was WRONG (see §Retraction).
> True root cause is under investigation. The NaN requires FSDP>1 AND EP>1 simultaneously.

---

## ~~Hypothesis 1 (retracted): `max_tpe` formula wrong~~ ❌

**Retracted.** The formula `max_tpe = 2 * T_fsdp * K // cfg.E` is actually correct because:
- avg tokens per local expert = T_fsdp × K × (E_local/E) / E_local = **T_fsdp × K / E**
- So `2 * T_fsdp * K / E` = 2× average ✓

Verified by `test_maxtpe_ep4.py` (EP=4, FSDP=1, T=32, E=16, K=4):
- max_tpe_buggy=16 = 2× avg=8 → no overflow, ratio=1.0000 → bug **not reproduced**
- max_tpe_correct=64 also passes (same result)

**Conclusion: the NaN is NOT caused by max_tpe overflow.**

---

## What we know about the actual bug

| Config | NaN? | Notes |
|--------|------|-------|
| EP=2, FSDP=2 (local tests) | No | All 5 gradient test cases pass |
| EP=4, FSDP=1 (local test) | No | Verified by `test_maxtpe_ep4.py` |
| EP=4, FSDP=16 (cluster v82, v93) | **Yes** | `loss: nan` at every step ≥ 5 |
| EP=4, FSDP=1 (cluster v83 jax ref) | No | `loss: 11.43` stable |

**Pattern:** NaN requires FSDP > 1 AND EP > 1 simultaneously. Testing FSDP=2, EP=4 needs 8
devices (we have 4) — cannot be locally reproduced.

---

## Hypothesis 2 (active): FSDP-EP interaction in `psum(g_l, "fsdp")`

The streaming_bwd backward (model.py:973) does:
```python
g_l_fsdp = jax.lax.psum(g_l, "fsdp")
g_full   = jax.lax.all_gather(g_l_fsdp, "ep", axis=0, tiled=True)
```

This is the VJP of the forward's final `psum("fsdp")` + `all_gather("ep")`. The psum
amplifies the incoming gradient by FSDP (8× at FSDP=2, 16× at FSDP=16). Since the
forward also broadcasts via psum, jax.vjp would compute the same amplification — so in
principle this should be correct.

However, at large FSDP the `psum(g_l, "fsdp")` is summing gradients for **16 different
token positions** (different FSDP stripes) into the same (T_ep, D) tensor. This is
correct VJP algebra, but whether the streaming_bwd then uses this correctly for all of
d_tok, d_w1, d_w2 simultaneously is under investigation.

---

## Next steps

1. Test FSDP=2, EP=4 on an 8-device setup (needs DWS Flex or ninja cluster).
2. Add a cluster job with jax.vjp backend at FSDP=16 to confirm baseline is stable.
3. Add `jax.debug.print` inside `_streaming_bwd_fn` to check `g_l_fsdp` norm is not NaN.
4. Test without `--gradient_checkpoint` — checkpoint changes backward recomputation and
   might interact with the custom_vjp in unexpected ways.

---

## Background: what `max_tpe` controls

Inside `shard_map`, after `all_gather("ep")`, each device holds:
- `T_fsdp = GBS / FSDP` tokens (the full FSDP stripe)
- `E_local = E / EP` local experts

Both the forward (`_expert_mlp_ep_body_ep_sharded`) and backward
(`fused_ep_moe_bwd_streaming`) use `max_tpe` as the **static bin size per expert**:

```
expert e's slots:  bins[e * max_tpe : (e+1) * max_tpe]
```

If any expert receives more than `max_tpe` token assignments:
- **Forward**: `dynamic_slice(..., (max_tpe,))` silently truncates — only the first `max_tpe`
  tokens contribute to the output. The rest are dropped with no error.
- **Backward**: `bin_positions = expert_id * max_tpe + local_index`. When `local_index >= max_tpe`,
  the slot overflows into the **next expert's bin**, corrupting that expert's gradient.
  Slots that overflow past `E_local * max_tpe` are silently dropped (OOB scatter, safe).

---

## The bug

```python
# model.py ~line 759  (expert_mlp_jax)
T_fsdp = B * S // fsdp_size
max_tpe = max(1, 2 * T_fsdp * K // cfg.E)   # ← BUG: cfg.E is global count
```

The intended meaning is "2× the average number of token assignments per **local** expert".
But the average per local expert is:

```
avg_per_local_expert = T_fsdp * K / E_local = T_fsdp * K * EP / E
```

Using `cfg.E` (global) instead of `E_local` underestimates by a factor of `EP`.

---

## Numbers at FSDP=16, EP=4, mini config (E=16, K=4, GBS=256)

| Quantity                            | Value  |
|-------------------------------------|--------|
| T_fsdp = GBS / FSDP                 | 16     |
| E_local = E / EP                    | 4      |
| avg tokens per local expert         | **16** |
| `max_tpe` (buggy formula)           | **8**  |
| `max_tpe` (correct formula)         | **32** |

With `max_tpe = 8` and an average of 16 tokens per expert, **roughly half of all token
assignments overflow** every expert's bin. Experts 0–2 spill into the next expert's bin
(corrupted gradients). Expert 3 spills past `pad_to` (silently dropped).

The streaming backward then computes wildly wrong weight gradients. After one SGD step
with these gradients the weights are corrupted, the next forward pass overflows, and loss
becomes NaN at step 5 (the first step that prints loss).

`jax.vjp` (v83 reference backend) also uses the wrong forward (only half of each expert's
tokens contribute), but it correctly differentiates *that specific wrong forward* — so
gradients are consistent with the loss, there is no mismatch, and training produces finite
(if slightly wrong) loss ≈ 11.43.

---

## Why local tests did NOT catch this

The local tests (`test_streaming_bwd_local.py`) run at **FSDP=2, EP=2** with a toy model
(`E=4, K=2, T=16`).

| Quantity                            | Local (FSDP=2, EP=2) | Cluster (FSDP=16, EP=4) |
|-------------------------------------|----------------------|--------------------------|
| T_fsdp                              | 8                    | 16                       |
| E_local                             | 2                    | 4                        |
| avg tokens per local expert         | **8**                | **16**                   |
| `max_tpe` (buggy formula)           | **8**                | **8**                    |
| slack = max_tpe / avg               | **1.0×** (at limit)  | **0.5×** (below avg!)    |

At FSDP=2, EP=2: the buggy formula accidentally gives `max_tpe = avg` (no safety margin,
but not below the average). Whether this triggers overflow depends on the routing
distribution for that specific random seed.

**Why the test seed works:** with `T_fsdp=8`, `K=2`, `E_local=2`, each local expert
receives on average `8 * 2 / 2 = 8` slots. The fixed seed (`SEED=0`) produces a routing
where both local experts get ≤ 8 tokens, so no overflow occurs. The test passes by luck of
the draw.

At FSDP=16, EP=4: `max_tpe = avg/2` — overflow is **guaranteed** for any routing since
the average already exceeds the bin size. No seed can avoid it.

**The structural gap:** the local tests only cover `EP ≤ FSDP` regimes where the formula
error is masked. The cluster is the first run with a high enough EP/FSDP ratio to expose it.

---

## The fix

```python
# model.py ~line 759  (expert_mlp_jax)
T_fsdp = B * S // fsdp_size
E_local = cfg.E // ep_size          # ← add this line
max_tpe = max(1, 2 * T_fsdp * K // E_local)   # ← was cfg.E
```

After fix at FSDP=16, EP=4, mini config:
- `max_tpe = 2 * 16 * 4 // 4 = 32` — 2× the average of 16 ✓

After fix at FSDP=2, EP=2, local test:
- `max_tpe = 2 * 8 * 2 // 2 = 16` — 2× the average of 8 ✓ (previously 8 = 1× average)

---

## Validation plan

1. Apply the one-line fix locally.
2. Re-run `test_streaming_bwd_local.py` — all 5 cases should still pass (they will now run
   with the larger `max_tpe=16` instead of `8`, which is strictly safer).
3. Build `v94-train` Docker image and submit a new `2x4x4` cluster job mirroring v93 but
   with the fix. Expect finite loss matching the v83 jax reference (~11.43).
4. Optionally add a test case at FSDP=4, EP=4 (or FSDP=2, EP=4 using 8 devices) that
   directly exercises the formula with EP > 1.

---

## File locations

| File | Relevant section |
|------|-----------------|
| `dsv3/mini_dsv3/model.py:759` | `max_tpe` formula (the fix) |
| `dsv3/fused_moe_bwd/backward_kernel.py:712-724` | 1D padded bin layout (where overflow corrupts) |
| `dsv3/mini_dsv3/model.py:606-628` | Forward `dynamic_slice` with `max_tpe` (where tokens are truncated) |
