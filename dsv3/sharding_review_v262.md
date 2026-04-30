# DSv3 671B — Sharding & Remat Review (v262-2k)

Config: FSDP=256, EP=1, TP=2, GBS=2048, `gmm` backend, 4×8×8 (512 devices, 256 chips).

## Mesh

```
(dp=1, ep=1, fsdp=256, tp=2) = 512 devices, 256 chips
4×8×8 topology, devices reshaped as (1, 1, 256, 2)
```

Sharding helper shortcuts (model.py:1523-1531):
```python
_batch_ax = "fsdp"                         # dp=1, so batch sharded on fsdp
_col = P("fsdp", "tp")                     # column-parallel: shard input by fsdp, output by tp
_row = P("tp", None)                       # row-parallel: shard input by tp, output replicated
_moe_wi = P("ep", "fsdp", "tp")           # MoE gate/up: E on ep, D on fsdp, D_moe on tp
_moe_wo = P("ep", "tp", "fsdp")           # MoE down: E on ep, D_moe on tp, D on fsdp
```

---

## Embedding

| Tensor | Global Shape | Spec | Per-Device Shape | Size/dev |
|--------|-------------|------|-----------------|----------|
| `embed` | (102400, 7168) | P(None, "fsdp") | (102400, 28) | 11 MB |
| `tokens` | (2048, 4096) int32 | P("fsdp", None) | (8, 4096) | 128 KB |
| `x` after embed | (2048, 4096, 7168) bf16 | — | (8, 4096, 7168) | 448 MB |

**Sharding constraint** at model.py:1660:
```python
x = with_sharding_constraint(x, P("fsdp", None, None))
```

> **VS: Does this apply to all the layer outputs as well?**
>
> This constraint applies once after embedding (model.py:1660). Inside the MoE scan,
> there's a second identical constraint at model.py:1612 that re-annotates x every
> iteration. Dense layers have no explicit constraint inside the scan body — GSPMD
> maintains sharding from the residual add (`x = x + attn_out`), since both sides
> are P("fsdp", None, None), the result stays P("fsdp", None, None).
> So effectively: yes, all layer inputs/outputs maintain this sharding.

Per-device: **(8, 4096, 7168) = 448 MB**. D is **replicated** across TP — both TP ranks hold identical x.

**Collective**: AllGather `embed` on fsdp for the gather lookup (chunked to satisfy SC 65536-row limit).

---

## Attention (shared by dense + MoE layers, ×61 total)

### Q projection

| Weight | Global Shape | Spec | Per-Dev Shape | Collective |
|--------|-------------|------|--------------|-----------|
| `wq_a` | (7168, 1536) | P("fsdp", None) | (28, 1536) | **AG(fsdp)** 22 MB |
| `q_norm_scale` | (1536,) | P(None) | (1536,) | none |
| `wq_b` | (1536, 24576) | P("fsdp", "tp") | (6, 12288) | **AG(fsdp)** 72 MB |

```
h @ wq_a:    (8,4096,7168) @ AG→(7168,1536) → (8,4096,1536)   P("fsdp",None,None)
q_a @ wq_b:  (8,4096,1536) @ AG→(1536,12288) → (8,4096,12288)  P("fsdp",None,"tp")
reshape → (8, 4096, 64, 192)  — 64 heads per TP rank (128 total)
```

### KV projection

| Weight | Global Shape | Spec | Per-Dev Shape | Collective |
|--------|-------------|------|--------------|-----------|
| `wkv_a` | (7168, 576) | P("fsdp", None) | (28, 576) | **AG(fsdp)** 8 MB |
| `kv_norm_scale` | (512,) | P(None) | (512,) | none |
| `wkv_b` | (512, 32768) | P("fsdp", "tp") | (2, 16384) | **AG(fsdp)** 32 MB |

```
h @ wkv_a:       (8,4096,7168) @ AG→(7168,576) → (8,4096,576)
kv_a @ wkv_b:    (8,4096,512) @ AG→(512,16384) → (8,4096,16384) P("fsdp",None,"tp")
reshape → k_nope: (8,4096,64,128), v: (8,4096,64,128), k_rope: (8,4096,64,64)
```

### Splash Attention (shard_map)

```python
# model.py:263 — TP>1 path
_bspec = P("fsdp", "tp", None, None)   # B by fsdp, H by tp

shard_map(in_specs=(_bspec, _bspec, _bspec), out_specs=_bspec)
```

Per-device inside shard_map:
```
q_t: (8, 64, 4096, 192)    # B_local=8, H_tp=64 heads, S=4096, qk_dim=192
k_t: (8, 64, 4096, 192)
v_t: (8, 64, 4096, 128)
```
**No collective** — fully local Splash Attention on each device's head shard.

### Output projection (row-parallel)

| Weight | Global Shape | Spec | Per-Dev Shape | Collective |
|--------|-------------|------|--------------|-----------|
| `w_out` | (16384, 7168) | P("tp", None) | (8192, 7168) | **AR(tp)** 448 MB |

```
attn_flat: (8,4096,8192) — H_tp × d_v per TP rank
attn_flat @ w_out: (8,4096,8192) @ (8192,7168) → (8,4096,7168)
GSPMD inserts AllReduce(tp) to sum partial products → full (8,4096,7168)
```

> **VS: what if we put a sharding constraint on d_model here by TP, we will get a reduce-scatter instead of all-reduce**
>
> Exactly right. Constraining the output to P("fsdp", None, "tp") would:
> - Turn AR(tp) 448 MB → **RS(tp) 224 MB** (half the volume)
> - Output becomes (8, 4096, 3584) per device — D sharded by TP
> - **Carry halved**: 448 → 224 MB/layer, total 26.7 → 13.3 GB
> - But the next layer's column-parallel matmul needs full D →
>   adds **AG(tp) 224 MB** on the activation before each wq_a/wkv_a matmul
>
> Net TP comm per layer: RS 224 + AG 224 = 448 MB = same as current AR 448 MB.
> Same total bytes, but **carry is halved**. The real win.
>
> **However**: the v251 profile shows AR(tp) at 171 GB/s — this is ICI bandwidth,
> not the ~540 GB/s d2d link between TCs within a chip. The TP axis is mapped to
> ICI, not d2d. Fixing the TP→d2d mapping should be done first — it would make
> all TP collectives 3× faster regardless of AR vs RS+AG decomposition.

### Attention collective summary (per layer)

| Collective | Volume |
|-----------|--------|
| 4× AG(fsdp): wq_a + wq_b + wkv_a + wkv_b | 134 MB |
| 1× AR(tp): w_out output | 448 MB |

---

## Dense MLP (layers 0-2 only, ×3)

| Weight | Global Shape | Spec | Per-Dev Shape | Collective |
|--------|-------------|------|--------------|-----------|
| `wi_gate` | (7168, 18432) | P("fsdp", "tp") | (28, 9216) | **AG(fsdp)** 252 MB |
| `wi_up` | (7168, 18432) | P("fsdp", "tp") | (28, 9216) | **AG(fsdp)** 252 MB |
| `wo_mlp` | (18432, 7168) | P("tp", None) | (9216, 7168) | **AR(tp)** 448 MB |

> **VS: if d_model is sharded on TP, this will add an all-gather on the activation.
> And reduce scatter on the output.**
>
> Yes. With D-sharded-by-TP carry:
> - x enters as (8, 4096, 3584) P("fsdp", None, "tp")
> - Need **AG(tp) 224 MB** on x to reconstruct full D before h @ wi_gate
> - Column-parallel matmul unchanged: produces (8, 4096, 9216) per TP rank
> - wo_mlp output: **RS(tp) 224 MB** instead of AR(tp) 448 MB
> - Same total TP bytes (224+224=448), but carry is half the size

```
Column-parallel: h @ AG→wi_gate → (8,4096,9216) per TP rank
                 h @ AG→wi_up   → (8,4096,9216) per TP rank
Row-parallel:    hidden @ wo_mlp → AllReduce(tp) → (8,4096,7168)
```

### Dense layer total collectives (per layer)

| Collective | Volume |
|-----------|--------|
| Attn: 4× AG(fsdp) | 134 MB |
| Attn: 1× AR(tp) | 448 MB |
| MLP: 2× AG(fsdp) | 504 MB |
| MLP: 1× AR(tp) | 448 MB |
| **Dense layer total** | **638 MB AG + 896 MB AR = 1.5 GB** |

---

## MoE Layer (layers 3-60, ×58) — `gmm` backend

### Sharding constraint (model.py:1609-1613)

```python
x = checkpoint_name(x, "moe_layer_input")        # for potential host offload
x = with_sharding_constraint(x, P("fsdp", None, None))  # re-annotate inside scan
```

> **VS: same change in the sharding constraint, here i.e. d_model sharded on TP**
>
> Yes — would change to `P("fsdp", None, "tp")`. Per-device carry drops from
> (8, 4096, 7168)=448 MB to (8, 4096, 3584)=224 MB. Total MoE carry: 58×224=12.7 GB.

```python
# (continued)
```

### Attention

Same as above: 134 MB AG(fsdp) + 448 MB AR(tp).

### MoE Routing

| Weight | Global Shape | Spec | Per-Dev Shape | Collective |
|--------|-------------|------|--------------|-----------|
| `gate` | (7168, 256) | P(None, None) | (7168, 256) | none (replicated) |
| `gate_bias` | (256,) f32 | — | (256,) | none |

Gate is fully replicated — each device computes identical routing decisions. No collective.

### MoE Experts (AllReduce-GMM, shard_map)

```python
# model.py:878 — TP>1 path
shard_map(
    in_specs=(P(("fsdp","tp"), None),    # activations: batch sharded fsdp×tp
              P(("fsdp","tp"), None),     # indices
              P(("fsdp","tp"), None),     # weights
              P("ep","fsdp","tp"),        # wi_0
              P("ep","fsdp","tp"),        # wi_1
              P("ep","tp","fsdp")),       # wo
    out_specs=P(("fsdp","tp"), None))
```

> **VS: if we use TP on d_model, we will get an all-gather on activation to reconstruct D**
>
> Yes. With D-sharded carry, x enters the MoE shard_map with D/tp per device.
> Inside the body, the ragged_dot needs full D for `local_x @ wi` matmuls.
> Two options:
> 1. AG(tp) on activation **before** shard_map entry — reconstruct full D, then
>    shard_map proceeds as today. Output RS(tp) to re-shard D.
> 2. AG(tp) **inside** shard_map body — gather D within the body before ragged_dot.
>    But this interacts with the existing AG(fsdp) on weights — would need both.
>
> Option 1 is simpler (no shard_map changes). Volume: 224 MB AG + 224 MB RS per layer.

**Per-device shapes inside shard_map:**

| Tensor | Per-Dev Shape | Notes |
|--------|--------------|-------|
| `flat_x` | (16384, 7168) | T_local = GBS×S / (fsdp×tp) = 16384 |
| `flat_indices` | (16384, 8) | top-K=8 expert indices |
| `flat_weights` | (16384, 8) f32 | routing weights |
| `wi_0` | (256, 28, 1024) | E=256, D/fsdp=28, D_moe/tp=1024 |
| `wi_1` | (256, 28, 1024) | same |
| `wo` | (256, 1024, 28) | E=256, D_moe/tp=1024, D/fsdp=28 |

**Inside body** (`_expert_mlp_gmm_ar_body`, model.py:833-867):

**Step 1: AllGather D on fsdp** (model.py:834-836)
```
wi_0: (256, 28, 1024)  →AG(fsdp,axis=1)→  (256, 7168, 1024)   3.5 GB
wi_1: (256, 28, 1024)  →AG(fsdp,axis=1)→  (256, 7168, 1024)   3.5 GB
wo:   (256, 1024, 28)  →AG(fsdp,axis=2)→  (256, 1024, 7168)   3.5 GB
```
**3× AllGather(fsdp) = 10.5 GB total weight data per MoE layer.**

**Step 2: Sort tokens by expert, extract local block**
```
max_local = T×K / EP = 16384×8 / 1 = 131072  (all tokens, since EP=1)
local_x: (131072, 7168) — gathered from flat_x by sorted token IDs
```

**Step 3: ragged_dot (gate/up)**
```
(131072, 7168) × (257, 7168, 1024) → (131072, 1024)   F_local=1024
```

**Step 4: ragged_dot (down)**
```
(131072, 1024) × (257, 1024, 7168) → (131072, 7168)
```

**Step 5: psum("tp")** (model.py:860)
```
AllReduce over TP: (16384, 7168) bf16 = 224 MB
Sums partial F contributions from each TP rank.
```

**Step 6: psum("ep")** — no-op (EP=1)

### Shared Expert

| Weight | Global Shape | Spec | Per-Dev Shape | Collective |
|--------|-------------|------|--------------|-----------|
| `shared_wi_0` | (7168, 2048) | P("fsdp", "tp") | (28, 1024) | **AG(fsdp)** 28 MB |
| `shared_wi_1` | (7168, 2048) | P("fsdp", "tp") | (28, 1024) | **AG(fsdp)** 28 MB |
| `shared_wo` | (2048, 7168) | P("tp", None) | (1024, 7168) | **AR(tp)** 448 MB |

> **VS: tp, fsdp ?**
>
> Currently P("tp", None) — D_moe sharded by tp, D replicated. Could use
> P("tp", "fsdp") → per-device (1024, 28), which would:
> - Eliminate the 448 MB AR(tp) on the output
> - Instead produce a partial-sum result that needs **RS(tp) + AG(fsdp)** or similar
> - But this changes the matmul semantics: `hidden @ shared_wo` with both dims sharded
>   requires a more complex collective pattern (reduce-scatter on tp, then the result
>   has D sharded by fsdp — needs AG(fsdp) to reconstruct for residual add)
>
> With D-sharded-by-TP carry (the proposed change above), the output naturally wants
> P("fsdp", None, "tp"), so shared_wo would stay P("tp", None) and use RS(tp) 224 MB
> instead of AR(tp) 448 MB. Adding fsdp sharding on the output dim isn't needed.

```
Column-parallel: x @ AG→shared_wi_0 → (8,4096,1024)
Row-parallel:    hidden @ shared_wo → AllReduce(tp) → (8,4096,7168)
```

### MoE layer total collectives (per layer)

| Component | Collective | Volume |
|-----------|-----------|--------|
| Attn weights | 4× AG(fsdp) | 134 MB |
| Attn w_out | 1× AR(tp) | 448 MB |
| MoE weights | 3× AG(fsdp) | 10.5 GB |
| MoE psum(tp) | 1× AR(tp) | 224 MB |
| Shared expert weights | 2× AG(fsdp) | 56 MB |
| Shared expert wo | 1× AR(tp) | 448 MB |
| **MoE layer total** | **9× AG(fsdp) + 3× AR(tp)** | **10.7 GB AG + 1.1 GB AR** |

---

## Scan Carry & Remat

```
Carry shape per device:  (8, 4096, 7168) bf16 = 448 MB
D is replicated across TP — both TP ranks store the full carry.

Dense scan:  3 layers × 448 MB =  1.3 GB
MoE scan:   58 layers × 448 MB = 25.4 GB
Total carry on HBM:               26.7 GB  (fits in 94.75 GB, no host offload)

Remat strategy: jax.checkpoint inside scan body.
  - Recomputes all layer intermediates (norm, projections, attention, MoE) in backward.
  - Scan still stores all 61 carry values on HBM for backward pass.
  - checkpoint_name("moe_layer_input") annotates carry for potential selective offload
    but carry_bytes formula is broken (missing D), so offload never triggers.
```

**carry_bytes bug** (model.py:1682-1684):
```python
B_l = B * S // (fsdp_size * tp_size)       # = 2048*4096/512 = 16384
carry_bytes = cfg.L_moe * B_l * 2          # = 58 * 16384 * 2 = 1.9 MB  ← WRONG

# Correct:
B_local = B // fsdp_size                    # = 2048/256 = 8  (TP doesn't shard batch)
carry_bytes = L_moe * B_local * S * D * 2  # = 58 * 8 * 4096 * 7168 * 2 = 25.4 GB
```

Two bugs: (1) D=7168 missing, (2) divides tokens by TP but D is replicated (TP doesn't reduce carry).
Result: `need_offload` is always False. At GBS=2048/FSDP=256 this doesn't matter (25.4 GB fits),
but at GBS=4096/FSDP=256/TP=2 the carry is 50.8 GB and offload SHOULD trigger but doesn't.

---

## Collective Summary (full step, forward pass only)

| Collective | Per Dense Layer | ×3 | Per MoE Layer | ×58 | Step Total |
|-----------|----------------|-----|--------------|------|-----------|
| AG(fsdp) attn | 134 MB | 402 MB | 134 MB | 7.8 GB | 8.2 GB |
| AG(fsdp) MoE wts | — | — | 10.5 GB | 609 GB | 609 GB |
| AG(fsdp) shared/dense MLP | 504 MB | 1.5 GB | 56 MB | 3.2 GB | 4.7 GB |
| **AG(fsdp) total** | | | | | **622 GB** |
| AR(tp) attn w_out | 448 MB | 1.3 GB | 448 MB | 26 GB | 27.3 GB |
| AR(tp) MoE psum | — | — | 224 MB | 13 GB | 13 GB |
| AR(tp) dense/shared wo | 448 MB | 1.3 GB | 448 MB | 26 GB | 27.3 GB |
| **AR(tp) total** | | | | | **67.6 GB** |

**Dominant cost: MoE weight AllGather(fsdp) = 609 GB/step (88% of all communication).**

This matches the v251 profile: `all-gather.293` (MoE weights, 40% overlap) and
`all-gather.295` (0.1% overlap) were the top communication bottlenecks,
with 3.18s total exposed stall = 15% of the 21.3s step.

---

## Backward Pass Collectives

The backward doubles most collectives (chain rule through the same matmuls):
- Each AG(fsdp) weight gather repeats in backward (GSPMD re-gathers for d_weight computation)
- Each AR(tp) repeats (gradients flow backward through the same row-parallel matmuls)
- MoE backward uses `jax.vjp` through `_expert_mlp_gmm_ar_body` (ragged_dot path),
  which re-does the same AllGather + psum pattern

**Estimated full-step (fwd+bwd) communication: ~1.4 TB AG(fsdp) + ~135 GB AR(tp)**

---

## TP AllReduce Bandwidth — ICI vs d2d (from v251 profile)

v251 profile shows AR(tp) achieving **171 GB/s**. On v7x, each chip has 2 TensorCores
connected via a device-to-device (d2d) link rated at **~540 GB/s**. The TP=2 axis should
map to this intra-chip d2d link, but 171 GB/s is consistent with **ICI bandwidth** instead.

**Root cause**: the mesh is created as `reshape(1, 1, 256, 2)` — 512 devices laid out
with tp as the innermost dimension. But JAX device ordering on 4×8×8 may not pair the
two TCs of each chip as adjacent indices. If device 0 and device 1 are on different chips,
the TP AllReduce goes over ICI instead of d2d.

**Impact**: all AR(tp) collectives (67.6 GB/step fwd) run at 171 GB/s instead of 540 GB/s.
At 540 GB/s, a 448 MB AR(tp) takes 0.83 ms; at 171 GB/s, it takes 2.6 ms — 3× slower.
Over 61 layers × 2-3 AR/layer × 2 (fwd+bwd) ≈ 300+ AR ops/step, this could add several
seconds of stall.

**Fix options**:
1. Verify device ordering: check if `jax.devices()[0]` and `jax.devices()[1]` share a chip.
   If not, reorder devices so TP pairs are intra-chip.
2. Use `Mesh` with explicit device assignment: `mesh_devices[..., 0]` and `mesh_devices[..., 1]`
   must be the two TCs of the same physical chip.
3. Check `jax.local_devices()` topology metadata for chip_id pairing.

---

## v264 — TP on D (reduction dim), FSDP on D_moe

**Result: 1740 TPS/chip, 18.8s/step, 28.2% MFU** (+13% over v251 baseline)

Profile: `gs://max-experiments/dsv3/profiles/v264-tp-d-fsdp256-tp2-gbs2048/`

### What changed

MoE weight sharding flipped: TP on D (reduction dim), FSDP on D_moe (non-reduction).

| | v258 (current before) | v264 (TP on D) | v251 (original) |
|---|---|---|---|
| wi spec | P("ep","fsdp","tp") | P("ep","tp","fsdp") | P("ep",None,("fsdp","tp")) |
| per-dev wi_0 | (256, 28, 1024) | (256, 3584, 8) | (256, 7168, 4) |
| D sharding | fsdp (28) | tp (3584) | full (7168) |
| F_local | 1024 | 8 | 4 |
| Weight AG | 3× AG(fsdp) 10.5 GB | **none** | **none** |
| Output psum | psum(tp) 224 MB | psum(fsdp) ~1.8 GB | psum(("fsdp","tp")) ~1.8 GB |
| TPS/chip | 560 | **1740** | 1541 |

### Why faster than v251

v251 used `P("ep", None, ("fsdp","tp"))` — D full (7168), F sharded by fsdp×tp jointly.
v264 uses `P("ep", "tp", "fsdp")` — D sharded by tp (3584), F sharded by fsdp.

Key difference: v264's activation enters the MoE shard_map as P("fsdp", "tp") — D already
sharded by TP. The contraction dimension D/tp=3584 matches between activation and weight
inside the shard_map, so no AllGather and no extra data movement on the reduction dim.

v251 had D full on device (7168) — more data per matmul, more bytes in the psum output.
v264's psum("fsdp") operates on (T_fsdp, D/tp) = (32768, 3584) tensors instead of
v251's psum(("fsdp","tp")) on (16384, 7168) — same total bytes but different decomposition.

The 13% improvement likely comes from:
1. Halved psum tensor width (3584 vs 7168) → better ICI utilization
2. psum("fsdp") is 256-way vs psum(("fsdp","tp")) 512-way → fewer reduction hops
3. GSPMD may overlap the MoE output AG(tp) with subsequent attention compute

### Shard_map specs (v264)

```python
# TP>1 path:
_act_x  = P("fsdp", "tp")        # flat_x: batch by fsdp, D by tp → (32768, 3584)
_act_iw = P("fsdp", None)        # indices/weights: batch by fsdp, K full → (32768, 8)
_wi     = P("ep", "tp", "fsdp")  # D by tp, D_moe by fsdp → (256, 3584, 8)
_wo     = P("ep", "fsdp", "tp")  # D_moe by fsdp, D by tp → (256, 8, 3584)

# Inside body: no AllGather, psum("fsdp") on output
# Output: P("fsdp", "tp") → (32768, 3584) → GSPMD AG(tp) before residual add
```

### MoE layer collectives (v264, per layer)

| Component | Collective | Volume |
|-----------|-----------|--------|
| Attn weights | 4× AG(fsdp) | 134 MB |
| Attn w_out | 1× AR(tp) | 448 MB |
| **MoE experts** | **1× psum(fsdp)** | **~1.8 GB** |
| MoE output AG(tp) | 1× AG(tp) (GSPMD) | ~224 MB |
| Shared expert weights | 2× AG(fsdp) | 56 MB |
| Shared expert wo | 1× AR(tp) | 448 MB |
| **MoE layer total** | | **~3.1 GB** (was 11.8 GB) |

---

## Key Observations

1. **v264 TP-on-D eliminates MoE weight AllGather**: 0 GB/layer vs 10.5 GB/layer.
   Result: 1740 TPS/chip (+13% over v251, +3.1× over v258 approach).

2. **Which axis shards the reduction dimension matters enormously**:
   FSDP on D (reduction) → forced 256-way AllGather = 609 GB/step.
   TP on D (reduction) → contraction matches activation → zero AllGather.

3. **TP AllReduce is cheap**: AR(tp) is only 2-way (intra-chip d2d on v7x).
   The GSPMD-inserted AG(tp) on MoE output is ~224 MB — negligible.

4. **Carry is still replicated on TP**: D not sharded in carry → 448 MB/layer.
   Total 26.7 GB. Future: D-sharded carry P("fsdp",None,"tp") would halve to 13.3 GB.

5. **EP=1 means no A2A**: All 256 experts on every device. No token dispatch needed.
   Simplifies communication but means each device does 256× expert compute.

6. **Remaining bottleneck is FSDP AllGather of attention weights**: 134 MB/layer × 61 ≈ 8 GB
   plus dense/shared MLP weights. This is now the dominant communication cost.
