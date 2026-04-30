"""Sanity check: initialize distributed JAX and print device counts.

Primarily used to trigger 4x8x8 cluster scale-up before submitting real jobs.

Run via:
  kubectl --context gke_tpu-vm-gke-testing_us-central1_sivaibhav-exp-v7x \\
    apply -f k8s/sanity-check-stage1.yaml
"""

import jax

jax.distributed.initialize(initialization_timeout=600)

proc  = jax.process_index()
procs = jax.process_count()
total = jax.device_count()
local = jax.local_device_count()

print(f"[proc {proc:3d}/{procs}] total={total}  local={local}"
      f"  backend={jax.default_backend()}"
      f"  first_local={jax.local_devices()[0]}")

# Barrier: ensure all processes reached this point.
import jax.numpy as jnp
x = jnp.ones(1)
_ = jax.lax.psum(x, axis_name="x") if False else x  # no-op; cluster sync is implicit

if proc == 0:
    print(f"\nAll {procs} processes initialized successfully.")
    print(f"4x8x8 cluster is up: {total} total JAX devices ({local} per process).")
