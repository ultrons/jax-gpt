import jax
import jax.numpy as jnp
from flax import nnx
import sys
import os
from dataclasses import dataclass

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gpt2 import CausalSelfAttention
from models.config import GPTConfig

# Define KVCache here for the test, as it's defined inside gpt2.py
@dataclass
class KVCache:
    key: jax.Array
    value: jax.Array
    pos: int = 0

def test_inference_forward_pass():
    """
    Tests only the Inference Forward Pass of the CausalSelfAttention module.
    """
    print("--- Starting CausalSelfAttention Inference Test ---")

    # --- Test Configuration ---
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)

    config = GPTConfig(
        d_model=128,
        d_head=32,
        d_ff=512,
        d_context=256,
        n_head=4,
        n_kv_head=4,
        n_layer=2,
        use_bias=True,
        dropout=0.1
    )

    B, T, C = 2, 16, config.d_model # Batch, Time, Channels
    dummy_input = jnp.ones((B, T, C))

    # --- Module Initialization ---
    try:
        attention = CausalSelfAttention(config, rngs=rngs)
        print("✅ CausalSelfAttention initialized successfully.")
    except Exception as e:
        print(f"❌ FAILED: Initialization error: {e}")
        return

    # --- Inference Forward Pass ---
    print("\nTesting Inference Forward Pass (with cache)...")
    try:
        # Initialize an empty cache for a single layer
        key_cache = jnp.zeros((B, config.d_context, config.n_head , config.d_head))
        value_cache = jnp.zeros((B, config.d_context, config.n_head , config.d_head))
        kv_cache = KVCache(key=key_cache, value=value_cache, pos=0)

        # Simulate one step of inference
        single_token_input = dummy_input[:, :1, :] # Input is just one token
        output, updated_cache = attention(single_token_input, deterministic=True, cache=kv_cache)

        expected_shape = (B, 1, C)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"✅ Single-token inference produced correct output shape: {output.shape}")

        assert updated_cache is not None, "Updated cache should not be None"
        assert updated_cache.pos == 1, f"Expected updated cache position to be 1, got {updated_cache.pos}"
        print("✅ Cache position was correctly updated.")

    except Exception as e:
        print(f"❌ FAILED: Inference forward pass error: {e}")
        return

    print("\n--- ✅ CausalSelfAttention Inference Test Passed Successfully! ---")


if __name__ == "__main__":
    test_inference_forward_pass()

