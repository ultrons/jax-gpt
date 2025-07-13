import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass

from jax_gpt.models.gpt2.model import Block, KVCache
from jax_gpt.models.gpt2.config import GPTConfig

def test_block_forward_pass():
    """
    Tests the forward pass of the Block module in both training and inference modes.
    """
    print("--- Starting Block Test ---")

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
        block = Block(config, rngs=rngs)
        print("✅ Block initialized successfully.")
    except Exception as e:
        print(f"❌ FAILED: Initialization error: {e}")
        return

    # --- Training Forward Pass (no cache) ---
    print("\nTesting Training Forward Pass (no cache)...")
    try:
        output, updated_cache = block(dummy_input, deterministic=True, cache=None)
        
        expected_shape = (B, T, C)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"✅ Training pass produced correct output shape: {output.shape}")
        
        assert updated_cache is None, "Cache should be None in training mode"
        print("✅ Cache is correctly None in training mode.")

    except Exception as e:
        print(f"❌ FAILED: Training forward pass error: {e}")
        return

    # --- Inference Forward Pass (with cache) ---
    print("\nTesting Inference Forward Pass (with cache)...")
    try:
        # Initialize an empty cache
        key_cache = jnp.zeros((B, config.d_context, config.n_head, config.d_head))
        value_cache = jnp.zeros((B, config.d_context, config.n_head, config.d_head))
        kv_cache = KVCache(key=key_cache, value=value_cache, pos=0)

        # Simulate one step of inference
        single_token_input = dummy_input[:, :1, :] # Input is just one token
        output, updated_cache = block(single_token_input, deterministic=True, cache=kv_cache)

        expected_shape = (B, 1, C)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"✅ Single-token inference produced correct output shape: {output.shape}")

        assert updated_cache is not None, "Updated cache should not be None"
        assert updated_cache.pos == 1, f"Expected updated cache position to be 1, got {updated_cache.pos}"
        print("✅ Cache position was correctly updated.")

    except Exception as e:
        print(f"❌ FAILED: Inference forward pass error: {e}")
        return

    print("\n--- ✅ Block Test Passed Successfully! ---")


if __name__ == "__main__":
    test_block_forward_pass()