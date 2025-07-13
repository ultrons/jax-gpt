import jax
import jax.numpy as jnp
from flax import nnx
import flax

from jax_gpt.models.gpt2.model import GPT, KVCache
from jax_gpt.models.gpt2.config import GPTConfig

def test_gpt_model_sanity():
    """
    Performs an end-to-end sanity check on the full GPT model.
    """
    print(f"--- Using Flax version: {flax.__version__} ---")
    print("--- Starting Full GPT Model Sanity Test ---")

    # --- Test Configuration ---
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)

    config = GPTConfig(
        d_model=64,  # Smaller model for faster testing
        d_head=16,
        d_ff=256,
        d_context=128,
        n_head=4,
        n_kv_head=4,
        n_layers=2, # Only 2 layers for speed
        use_bias=True,
        dropout=0.1
    )

    B, T, C = 2, 10, config.d_model # Batch, Time, Channels
    vocab_size = 100 # Define vocab_size for the test

    # --- 1. Model Initialization ---
    try:
        # Temporarily add vocab_size to the config for initialization
        config.vocab_size = vocab_size
        model = GPT(config, rngs=rngs)
        print("✅ GPT model initialized successfully.")
    except Exception as e:
        print(f"❌ FAILED: Initialization error: {e}")
        raise
    finally:
        # Clean up the added attribute
        delattr(config, 'vocab_size')

    # --- 2. Training Forward Pass ---
    print("\nTesting Training Forward Pass...")
    try:
        dummy_input = jnp.ones((B, T), dtype=jnp.int32)
        logits, cache = model(dummy_input, deterministic=True, cache=None)
        
        expected_shape = (B, T, vocab_size)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        print(f"✅ Training pass produced correct logit shape: {logits.shape}")
        
        assert cache is None, "Cache should be None in training mode"
        print("✅ Cache is correctly None in training mode.")

    except Exception as e:
        print(f"❌ FAILED: Training forward pass error: {e}")
        raise

    # --- 3. Inference Forward Pass (Single Step) ---
    print("\nTesting Single-Step Inference Pass...")
    try:
        # Initialize an empty cache
        cache_shape = (B, config.d_context, config.n_head, config.d_head)
        initial_keys = jnp.zeros((config.n_layers, *cache_shape), dtype=jnp.float32)
        initial_values = jnp.zeros((config.n_layers, *cache_shape), dtype=jnp.float32)
        initial_cache = [
            KVCache(key=initial_keys[i], value=initial_values[i], pos=0)
            for i in range(config.n_layers)
        ]

        single_token_input = dummy_input[:, :1]
        logits, updated_cache = model(single_token_input, deterministic=True, cache=initial_cache)

        expected_shape = (B, 1, vocab_size)
        assert logits.shape == expected_shape, f"Expected logit shape {expected_shape}, got {logits.shape}"
        print(f"✅ Single-token inference produced correct logit shape: {logits.shape}")

        assert updated_cache is not None, "Updated cache should not be None"
        assert updated_cache[0].pos == 1, f"Expected updated cache position to be 1, got {updated_cache[0].pos}"
        print("✅ Cache position was correctly updated.")

    except Exception as e:
        print(f"❌ FAILED: Single-step inference pass error: {e}")
        raise

    # --- 4. Full Generation Method ---
    print("\nTesting Full `generate` Method...")
    try:
        start_tokens = jnp.zeros((B, 1), dtype=jnp.int32)
        new_tokens_to_gen = 5
        
        generated_tokens = model.generate(start_tokens, new_tokens=new_tokens_to_gen, rngs=rngs)

        expected_length = start_tokens.shape[1] + new_tokens_to_gen
        assert generated_tokens.shape[1] == expected_length, f"Expected sequence length {expected_length}, got {generated_tokens.shape[1]}"
        print(f"✅ `generate` produced the correct number of new tokens.")

    except Exception as e:
        print(f"❌ FAILED: `generate` method error: {e}")
        raise

    print("\n--- ✅ Full GPT Model Sanity Test Passed Successfully! ---")


if __name__ == "__main__":
    test_gpt_model_sanity()
