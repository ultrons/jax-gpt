import math
from flax import nnx
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from config import GPTConfig
# Phase 1: The Foundational Model

# Step 1.2: LayerNorm Module
# TODO: Implement the LayerNorm nnx.Module.
class LayerNorm(nnx.Module):
    def __init__(self, num_features: int, use_bias: bool, *, rngs: nnx.Rngs):
        super().__init__()
        self.weight = nnx.Param(
            jax.nn.initializers.ones(rngs(), (num_features, ))
        )
        self.bias = nnx.Param(
            jax.nn.initializers.zeros(rngs(), (num_features, ))
        ) if use_bias else None

    def __call__(self, x: jax.Array):
        eps = 1e-5
        mu = jnp.mean(x, axis=-1, keepdims=True)
        sigma_2 = jnp.mean(jnp.square(x -mu), axis=-1, keepdims=True)
        x_hat = (x - mu) / jnp.sqrt(sigma_2 + eps)
        return self.weight.value * x_hat + self.bias.value if self.bias else 0.0

# Step 1.3: MLP Block
# TODO: Implement the MLP nnx.Module.
class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        # Expansion layer weights
        self.c_fc = nnx.Linear(
            in_features=config.d_model,
            out_features=config.d_ff,
            use_bias=config.use_bias,
            rngs=rngs
        )
        # Contraction layer weights
        self.c_proj = nnx.Linear(
            in_features=config.d_ff,
            out_features=config.d_model,
            use_bias=config.use_bias,
            rngs=rngs
        )
        self.dropout = nnx.Dropout(config.dropout)

    def __call__(self, x: jax.Array, *, deterministic: bool):
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x, deterministic=deterministic)

# Step 1.4: Causal Self-Attention Module
# TODO: Implement the CausalSelfAttention nnx.Module.
# TODO: Ensure the __call__ method accepts an optional 'cache' argument for inference.
@dataclass
class KVCache:
    key: jax.Array
    value: jax.Array
    pos: int = 0

class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.c_attn = nnx.Linear(
            in_features=config.d_model,
            out_features=3 * config.d_model,
            use_bias=config.use_bias,
            rngs=rngs
        )
        self.c_proj = nnx.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            use_bias=config.use_bias,
            rngs=rngs
        )
        self.attn_dropout = nnx.Dropout(config.dropout)
        self.resid_dropout = nnx.Dropout(config.dropout)
        self.config = config

    def __call__(self, x: jax.Array, *, deterministic: bool, cache: KVCache | None = None):
        B, T, C = x.shape
        d_head = self.config.d_head
        n_head = self.config.n_head
        n_kv_head = self.config.n_kv_head
        assert n_head == n_kv_head, "Only Multihead Attention is supported"

        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        q = jnp.reshape(q, (B, T, n_head, d_head))
        k = jnp.reshape(k, (B, T, n_kv_head, d_head))
        v = jnp.reshape(v, (B, T, n_kv_head, d_head))

        if cache is not None:
            pos = cache.pos
            # Update the cache
            new_key = cache.key.at[:, pos:pos+1, :, : ].set(k)
            new_value = cache.value.at[:, pos:pos+1, :, : ].set(v)
            updated_cache = KVCache(key=new_key, value=new_value, pos=pos+1)
            k, v = new_key, new_value
        else:
            updated_cache = None

        # Attention score
        # 'bqhd,bkhd->bhqk'
        att = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        att = att * (1 / jnp.sqrt(d_head))
        # Apply causal mask only if when not using the cache
        if cache is None:
            causal_mask = jnp.tril(jnp.ones((T,T))).reshape(1, 1, T, T)
            att = jnp.where(causal_mask == 0, -jnp.inf, att)
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=deterministic)

        # Elementwise Multiplication with value projection
        #'bhqk,bkhd=>bqhd'
        y = jnp.einsum('bhqk,bkhd->bqhd', att, v)
        y = y.reshape(B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout(y, deterministic=deterministic)
        return y, updated_cache

class Block(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs:nnx.Rngs):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, config.use_bias, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = LayerNorm(config.d_model, config.use_bias, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True, cache: KVCache | None = None):
        attn_output , updated_cache = self.attn(self.ln_1(x), deterministic=deterministic, cache=cache)
        x = x + attn_output
        mlp_output = self.mlp(self.ln_2(x), deterministic=deterministic)
        x = x + mlp_output
        return x, updated_cache



# Putting it all together
class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.wpe = nnx.Embed(config.d_context, config.d_model, rngs=rngs)
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layers)]
        self.drop = nnx.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.d_model, config.use_bias, rngs=rngs)
        self._init_weights(rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool, cache: list[KVCache] | None = None):
        B, T = x.shape
        embed = self.wte(x)
        
        pos_idx = cache[0].pos if cache is not None else 0
        pos = self.wpe(jnp.arange(pos_idx, pos_idx + T))
        
        x = embed + pos
        x = self.drop(x, deterministic=deterministic)

        new_cache = []
        for i, block in enumerate(self.h):
            layer_cache = cache[i] if cache is not None else None
            x, updated_layer_cache = block(x, deterministic=deterministic, cache=layer_cache)
            if updated_layer_cache is not None:
                new_cache.append(updated_layer_cache)

        x = self.ln_f(x)
        logits = x @ self.wte.embedding.value.T
        
        return logits, new_cache if cache is not None else None

    def _init_weights(self, *, rngs: nnx.Rngs):
        initializer_fn = jax.nn.initializers.normal(stddev=0.02)
        self.wte.embedding.value = initializer_fn(
            rngs(),
            self.wte.embedding.value.shape,
            self.wte.embedding.value.dtype,
        )
        self.wpe.embedding.value = initializer_fn(
            rngs(),
            self.wpe.embedding.value.shape,
            self.wpe.embedding.value.dtype,
        )
        for path, module in self.iter_modules():
            if isinstance(module, nnx.Linear):
                module.kernel.value = initializer_fn(
                    rngs(),
                    module.kernel.value.shape,
                    module.kernel.value.dtype,
                )
                if module.bias is not None:
                    module.bias.value = jax.nn.initializers.zeros(
                        rngs(),
                        module.bias.value.shape,
                        module.bias.value.dtype
                    )
        for block in self.h:
            scaled_initializer = jax.nn.initializers.normal(
                stddev = 0.02 / math.sqrt(2 * self.config.n_layers)
            )
            attn_proj_kernel = block.attn.c_proj.kernel
            attn_proj_kernel.value = scaled_initializer(
                rngs(),
                attn_proj_kernel.value.shape,
                attn_proj_kernel.value.dtype,
            )
            mlp_proj_kernel = block.mlp.c_proj.kernel
            mlp_proj_kernel.value = scaled_initializer(
                rngs(),
                mlp_proj_kernel.value.shape,
                mlp_proj_kernel.value.dtype,
            )

    def generate(self, idx: jax.Array, new_tokens: int, temperature: float = 1.0, top_k: int | None = None, *, rngs: nnx.Rngs):
        #1. Initialize KVCache
        B = idx.shape[0]
        cache_shape = (B, self.config.d_context, self.config.n_head, self.config.d_head)
        initial_keys = jnp.zeros((self.config.n_layers, *cache_shape), dtype=jnp.float32)
        initial_values = jnp.zeros((self.config.n_layers, *cache_shape), dtype=jnp.float32)
        cache = [
            KVCache(key=initial_keys[i], value=initial_values[i], pos=0)
            for i in range(self.config.n_layers)
        ]
        #2. Generation Loop
        for _ in range(new_tokens):
            # --- A. Crop Context ---
            # The input to the model should only be the most recent token.
            # The context is handled by the KV cache.
            idx_cond = idx[:, -1:]

            # --- B. Forward Pass ---
            # Call the model with the single token and the cache.
            logits, cache = self(idx_cond, deterministic=True, cache=cache)
            # get the logit for the last token
            logits = logits[:, -1, :] # [B, C]
            # -- C. Apply temperature
            logits = logits / temperature
            # -- D. top_k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, k=top_k)
                # Create a mask to set the rest to -inf
                mask = jnp.full_like(logits, -jnp.inf).at[:, top_k_indices].set(top_k_logits)
                logits = mask

            # -- E. Sample new token
            key = rngs()
            idx_next = jax.random.categorical(key, logits, axis=-1)
            idx_next = idx_next[:, None] #[B, 1] for concatenation
            idx = jnp.concatenate([idx, idx_next], axis=1)

        return idx













# Step 1.5: Transformer Block
# TODO: Implement the Block nnx.Module, combining LayerNorm, Attention, and MLP.
# TODO: Ensure the __call__ method passes the 'cache' argument down.

# Step 1.6: Vectorized Blocks
# TODO: Use nnx.vmap to create a VmappedBlock that stacks the Block module.

# Step 1.7: GPT Module Assembly
# TODO: Implement the main GPT nnx.Module.

# Step 1.8: Sequential Forward Pass
# TODO: Implement the GPT.__call__ method using jax.lax.scan over the VmappedBlock.

# Step 1.9: Weight Initialization
# TODO: Implement the _init_weights private method for GPT-2 style initialization.

