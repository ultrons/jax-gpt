import math
from flax import nnx
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from .config import GPTConfig
import torch
from transformers import GPT2LMHeadModel
# Phase 1: The Foundational Model

# Step 1.2: LayerNorm Module
# TODO: Implement the LayerNorm nnx.Module.
class LayerNorm(nnx.Module):
    def __init__(self, num_features: int, use_bias: bool, *, rngs: nnx.Rngs):
        super().__init__()
        self.layer_norm = nnx.LayerNorm(num_features, use_bias=use_bias, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array):
        return self.layer_norm(x)

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

        # --- Start: Corrected Attention Logic (matches nanoGPT) ---
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        q = q.reshape(B, T, n_head, d_head).transpose(0, 2, 1, 3) # (B, n_head, T, d_head)
        k = k.reshape(B, T, n_kv_head, d_head).transpose(0, 2, 1, 3) # (B, n_head, T, d_head)
        v = v.reshape(B, T, n_kv_head, d_head).transpose(0, 2, 1, 3) # (B, n_head, T, d_head)

        if cache is not None:
            # The cache logic needs to be updated to match the new tensor shapes
            pos = cache.pos
            # The cache shape is (B, n_head, d_context, d_head)
            new_key = cache.key.at[:, :, pos:pos+T, :].set(k)
            new_value = cache.value.at[:, :, pos:pos+T, :].set(v)
            updated_cache = KVCache(key=new_key, value=new_value, pos=pos+T)
            k, v = new_key, new_value
        else:
            updated_cache = None

        # Causal self-attention
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(d_head))
        if cache is None: # Apply causal mask only during training/prompt processing
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_)).reshape(1, 1, T, T)
            att = jnp.where(causal_mask == 0, -jnp.inf, att)

        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=deterministic)
        y = att @ v # (B, n_head, T, T) x (B, n_head, T, d_head) -> (B, n_head, T, d_head)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=deterministic)
        # --- End: Corrected Attention Logic ---

        # # --- Start: Old Incorrect Attention Logic ---
        # q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        # q = jnp.reshape(q, (B, T, n_head, d_head))
        # k = jnp.reshape(k, (B, T, n_kv_head, d_head))
        # v = jnp.reshape(v, (B, T, n_kv_head, d_head))

        # if cache is not None:
        #     pos = cache.pos
        #     # Update the cache
        #     new_key = cache.key.at[:, pos:pos+T, :, : ].set(k)
        #     new_value = cache.value.at[:, pos:pos+T, :, : ].set(v)
        #     updated_cache = KVCache(key=new_key, value=new_value, pos=pos+T)
        #     k, v = new_key, new_value
        # else:
        #     updated_cache = None

        # # Attention score
        # # 'bqhd,bkhd->bhqk'
        # att = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        # att = att * (1 / jnp.sqrt(d_head))
        # # Apply causal mask only if when not using the cache
        # if cache is None:
        #     causal_mask = jnp.tril(jnp.ones((T,T))).reshape(1, 1, T, T)
        #     att = jnp.where(causal_mask == 0, -jnp.inf, att)
        # att = jax.nn.softmax(att, axis=-1)
        # att = self.attn_dropout(att, deterministic=deterministic)

        # # Elementwise Multiplication with value projection
        # #'bhqk,bkhd=>bqhd'
        # y = jnp.einsum('bhqk,bkhd->bqhd', att, v)
        # y = y.reshape(B, T, C)

        # y = self.c_proj(y)
        # y = self.resid_dropout(y, deterministic=deterministic)
        # # --- End: Old Incorrect Attention Logic ---
        return y, updated_cache

class Block(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs:nnx.Rngs):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, config.use_bias, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = LayerNorm(config.d_model, config.use_bias, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True, cache: KVCache | None = None, block_idx: int, return_activations: bool = False):
        activations = {}
        ln1_output = self.ln_1(x)
        if return_activations:
            activations[f"block_{block_idx}_ln1_output"] = ln1_output
        attn_output , updated_cache = self.attn(ln1_output, deterministic=deterministic, cache=cache)
        if return_activations:
            activations[f"block_{block_idx}_attn_output_pre_residual"] = attn_output
        x = x + attn_output
        mlp_input_for_ln2 = self.ln_2(x)
        mlp_output = self.mlp(mlp_input_for_ln2, deterministic=deterministic)
        if return_activations:
            activations[f"block_{block_idx}_mlp_output_pre_residual"] = mlp_output
        x = x + mlp_output
        
        if return_activations:
            return x, updated_cache, activations
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
        self.lm_head = nnx.Linear(config.d_model, config.vocab_size, use_bias=False, rngs=rngs)
        # Weight tying
        #self.lm_head.kernel.value = self.wte.embedding.value.T
        self._init_weights(rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool, cache: list[KVCache] | None = None, return_activations: bool = False):
        B, T = x.shape
        tok_emb = self.wte(x) # token embeddings of shape (B, T, C)

        pos_idx = cache[0].pos if cache is not None else 0
        pos_emb = self.wpe(jnp.arange(pos_idx, pos_idx + T)) # position embeddings of shape (T, C)

        x = tok_emb + pos_emb
        x = self.drop(x, deterministic=True)

        all_activations = {}
        if return_activations:
            all_activations["initial_embedding"] = x

        new_cache = []
        for i, block in enumerate(self.h):
            layer_cache = cache[i] if cache is not None else None
            if return_activations:
                x, updated_layer_cache, block_activations = block(x, deterministic=deterministic, cache=layer_cache, block_idx=i, return_activations=True)
                all_activations.update(block_activations)
            else:
                x, updated_layer_cache = block(x, deterministic=deterministic, cache=layer_cache, block_idx=i)
            if updated_layer_cache is not None:
                new_cache.append(updated_layer_cache)

        x = self.ln_f(x)
        if return_activations:
            all_activations["final_layernorm"] = x

        logits = x @ self.wte.embedding.value.T
        if return_activations:
            all_activations["final_logits"] = logits

        if return_activations:
            return logits, new_cache if cache is not None else None, all_activations
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
            elif isinstance(module, LayerNorm): # Custom LayerNorm wrapper
                module.layer_norm.scale.value = jax.nn.initializers.ones(
                    rngs(),
                    module.layer_norm.scale.value.shape,
                    module.layer_norm.scale.value.dtype,
                )
                if module.layer_norm.bias is not None:
                    module.layer_norm.bias.value = jax.nn.initializers.zeros(
                        rngs(),
                        module.layer_norm.bias.value.shape,
                        module.layer_norm.bias.value.dtype
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
        B = idx.shape[0]
        for _ in range(new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.shape[1] <= self.config.d_context else idx[:, -self.config.d_context:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, deterministic=True)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = jax.lax.top_k(logits, k=jnp.minimum(top_k, logits.shape[-1]))
                logits = jnp.where(logits < v[:, [-1]], -jnp.inf, logits)
            # sample from the distribution
            key = rngs()
            idx_next = jax.random.categorical(key, logits, axis=-1)
            # append sampled index to the running sequence and continue
            idx = jnp.concatenate([idx, idx_next[:, None]], axis=1)

        return idx
    @classmethod
    def from_pretrained(cls, model_type: str, override_args: dict | None = None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}

        # 1. Create the model with the correct config
        config = GPTConfig.from_model_type(model_type, **override_args)
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key, params=key, dropout=key)
        model = cls(config, rngs=rngs)

        # 2. Load the Hugging Face model and its state dictionary
        print(f"Loading weights from Hugging Face model: {model_type}")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 3. Copy weights, transposing where necessary
        # Assert that the lm_head weights are the transpose of the wte weights
        wte_hf = jnp.asarray(sd_hf['transformer.wte.weight'])
        lm_head_hf = jnp.asarray(sd_hf['lm_head.weight'])
        assert jnp.array_equal(wte_hf, lm_head_hf), "Mismatch between wte and lm_head weights"

        # Embeddings
        model.wte.embedding.value = jnp.asarray(wte_hf)
        model.wpe.embedding.value = jnp.asarray(sd_hf['transformer.wpe.weight'])

        for i in range(config.n_layers):
            model.h[i].ln_1.layer_norm.scale.value = jnp.asarray(sd_hf[f'transformer.h.{i}.ln_1.weight'])
            model.h[i].ln_1.layer_norm.bias.value = jnp.asarray(sd_hf[f'transformer.h.{i}.ln_1.bias'])
            model.h[i].attn.c_attn.kernel.value = jnp.asarray(sd_hf[f'transformer.h.{i}.attn.c_attn.weight'])
            model.h[i].attn.c_attn.bias.value = jnp.asarray(sd_hf[f'transformer.h.{i}.attn.c_attn.bias'])
            model.h[i].attn.c_proj.kernel.value = jnp.asarray(sd_hf[f'transformer.h.{i}.attn.c_proj.weight'])
            model.h[i].attn.c_proj.bias.value = jnp.asarray(sd_hf[f'transformer.h.{i}.attn.c_proj.bias'])
            model.h[i].ln_2.layer_norm.scale.value = jnp.asarray(sd_hf[f'transformer.h.{i}.ln_2.weight'])
            model.h[i].ln_2.layer_norm.bias.value = jnp.asarray(sd_hf[f'transformer.h.{i}.ln_2.bias'])
            model.h[i].mlp.c_fc.kernel.value = jnp.asarray(sd_hf[f'transformer.h.{i}.mlp.c_fc.weight'])
            model.h[i].mlp.c_fc.bias.value = jnp.asarray(sd_hf[f'transformer.h.{i}.mlp.c_fc.bias'])
            model.h[i].mlp.c_proj.kernel.value = jnp.asarray(sd_hf[f'transformer.h.{i}.mlp.c_proj.weight'])
            model.h[i].mlp.c_proj.bias.value = jnp.asarray(sd_hf[f'transformer.h.{i}.mlp.c_proj.bias'])

        model.ln_f.layer_norm.scale.value = jnp.asarray(sd_hf['transformer.ln_f.weight'])
        model.ln_f.layer_norm.bias.value = jnp.asarray(sd_hf['transformer.ln_f.bias'])

        # The lm_head weights are tied to the wte weights in the reference implementation.
        # We load the lm_head weights into our wte embedding matrix.
        #model.wte.embedding.value = jnp.asarray(sd_hf['lm_head.weight'])

        # The lm_head weights are tied to the wte weights in the reference implementation
        # We handle this in the __init__ and __call__ methods

        print("Weights loaded successfully.")
        return model

    def compare_activations(self, batch_size: int = 1, sequence_length: int = 128, seed: int = 42):
        import torch
        from transformers import GPT2LMHeadModel, GPT2Config as HF_GPT2Config

        print("--- Comparing Activations (JAX vs. PyTorch) ---")

        # Set default precision to float32 for both JAX and PyTorch
        jax.config.update("jax_enable_x64", False)
        torch.set_default_dtype(torch.float32)

        # 1. Create default GPTConfig
        config = GPTConfig()
        print(f"Using GPTConfig: {config}")

        # 2. Instantiate JAX GPT model
        key = jax.random.key(seed)
        rngs = nnx.Rngs(default=key, params=key, dropout=key)
        jax_model = GPT(config, rngs=rngs)
        print("JAX model instantiated.")

        # 3. Instantiate PyTorch GPT2LMHeadModel
        hf_config = HF_GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.d_context,
            n_embd=config.d_model,
            n_layer=config.n_layers,
            n_head=config.n_head,
            n_inner=config.d_ff,
            activation_function="gelu_new",
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_token=None,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
        )
        torch_model = GPT2LMHeadModel(hf_config)
        torch_model.eval() # Set to evaluation mode
        print("PyTorch model instantiated.")

        # 4. Generate random input IDs
        jax_input_ids = jax.random.randint(key, (batch_size, sequence_length), 0, config.vocab_size)
        torch_input_ids = torch.tensor(jax_input_ids.tolist(), dtype=torch.long)
        print(f"Generated random input IDs with shape: {jax_input_ids.shape}")

        # 5. Run forward pass and collect activations
        print("\n--- Running JAX Model ---")
        jax_logits, _, jax_activations = jax_model(jax_input_ids, deterministic=True, return_activations=True)

        print("\n--- Running PyTorch Model ---")
        # To get activations from PyTorch, we need to register hooks
        torch_activations = {}

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple): # Handle cases where output is a tuple (e.g., attention)
                    torch_activations[name] = output[0].detach().numpy()
                else:
                    torch_activations[name] = output.detach().numpy()
            return hook

        # Register hooks for PyTorch model
        # Embeddings
        torch_model.transformer.wte.register_forward_hook(get_activation("initial_embedding"))

        # Blocks
        for i, block in enumerate(torch_model.transformer.h):
            block.ln_1.register_forward_hook(get_activation(f"block_{i}_ln1_output"))
            block.attn.register_forward_hook(get_activation(f"block_{i}_attn_output_pre_residual"))
            block.mlp.register_forward_hook(get_activation(f"block_{i}_mlp_output_pre_residual"))

        # Final LayerNorm
        torch_model.transformer.ln_f.register_forward_hook(get_activation("final_layernorm"))

        # Logits
        torch_model.lm_head.register_forward_hook(get_activation("final_logits"))

        with torch.no_grad():
            torch_logits = torch_model(torch_input_ids).logits.numpy()

        # 6. Compare and print activation details
        print("\n--- Activation Comparison ---")
        print(f"{'Activation Name':<35} | {'JAX (Mean, Std, Shape)':<45} | {'PyTorch (Mean, Std, Shape)':<45}")
        print("-" * 128)

        # Compare initial embedding
        jax_emb = jax_activations["initial_embedding"]
        torch_emb = torch_activations["initial_embedding"]
        print(f"{'initial_embedding':<35} | {jax_emb.mean():.4f}, {jax_emb.std():.4f}, {jax_emb.shape} | {torch_emb.mean():.4f}, {torch_emb.std():.4f}, {torch_emb.shape}")

        # Compare block activations
        for i in range(config.n_layers):
            # LayerNorm1
            jax_ln1 = jax_activations[f"block_{i}_ln1_output"]
            torch_ln1 = torch_activations[f"block_{i}_ln1_output"]
            print(f"{f'block_{i}_ln1_output':<35} | {jax_ln1.mean():.4f}, {jax_ln1.std():.4f}, {jax_ln1.shape} | {torch_ln1.mean():.4f}, {torch_ln1.std():.4f}, {torch_ln1.shape}")

            # Attention output (pre-residual)
            jax_attn = jax_activations[f"block_{i}_attn_output_pre_residual"]
            torch_attn = torch_activations[f"block_{i}_attn_output_pre_residual"]
            print(f"{f'block_{i}_attn_output_pre_residual':<35} | {jax_attn.mean():.4f}, {jax_attn.std():.4f}, {jax_attn.shape} | {torch_attn.mean():.4f}, {torch_attn.std():.4f}, {torch_attn.shape}")

            # MLP output (pre-residual)
            jax_mlp = jax_activations[f"block_{i}_mlp_output_pre_residual"]
            torch_mlp = torch_activations[f"block_{i}_mlp_output_pre_residual"]
            print(f"{f'block_{i}_mlp_output_pre_residual':<35} | {jax_mlp.mean():.4f}, {jax_mlp.std():.4f}, {jax_mlp.shape} | {torch_mlp.mean():.4f}, {torch_mlp.std():.4f}, {torch_mlp.shape}")

        # Compare final LayerNorm
        jax_final_ln = jax_activations["final_layernorm"]
        torch_final_ln = torch_activations["final_layernorm"]
        print(f"{'final_layernorm':<35} | {jax_final_ln.mean():.4f}, {jax_final_ln.std():.4f}, {jax_final_ln.shape} | {torch_final_ln.mean():.4f}, {torch_final_ln.std():.4f}, {torch_final_ln.shape}")

        # Compare final logits
        print(f"{'final_logits':<35} | {jax_logits.mean():.4f}, {jax_logits.std():.4f}, {jax_logits.shape} | {torch_logits.mean():.4f}, {torch_logits.std():.4f}, {torch_logits.shape}")

        print("\n--- Comparison Complete ---")

