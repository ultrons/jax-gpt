# Phase 1: The Foundational Model

# Step 1.2: LayerNorm Module
# TODO: Implement the LayerNorm nnx.Module.

# Step 1.3: MLP Block
# TODO: Implement the MLP nnx.Module.

# Step 1.4: Causal Self-Attention Module
# TODO: Implement the CausalSelfAttention nnx.Module.
# TODO: Ensure the __call__ method accepts an optional 'cache' argument for inference.

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

# Step 1.10: Inference Method
# TODO: Implement the generate() method for autoregressive text generation.
