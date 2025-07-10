# Project Plan: JAX-nanoGPT-vLLM

A step-by-step guide to building a GPT model in JAX/Flax NNX, structured for advanced distribution and efficient inference.

## Phase 1: The Foundational Model (`model.py`)

*Objective: Build a robust, self-contained `GPT` module that is aware of distribution and KV caching from day one, and verify its correctness against established benchmarks.*

| Step | Task | Description | Verification | Est. Time |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | **Configuration** | Create the `GPTConfig` dataclass to hold all model hyperparameters (`n_layer`, `n_head`, `n_embd`, etc.). | Instantiate the config object and access its attributes. | 15-30 mins |
| 1.2 | **LayerNorm Module** | Implement a `LayerNorm` module using `nnx.Module`. | Test its forward pass on a dummy tensor to ensure numerical stability. | 30-45 mins |
| 1.3 | **MLP Block** | Implement the `MLP` module (`Linear` -> `GELU` -> `Linear` -> `Dropout`). | Test its forward pass on a dummy tensor. | 30-45 mins |
| 1.4 | **Causal Self-Attention** | Implement `CausalSelfAttention`. Crucially, design its `__call__` to accept an optional `cache` argument. If `cache` is present, it should perform incremental decoding. If not, it computes the full attention mask for training. | Test both modes: with and without passing a cache, and check output shapes. | 2-3 hours |
| 1.5 | **Transformer Block** | Combine `LayerNorm` and `CausalSelfAttention` into a single `Block` module with residual connections. The `__call__` method must also pass the `cache` argument down to the attention layer. | Test its forward pass, checking for correct shapes and residual addition. | 30 mins |
| 1.6 | **Vectorized Blocks** | Use `nnx.vmap` to create a `VmappedBlock` that stacks `Block` `n_layer` times. This is the key step for enabling future distribution. | Instantiate it with `length=n_layer` and check the parameter shapes to confirm the new "layers" axis. | 1-2 hours |
| 1.7 | **GPT Module Assembly** | Create the main `GPT` module. In `__init__`, assemble the embeddings (`wte`, `wpe`), the `VmappedBlock`, and the final `LayerNorm`. | Instantiate the full model without errors. | 30 mins |
| 1.8 | **Sequential Forward Pass** | Implement the `GPT.__call__` method. Use `jax.lax.scan` to iterate over the `VmappedBlock`. This is the correct functional way to handle a sequential chain of layers with residual connections and is compatible with JAX transformations. | Test the full forward pass with a dummy input and verify the final output shape. | 2-3 hours |
| 1.9 | **Weight Initialization** | Add a private `_init_weights` method to the `GPT` class that correctly initializes all parameters according to GPT-2 conventions and call it from `__init__`. | Inspect parameter values after initialization to ensure they are not all zeros or ones. | 1 hour |
| 1.10| **Inference Method** | Add the `generate(self, idx, ...)` method to the `GPT` class. This method will initialize the KV cache and loop, calling the model's `__call__` method with the cache at each step. It must handle the JAX random key for sampling. | Generate a few tokens from an un-trained model to ensure the loop runs. | 1.5-2 hours |
| 1.11| **`from_pretrained` Method** | Implement a class method `from_pretrained(model_type)` that downloads GPT-2 weights from Hugging Face and loads them into the JAX model. This will involve mapping Hugging Face parameter names to the JAX model's parameter names. | Load weights for `gpt2` and run the `generate` method to produce coherent text, confirming the weights were loaded correctly. | 2-3 hours |
| 1.12| **Evaluation Harness** | Integrate a standard evaluation framework like `lm-evaluation-harness`. Create an adapter to make the JAX model compatible with the harness. Phase 1 is complete only when the score is satisfactory. | Successfully run the harness on a benchmark like Hellaswag and get a score comparable to the reference GPT-2 model. | 3-5 hours |

## Phase 2: Training Infrastructure (`train.py`, `utils.py`)

*Objective: Build the scripts and utilities required to train the model, following JAX best practices.*

| Step | Task | Description | Verification | Est. Time |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | **Configuration** | Add a `TrainConfig` dataclass to `config.py` for training hyperparameters (`learning_rate`, `batch_size`, etc.). | Instantiate the training config object. | 15 mins |
| 2.2 | **Data Loading** | Create a `get_batch` function in `train.py` to read tokenized data from a memory-mapped binary file. | The function should return batches of the correct shape from `train.bin` and `val.bin`. | 1 hour |
| 2.3 | **TrainState Definition** | In `utils.py`, define a `TrainState` dataclass to hold the `step`, `model`, `optimizer`, and `optimizer_state`. | Instantiate it with a model and a basic optimizer. | 30 mins |
| 2.4 | **Optimizer Definition** | In `utils.py`, create a `create_optimizer` function. This function will implement the logic to separate parameters for weight decay (kernels) from those without (biases, norms) and use `optax.multi_transform`. | The function should return a valid `optax` optimizer without errors. | 2-3 hours |
| 2.5 | **Training Step** | In `train.py`, create the `@jax.jit`-decorated `train_step` function. It will take the `TrainState` and a batch, define the loss function internally, and use `jax.value_and_grad` to get gradients and update the state. | Run a single `train_step` with a dummy batch and see the loss value and updated parameters. | 2-3 hours |
| 2.6 | **Evaluation Helper** | In `train.py`, create an `estimate_loss` function to calculate the average loss over multiple batches for the train and validation sets. | Run it on the un-trained model; it should return a loss value close to `-ln(1/vocab_size)`. | 1 hour |
| 2.7 | **Checkpointing** | In `train.py`, integrate `orbax.checkpoint` to save and restore the `TrainState`. This is crucial for long training runs. | Save a checkpoint and then load it back successfully, verifying the step number is correct. | 1-2 hours |
| 2.8 | **Main Training Loop** | In `train.py`, write the main script body that initializes everything and runs the training loop, calling `train_step`, `estimate_loss`, and saving checkpoints. | Run training for ~100 iterations and see the loss measurably decrease. | 1 hour |

## Phase 3: Distribution Strategies (Advanced)

*Objective: Extend the training script to support data and model parallelism on multi-device hardware (TPU/GPU).*

| Step | Task | Description | Verification | Est. Time |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | **Device Mesh** | In `utils.py`, create a function to define a `jax.sharding.Mesh` based on the available devices. | The mesh should correctly identify the device topology (e.g., a 1D array of 8 devices). | 30 mins |
| 3.2 | **Sharding Rules** | In `utils.py`, define sharding rules (`PartitionSpec`). Create a spec for data parallelism (replicating model, sharding data) and another for model parallelism (sharding the "layers" axis of the `VmappedBlock`). | The partition specs should be valid JAX objects. | 1-2 hours |
| 3.3 | **Distributed TrainState** | Modify the `TrainState` creation to apply these sharding rules to the initial model parameters and optimizer state using `jax.device_put`. | Inspect the `sharding` attribute of the JAX arrays in the state to confirm they are sharded. | 1-2 hours |
| 3.4 | **Distributed Training** | The `train_step` function should now correctly handle distributed inputs and outputs. JAX's `jit` will automatically use the sharding information from the `TrainState` to perform the distributed computation. | Run training on multiple devices and observe the speedup and device memory usage. | 1 hour |

## Phase 4: Application and vLLM-style Inference (Advanced)

*Objective: Use the trained model and implement a more advanced, high-throughput inference strategy.*

| Step | Task | Description | Verification | Est. Time |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | **Data Preparation** | Create `prepare.py` by adapting the script from nanoGPT to download and tokenize a dataset (e.g., Shakespeare) using `tiktoken`. | The script should produce `train.bin` and `val.bin` files. | 1 hour |
| 4.2 | **Basic Sampling** | Create `sample.py` to load a trained checkpoint and use the `model.generate` method to produce text. | Generate coherent-looking text from your trained model. | 1 hour |
| 4.3 | **PagedAttention (vLLM)** | **(Highly Advanced)** This is a research-level task. The goal is to replace the dense KV cache with a paged memory system. This involves creating a block manager, and a custom attention kernel (likely via JAX's pallas or custom calls) that can read from non-contiguous memory blocks based on a block table. | The new attention module should produce numerically identical results to the original attention but with significantly better memory usage for batched inference with varied sequence lengths. | 10-20+ hours |
| 4.4 | **Inference Server** | Create `server.py`. This will be a simple web server (e.g., using FastAPI) that exposes an endpoint for text generation. It will manage a queue of requests and use the `PagedAttention` model to serve them in batches for high throughput. | Send multiple concurrent requests to the server and receive correct, generated text. | 2-3 hours |

---

## Summary Table

| Phase | Objective | Total Estimated Time |
| :--- | :--- | :--- |
| **Phase 1** | Core Model Implementation | 14 - 23 hours |
| **Phase 2** | Training Infrastructure | 8.5 - 13.5 hours |
| **Phase 3** | Distribution Strategies | 3.5 - 5.5 hours |
| **Phase 4** | Application & vLLM-style Inference | 14 - 25+ hours |
| **Total** | **End-to-End Project** | **~40 - 67+ hours** |
