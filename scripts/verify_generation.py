import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
from jax_gpt.models.gpt2.model import GPT
from jax_gpt.models.gpt2.config import GPTConfig

# 1. Load the pre-trained model
print("Loading pre-trained GPT-2 model...")
# Ensure the from_pretrained method is called with the model type
model = GPT.from_pretrained('gpt2')
print("Model loaded.")

# 2. Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")

# 3. Encode a prompt
#prompt_text = "The answer to the ultimate question of life, the universe, and everything is"
prompt_text = "The answer to the ultimate question of life, what is a language model?"
prompt_tokens = enc.encode_ordinary(prompt_text)
print(f"Prompt tokens: {prompt_tokens}")

# 4. Format the input tensor
input_idx = jnp.array([prompt_tokens])

# 5. Set up for generation
rng_key = jax.random.key(4)
rngs = nnx.Rngs(default=rng_key)

# 6. Generate text
print("Generating text...")
output_tokens = model.generate(
    input_idx,
    new_tokens=50,
    temperature=1.0,
    rngs=rngs
)
print(f"Output Tokens: {output_tokens}")

# 7. Decode and print the output
generated_text = enc.decode(output_tokens[0].tolist())

print("\n--- Generated Text ---")
print(generated_text)

# 8. Compare activations
print("\n--- Running Activation Comparison ---")
model.compare_activations(batch_size=1, sequence_length=128, seed=42)
