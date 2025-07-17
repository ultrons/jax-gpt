import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import numpy as np
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

# Import your JAX GPT model and config from your jax_gpt package
# Assuming jax_gpt is installed in your environment or accessible via PYTHONPATH
from jax_gpt.models.gpt2.model import GPT
from jax_gpt.models.gpt2.config import GPTConfig

@register_model("jax_gpt_adapter")
class JaxGPTLM(TemplateLM):
    def __init__(self, model_path: str = "gpt2", batch_size: int = 1, device: str = "cpu", **kwargs):
        super().__init__()
        self.model_path = model_path
        self.batch_size = int(batch_size)
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Initialize your JAX model
        # Note: The 'from_pretrained' method in your GPT class handles loading weights
        self.model = GPT.from_pretrained(model_path)

    @property
    def eot_token_id(self):
        # End Of Text token ID for GPT-2
        return self.tokenizer.eot_token

    @property
    def max_length(self):
        # Max context length of the model
        return self.model.config.d_context

    @property
    def max_gen_toks(self):
        # Maximum number of tokens to generate in a single call
        return 256  # This can be adjusted based on typical generation needs

    def tok_encode(self, string: str):
        # Encode a string into tokens
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: list):
        # Decode tokens back into a string
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        for i in tqdm(range(0, len(requests), self.batch_size), disable=disable_tqdm):
            chunk = requests[i:i + self.batch_size]
            contexts, continuations = zip(*[(context, continuation) for (_, context, continuation) in chunk])

            # Pad contexts and continuations
            max_context_len = max(len(ctx) for ctx in contexts)
            max_continuation_len = max(len(cont) for cont in continuations)

            padded_contexts = np.array([np.pad(ctx, (max_context_len - len(ctx), 0), 'constant', constant_values=self.eot_token_id) for ctx in contexts])
            padded_continuations = np.array([np.pad(cont, (0, max_continuation_len - len(cont)), 'constant', constant_values=self.eot_token_id) for cont in continuations])

            # Create input tokens
            inp = jnp.array(np.concatenate([padded_contexts, padded_continuations], axis=1))

            # Get logits
            logits = self.model(inp, deterministic=True)[0]

            # Slice logits to only include continuation tokens
            continuation_logits = logits[:, max_context_len-1:-1, :]

            # Get log probabilities
            log_probs = jax.nn.log_softmax(continuation_logits, axis=-1)

            # Gather log probabilities for the continuation tokens
            for i, cont in enumerate(continuations):
                cont_log_probs = log_probs[i, :len(cont), :]
                gathered_log_probs = jnp.take_along_axis(cont_log_probs, jnp.array(cont)[:, None], axis=-1).squeeze(-1)

                # Check if the greedy choice would have been the same
                greedy_tokens = jnp.argmax(cont_log_probs, axis=-1)
                is_greedy = bool(jnp.all(greedy_tokens == jnp.array(cont)))

                res.append((float(jnp.sum(gathered_log_probs)), is_greedy))

        return res

    def generate_until(self, requests):
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()
