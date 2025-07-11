from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("The answer to the ultimate question of life, what is a language model?", max_length=30, num_return_sequences=5)
