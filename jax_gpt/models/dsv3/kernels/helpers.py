"""Patch jax/_src/pallas/helpers.py — attempt 3: drop the inner @api.jit
on the wrapper. Hypothesis: with the wrapper inlined into the caller's trace,
ref operations sit inside the same scan-body jaxpr as everything else, and
the standard transpose path (which special-cases ref_p / empty_ref_p in
ad.py) handles them correctly."""
p = '/usr/local/lib/python3.12/site-packages/jax/_src/pallas/helpers.py'
src = open(p).read()
old = '''  @api.jit
  def wrapper(*operands):'''
new = '''  # PATCH: drop @api.jit so the wrapper is inlined into the caller's trace
  # def wrapper(*operands):
  def wrapper(*operands):'''
assert old in src, f'patch target not found in {p}'
open(p, 'w').write(src.replace(old, new))
print('patched ok (attempt 3 — drop @api.jit on wrapper)')
