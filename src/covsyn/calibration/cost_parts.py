"""The one dict the objective writes its decomposition into.

This exists because of finding E71. `firefly_optimizer.py` is RUN as a script, so while it is
executing it is the module `__main__`. When `fast_cost` then does `import firefly_optimizer`,
Python has no record of that file being loaded (it is registered as `__main__`, not as
`firefly_optimizer`), so it imports and executes it a SECOND time under its real name. The two
copies have separate module globals, and every constant in them happens to be identical -- so the
fit is unaffected -- but a mutable shared dict is not: fast_cost filled
`firefly_optimizer.LAST_COST_PARTS` while the running `__main__` kept reading its own, which
stayed empty for the whole of run 7. The result was a progress_metrics.csv with 34 columns
instead of 73: every cost_* and measured_* diagnostic silently gone.

Keeping the dict in its own module fixes it, because `cost_parts` is only ever imported under
that name, so both copies of firefly_optimizer bind the same object.

Do not add anything else here, and do not rebind LAST -- mutate it in place (clear()/update()),
which is what both writers do.
"""

LAST = {}
