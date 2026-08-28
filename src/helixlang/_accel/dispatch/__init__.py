"""VM opcode dispatch (doc/36 P0).

The P0 hot path shared by every cell in a population simulation.  A native
dispatching loop (C/Cython, built under the ``[native]`` extra) replaces
the pure-Python interpreter on the same bytecode buffer and operand stack,
consuming a bounded ``quota`` of ops per call so ticks/checkpoints stay
deterministic.

The canonical implementation lives in :mod:`helixlang.core.vm`; this package
provides the isolated, stack-agnostic dispatch kernel plus the selection hooks
(:func:`backend.step` does an alias through :mod:`~helixlang._accel._loaders`).
Binary format compatibility with the compiled :class:`~helixlang.core.bytecode.Chunk`
is what keeps ``impl_cext``/``impl_cython`` interchangeable.
"""
