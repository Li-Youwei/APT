"""R1 (VA + selector on LIBERO) tooling. See exp/R1_va_jev_libero/spec.md.

Modules that run inside the LIBERO simulator env (eval_r1, diag_r1d,
convert_libero) must only import numpy / cv2 / h5py / LIBERO and
`shm_transport` -- never torch or `apt`.
"""
