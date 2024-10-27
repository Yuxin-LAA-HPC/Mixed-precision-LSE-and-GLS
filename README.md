This code is for testing mixed precision iterative refinements for least
squares with linear equality constraints (LSE) and generalized least squares
problems (GLS), according to our paper:

Mixed precision iterative refinement for least squares with linear equality
constraints and generalized least squares problems. Bowen Gao, Yuxin Ma, and
Meiyue Shao. (2024) arXiv:2406.16499

For the LSE problem, three iterative refinement (IR) based mixed precision
methods are used including GMRES-based IR with left preconditioning,
GMRES-based IR with two-sided preconditioning, and traditional IR.
These three methods are compared with the LAPACK LSE subroutine in the
TESTS/test\_lse\_gmres.c file.

For the GLS problem, two iterative refinement (IR) based mixed precision
methods are used including GMRES-based IR with two-sided preconditioning
and traditional IR.
These methods are compared with the LAPACK GLS subroutine in the
TESTS/test\_gls.c file.

SRC:


TESTS:


