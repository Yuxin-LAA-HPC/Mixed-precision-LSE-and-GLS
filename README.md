This code is for testing mixed precision iterative refinements for least
squares with linear equality constraints (LSE) and generalized least squares
problems (GLS), according to our paper:

    Mixed precision iterative refinement for least squares with linear equality
    constraints and generalized least squares problems. Bowen Gao, Yuxin Ma, and
    Meiyue Shao. (2024) arXiv:2406.16499

For the LSE problem, three iterative refinement (IR) based mixed precision
methods are used including GMRES-based IR with left preconditioning,
GMRES-based IR with two-sided preconditioning, and traditional IR.

For the GLS problem, two iterative refinement (IR) based mixed precision
methods are used including GMRES-based IR with two-sided preconditioning
and traditional IR.

SRC/:

util.c: Includes functions to generate test matrices, to print matrices, and needed by the other files.

gmres_lse.c: The GMRES algorithm with left preconditioner for LSE.

gmres_lse_twoside: The GMRES algorithm with left and right preconditioners for LSE.

gmres_gls_twoside: The GMRES algorithm with left and right preconditioners for GLS.

mplse.c: The traditional IR-based mixed precision LSE algorithm.

mplse_gmres.c: The GMRES-based mixed precision LSE algorithm, where GMRES employs left preconditioner.

mplse_gmres_twoside.c: The GMRES-based mixed precision LSE algorithm, where GMRES employs both left and right preconditioners.

mpgls.c: The traditional IR-based mixed precision GLS algorithm.

mpgls_gmres_twoside.c: The GMRES-based mixed precision GLS algorithm, where GMRES employs both left and right preconditioners.


TESTS/:

test_lse_gmres.c: Compare our IR-based mixed precision algorithms with the LAPACK LSE subroutine.

test_gls.c: Compare our IR-based mixed precision algorithms with the LAPACK GLS subroutine.
