import functools
import time
from typing import Optional, Callable

import torch

from falkon.options import ConjugateGradientOptions, FalkonOptions
from falkon.mmv_ops.fmmv_incore import incore_fdmmv, incore_fmmv
from falkon.utils.tensor_helpers import copy_same_stride, create_same_stride
from falkon.utils import TicToc

# More readable 'pseudocode' for conjugate gradient.
# function [x] = conjgrad(A, b, x)
#     r = b - A * x;
#     p = r;
#     rsold = r' * r;
#
#     for i = 1:length(b)
#         Ap = A * p;
#         alpha = rsold / (p' * Ap);
#         x = x + alpha * p;
#         r = r - alpha * Ap;
#         rsnew = r' * r;
#         if sqrt(rsnew) < 1e-10
#               break;
#         end
#         p = r + (rsnew / rsold) * p;
#         rsold = rsnew;
#     end
# end


class Optimizer(object):
    def __init__(self):
        pass


class ConjugateGradient(Optimizer):
    def __init__(self, opt: Optional[ConjugateGradientOptions] = None):
        super().__init__()
        self.params = opt or ConjugateGradientOptions()

    def solve(self,
              X0: Optional[torch.Tensor],
              B: torch.Tensor,
              mmv: Callable[[torch.Tensor], torch.Tensor],
              max_iter: int,
              callback: Optional[Callable[[int, torch.Tensor, float], None]] = None) -> torch.Tensor:
        """Conjugate-gradient solver with optional support for preconditioning via generic MMV.

        This solver can be used for iterative solution of linear systems of the form $AX = B$ with
        respect to the `X` variable. Knowledge of `A` is only needed through matrix-vector
        multiplications with temporary solutions (must be provided through the `mmv` function).

        Preconditioning can be achieved by incorporating the preconditioner matrix in the `mmv`
        function.

        Parameters
        ----------
        X0 : Optional[torch.Tensor]
            Initial solution for the solver. If not provided it will be a zero-tensor.
        B : torch.Tensor
            Right-hand-side of the linear system to be solved.
        mmv
            User-provided function to perform matrix-vector multiplications with the design matrix
            `A`. The function must accept a single argument (the vector to be multiplied), and
            return the result of the matrix-vector multiplication.
        max_iter : int
            Maximum number of iterations the solver will perform. Early stopping is implemented
            via the options passed in the constructor of this class (in particular look at
            `cg_tolerance` options)
            i + 1, X, e_train
        callback
            An optional, user-provided function which shall be called at the end of each iteration
            with the current solution. The arguments to the function are the iteration number,
            a tensor containing the current solution, and the total time elapsed from the beginning
            of training (note that this time explicitly excludes any time taken by the callback
            itself).
        Returns
        -------
        The solution to the linear system `X`.
        """
        t_start = time.time()

        if X0 is None:
            R = copy_same_stride(B)
            X = create_same_stride(B.size(), B, B.dtype, B.device)
            X.fill_(0.0)
        else:
            R = B - mmv(X0)
            X = X0

        m_eps = self.params.cg_epsilon(X.dtype)

        P = R
        # noinspection PyArgumentList
        Rsold = torch.sum(R.pow(2), dim=0)

        e_train = time.time() - t_start

        for i in range(max_iter):
            with TicToc("Chol Iter", debug=False):  # TODO: FIXME
                t_start = time.time()
                AP = mmv(P)
                # noinspection PyArgumentList
                alpha = Rsold / (torch.sum(P * AP, dim=0) + m_eps)
                X.addmm_(P, torch.diag(alpha))

                if (i + 1) % self.params.cg_full_gradient_every == 0:
                    if X.is_cuda:
                        # addmm_ may not be finished yet causing mmv to get stale inputs.
                        torch.cuda.synchronize()
                    R = B - mmv(X)
                else:
                    R = R - torch.mm(AP, torch.diag(alpha))
                    # R.addmm_(mat1=AP, mat2=torch.diag(alpha), alpha=-1.0)

                # noinspection PyArgumentList
                Rsnew = torch.sum(R.pow(2), dim=0)
                if Rsnew.abs().max().sqrt() < self.params.cg_tolerance:
                    #print("Stopping conjugate gradient descent at "
                    #      "iteration %d. Solution has converged." % (i + 1))
                    break

                P = R + torch.mm(P, torch.diag(Rsnew / (Rsold + m_eps)))
                if P.is_cuda:
                    # P must be synced so that it's correct for mmv in next iter.
                    torch.cuda.synchronize()
                Rsold = Rsnew

                e_iter = time.time() - t_start
                e_train += e_iter
            with TicToc("Chol callback", debug=False):
                if callback is not None:
                    callback(i + 1, X, e_train)

        return X


class FalkonConjugateGradient(Optimizer):
    def __init__(self, kernel, preconditioner, opt: FalkonOptions, N=None):
        super().__init__()
        self.kernel = kernel
        self.preconditioner = preconditioner
        self.params = opt
        self.optimizer = ConjugateGradient(opt.get_conjgrad_options())
        self.N = N

    def flk_mmv(self,
                Knm: Optional[torch.Tensor],
                X: torch.Tensor,
                M: torch.Tensor,
                s1: Optional[torch.cuda.Stream],
                N: int,
                _lambda: float,
                sol: torch.Tensor):
        prec = self.preconditioner
        with TicToc("MMV", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)

            if Knm is not None:
                cc = incore_fdmmv(Knm, v_t, None, opt=self.params)
            else:
                cc = self.kernel.dmmv(X, M, v_t, None, opt=self.params)

            if X.is_cuda:
                with torch.cuda.stream(s1), torch.cuda.device(X.device):
                    # We must sync before calls to prec.inv* which use a different stream
                    cc_ = cc.div_(N)
                    v_ = v.mul_(_lambda)
                    s1.synchronize()
                    cc_ = prec.invTt(cc_).add_(v_)
                    s1.synchronize()
                    return prec.invAt(cc_)
            else:
                return prec.invAt(prec.invTt(cc / N) + _lambda * v)

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter, callback=None):
        N = self.N or X.size(0)
        prec = self.preconditioner

        with TicToc("ConjGrad preparation", False):
            if M is None:
                Knm = X
            else:
                Knm = None
            # Compute the right hand side
            if Knm is not None:
                B = incore_fmmv(Knm, Y / N, None, transpose=True, opt=self.params)
            else:
                B = self.kernel.mmv(M, X, Y / N, opt=self.params)
            B = prec.apply_t(B)

            # Define the Matrix-vector product iteration
            s1 = torch.cuda.Stream(X.device) if X.is_cuda else None
            capture_mmv = functools.partial(self.flk_mmv, Knm, X, M, s1, N, _lambda)

        # Run the conjugate gradient solver
        return self.optimizer.solve(initial_solution, B, capture_mmv, max_iter, callback)

    def solve_val_rhs(self, Xtr, Xval, M, Y, _lambda, initial_solution, max_iter, callback=None):
        N = self.N or Xtr.size(0)
        prec = self.preconditioner

        with TicToc("ConjGrad preparation", False):
            B = self.kernel.mmv(M, Xval, Y / N, opt=self.params)
            B = prec.apply_t(B)

            # Define the Matrix-vector product iteration
            s1 = torch.cuda.Stream(Xtr.device) if Xtr.is_cuda else None
            capture_mmv = functools.partial(self.flk_mmv, None, Xtr, M, s1, N, _lambda)

        # Run the conjugate gradient solver
        return self.optimizer.solve(initial_solution, B, capture_mmv, max_iter, callback)
