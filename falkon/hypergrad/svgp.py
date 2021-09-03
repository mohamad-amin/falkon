import gpytorch

from benchmark.models.gpytorch_variational_models import GenericApproxGP
from falkon.hypergrad.common import get_scalar
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, HyperOptimModel


class SVGP(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            num_data: int,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        if not opt_sigma or not opt_penalty:
            raise ValueError("Sigma, Penalty always optimized...")
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        self.variational_distribution = "diag"

        self.kernel = gpytorch.kernels.RBFKernel(ard_num_dims=sigma_init.shape[0])
        self.kernel.lengthscale = sigma_init
        mean_module = gpytorch.means.ConstantMean()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GenericApproxGP(centers_init,
                                     mean_module=mean_module,
                                     covar_module=self.kernel,
                                     var_strat="var_strat",
                                     var_distrib=self.variational_distribution,
                                     likelihood=likelihood,
                                     learn_ind_pts=self.opt_centers,
                                     )
        self.loss_fn = gpytorch.mlls.VariationalELBO(likelihood, self.model, num_data=num_data)
        if cuda:
            self.model = self.model.cuda()

    @property
    def penalty(self):
        return self.model.likelihood.noise

    @property
    def sigma(self):
        return self.kernel.lengthscale

    @property
    def centers(self):
        return self.model.strategy.inducing_points

    @centers.setter
    def centers(self, value):
        raise NotImplementedError("No setter implemented for centers")

    @sigma.setter
    def sigma(self, value):
        raise NotImplementedError("No setter implemented for sigma")

    @penalty.setter
    def penalty(self, value):
        raise NotImplementedError("No setter implemented for penalty")

    def hp_loss(self, X, Y):
        output = self.model(X)
        loss = -self.loss_fn(output, Y)
        return (loss,)

    def predict(self, X):
        preds = self.model.likelihood(self.model(X))
        return preds.mean

    @property
    def loss_names(self):
        return "mll"

    def __repr__(self):
        return f"SVGP(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}," \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}," \
               f"var_distrib={self.variational_distribution}, likelihood={self.model.likelihood})"
