# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy
from  GPy.models.multioutput_gp import MultioutputGP
from GPyOpt.models.gpmodel import GPModel
from Subset.SparseGPR import SparseGPRegression



class SparseGPBO(GPModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).
    :param mean_function: GPy Mapping to use as the mean function for the GP model (default, None).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self,num_inducing, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000, optimize_restarts=1
                 , verbose=True, ARD=False, mean_function=None, sparse_method = 'FITC'):

        super(SparseGPBO, self).__init__(
            kernel=kernel, noise_var=noise_var, exact_feval=exact_feval, optimizer=optimizer, max_iters=max_iters, optimize_restarts=optimize_restarts,
            sparse=True, num_inducing=num_inducing, verbose=verbose, ARD=ARD, mean_function=mean_function
        )


        self.sigma = 0.05 # Noise of observations
        self.sparse_method = sparse_method

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        self.model = SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing, mean_function=self.mean_function, sparse_method = self.sparse_method)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.inducing_inputs.fix()
                self.model.optimize()
                self.model.randomize()
                self.model.Z.unconstrain()
                self.model.optimize()
                # self.model.optimize(messages=True, ipython_notebook=False)
            else:
                self.model.optimize_restarts(messages=True, num_restarts=self.optimize_restarts,  verbose=self.verbose)








