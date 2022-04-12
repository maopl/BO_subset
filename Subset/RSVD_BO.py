# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy
from  GPy.models.multioutput_gp import MultioutputGP
from GPyOpt.models.gpmodel import GPModel
from library.Approximation_inference_method import Approximate_RSVD_Inference, ExactGaussianInference

from Subset.RSVD_GP import RSVDGP


class RSVDGPBO(GPModel):
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

    def __init__(self,num_inducing=None, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000, optimize_restarts=1
                 , verbose=True, ARD=False, mean_function=None):

        super(RSVDGPBO, self).__init__(
            kernel=kernel, noise_var=noise_var, exact_feval=exact_feval, optimizer=optimizer, max_iters=max_iters, optimize_restarts=optimize_restarts,
            sparse=False, num_inducing=num_inducing, verbose=verbose, ARD=ARD, mean_function=mean_function
        )


        self.sigma = 0.05 # Noise of observations

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.RBF(self.input_dim) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        # inference_method = ExactGaussianInference()
        inference_method = Approximate_RSVD_Inference()
        likelihood = GPy.likelihoods.Gaussian()
        self.model = RSVDGP(X, Y, kernel=kern, inference_method=inference_method,likelihood=likelihood, mean_function=self.mean_function)


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
                self.model.optimize(messages=True, ipython_notebook=False)
            else:
                self.model.optimize_restarts(messages=True, num_restarts=self.optimize_restarts,  verbose=self.verbose)








