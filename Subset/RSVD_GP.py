import numpy as np
import GPy
import abc
from six import with_metaclass
from GPy import kern
from GPy import util
from paramz import ObsAr
from GPy.core.gp import GP
from GPy.core.parameterization import Param
from  GPy.models.multioutput_gp import MultioutputGP
from numpy.linalg.linalg import LinAlgError
import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag


log_2_pi = np.log(2*np.pi)

def opt_wrapper(args):
    m = args[0]
    kwargs = args[1]
    return m.optimize(**kwargs)



class RSVDGP(GP):
    def __init__(self,  X, Y, kernel, likelihood, mean_function=None, inference_method=None, name='RSVDGP', Y_metadata=None, normalizer=False):
        super(RSVDGP, self).__init__(X, Y, kernel, likelihood, mean_function=mean_function, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)



    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)

        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

    def objective_function(self):
        """
        The objective function for the given algorithm.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the objective function here.

        For probabilistic models this is the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your objective to minimize here!
        """
        if self.log_likelihood() is None:
            return np.nan
        #     K = self.kern.K(self.X)
        #     variance = self.likelihood.gaussian_variance(self.Y_metadata)
        #     diag.add(K, variance + 1e-8)
        #     K_logdet = np.log(np.linalg.det(K))
        #
        #     if self.mean_function is None:
        #         m = 0
        #     else:
        #         m = self.mean_function.f(self.X)
        #
        #     YYT_factor = self.Y_normalized - m
        #
        #     alpha = self.posterior.woodbury_vector
        #
        #     log_marginal = 0.5 * (-self.Y_normalized.size * log_2_pi - self.Y_normalized.shape[1] * K_logdet - np.sum(alpha * YYT_factor))
        #     self._log_marginal_likelihood = log_marginal
        #
        #     print('11111111111111111111111111111111111')

        return -float(self.log_likelihood()) - self.log_prior()







