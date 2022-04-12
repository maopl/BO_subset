# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag
import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from library.Approximate_posterior import PosteriorApproximate_RSVD as PosteriorA_RSVD, PosteriorExact
from sklearn.utils.extmath import randomized_svd

log_2_pi = np.log(2*np.pi)


class Approximate_RSVD_Inference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Approximate_RSVD_Inference, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"
        return input_dict

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)

        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y-m

        if K is None:
            K = kern.K(X)

        Ky = K.copy()
        diag.add(Ky, variance + 1e-8)

        if Ky.shape[0] >= 50:
            rank = int(Ky.shape[0] / 2)

            U,S,V = randomized_svd(K, rank)
            S = np.diag(S)
            # S_inv = np.diag(1./np.diag(S))

            variance_inv = np.eye(K.shape[0])*(1 / (variance + 1e-8))

            K_inv = variance_inv - variance_inv @ U @ S @ np.linalg.inv(S + S@ V @ variance_inv @ U @  S)@ S @ V@ variance_inv

            # K_inv_1 = variance_inv - variance_inv @ U @ np.linalg.inv(S_inv + V @ variance_inv @ U)@V @ variance_inv
            # difference = np.linalg.norm(K_inv-K_inv_1, 'fro')
            # print(difference)

            alpha = K_inv @ YYT_factor

            dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * K_inv)

            dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)


            # K_logdet = np.log(np.linalg.det(Ky))
            # log_marginal = 0.5 * (-Y.size * log_2_pi - Y.shape[1] * K_logdet - np.sum(alpha * YYT_factor))

            return PosteriorA_RSVD(woodbury_inv=K_inv, woodbury_vector=alpha, K=K), None, {'dL_dK': dL_dK,
                                                                                                   'dL_dthetaL': dL_dthetaL,
                                                                                                   'dL_dm': alpha}
        else:
            diag.add(Ky, variance + 1e-8)

            Wi, LW, LWi, W_logdet = pdinv(Ky)

            alpha, _ = dpotrs(LW, YYT_factor, lower=1)

            log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

            if Z_tilde is not None:
                # This is a correction term for the log marginal likelihood
                # In EP this is log Z_tilde, which is the difference between the
                # Gaussian marginal and Z_EP
                log_marginal += Z_tilde

            dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

            dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

            return PosteriorExact(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}



    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5*np.log(2*np.pi) - 0.5*np.log(c_diag) + 0.5*(g**2)/c_diag
        #believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        #this is the negative marginal LOO
        return -neg_log_marginal_LOO




    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5*np.log(2*np.pi) - 0.5*np.log(c_diag) + 0.5*(g**2)/c_diag
        #believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        #this is the negative marginal LOO
        return -neg_log_marginal_LOO

class ExactGaussianInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(ExactGaussianInference, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"
        return input_dict

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)

        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y-m

        if K is None:
            K = kern.K(X)

        Ky = K.copy()
        diag.add(Ky, variance+1e-8)

        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        return PosteriorExact(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5*np.log(2*np.pi) - 0.5*np.log(c_diag) + 0.5*(g**2)/c_diag
        #believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        #this is the negative marginal LOO
        return -neg_log_marginal_LOO