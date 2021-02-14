import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Distribution, kl_divergence, Bernoulli

def get_estimator(estimator: str, *args, **kwargs):
    """
    Note: Estimators are implemented as objects instead of functions,
    to add hyperparameters or initialize control variates in the init.
    """
    if estimator.lower() == "pathwise":
        # Pathwise estimator is already handled by autograd.
        return None
    if estimator.lower() == "sfe":
        return SFEstimator(*args, **kwargs)
    elif estimator.lower() == "arm":
        return ARMEstimator(*args, **kwargs)
    elif estimator.lower() == "disarm":
        return DisARMEstimator(*args, **kwargs)
    else:
        raise ValueError(f"unknown estimator: {estimator}")


class SFEstimator(nn.Module):
    def forward(self, q_z: Distribution, f, num_samples=1, return_samples=False, return_scores=False):
        """
        Standard SFE, no control variates. 

        Args:
            q_z (Distribution): Distribution with an rsample method.
            f (function): Function that takes a sample b [B, ndims] ~ q_z,
                and returns a score of shape [B].
                For example, in a VAE setting this could be the log-likelihood log p(x | b).
            return_samples (bool, optional): If True, return samples z. Defaults to False.
            return_scores (bool, optional): If True, return the scores f(z). Defaults to False.

        Returns:
            dict: dictionary with "estimate" pathwise estimate, and optional "samples" and "scores" with shape [B, num_samples, *event_size].
        """

        if num_samples > 1:
            raise NotImplementedError("TODO multi-sample SFE")
        z = q_z.sample()
        score = f(z)
        
        # gradients w.r.t the parameters of q_z with SFE
        estimate = score.detach() * q_z.log_prob(z).sum(-1)

        return_dict = {"estimate": estimate}
        if return_samples:
            return_dict["samples"] = z.unsqueeze(1)
        if return_scores:
            return_dict["scores"] = score.unsqueeze(1)
        return return_dict


class ARMEstimator(nn.Module):
    def forward(self, q_z: Bernoulli, f, num_samples=2, return_samples=False, return_scores=False):
        """
        Implementation of the ARM estimator [1], 
        a low variance score function estimator for Bernoulli distributions.

        This implementation is for completeness only. DisARM will always have 
        lower gradient variance than ARM, and performs significantly better.

        Args:
            q_z (Bernoulli): Bernoulli distribution.
            f : Function that takes a sample b [B, ndims] ~ q_z, and returns a score of shape [B].
            return_samples (bool, optional): If True, return sample b and antithetic sample b_. Defaults to False.
            return_scores (bool, optional): If True, return the scores f(b) and f(b_). Defaults to False.
            
        Returns:
            dict: dictionary with "estimate" key as ARM estimate, and optional "samples" and "scores".
        """

        if num_samples % 2 != 0:
            raise ValueError("ARMEstimator requires an even number of samples.")

        if num_samples != 2:
            raise NotImplementedError("TODO multi-sample ARM")

        logits = q_z.logits
        with torch.no_grad():
            # Sample from logistic, generate corresponding Bernoulli sample and antithetic.
            u = torch.rand(logits.size()).to(logits.device)
            eps = torch.logit(u)
            b = (logits + eps > 0.).float()
            b_ = (logits - eps > 0.).float()

            # ARM Estimate
            fb = f(b)
            fb_ = f(b_)
            reward = 0.5 * (fb - fb_).unsqueeze(-1) * (2 * u - 1)
        estimate = (reward * logits).sum(-1)

        return_dict = {"estimate": estimate}
        if return_samples:
            return_dict["samples"] = torch.stack([b, b_], dim=1)
        if return_scores:
            return_dict["scores"] = torch.stack([fb, fb_], dim=1)
        return return_dict


class DisARMEstimator(nn.Module):
    def forward(self, q_z: Bernoulli, f, num_samples=2, return_samples=False, return_scores=False):
        """
        Implementation of the DisARM estimator [2], 
        a low variance score function estimator for Bernoulli distributions.

        Calculates a low variance gradient estimate w.r.t. the parameters of a product of Bernoullis q(b):
        Nabla_{lambda} E_{q(b; lambda)}[ f(b) ]

        [2] Dong, Zhe, Andriy Mnih, and George Tucker. 
            "DisARM: An antithetic gradient estimator for binary latent variables."
            Advances in Neural Information Processing Systems 33 (2020).

        Args:
            q_z (Bernoulli): Bernoulli distribution.
            f : Function that takes a sample b [B, ndims] ~ q_z, and returns a score of shape [B]. 
            return_samples (bool, optional): If True, return sample b and antithetic sample b_. Defaults to False.
            return_scores (bool, optional): If True, return the scores f(b) and f(b_). Defaults to False.
            
        Returns:
            dict: dictionary with "estimate" key as DisARM estimate, and optional "samples" and "scores".
        """
        if num_samples % 2 != 0:
            raise ValueError("DisARMEstimator requires an even number of samples.")

        if num_samples != 2:
            raise NotImplementedError("TODO multi-sample DisARM")
        
        logits = q_z.logits
        with torch.no_grad():
            # Sample from logistic, generate corresponding Bernoulli sample and antithetic.
            u = torch.rand(logits.size()).to(logits.device)
            eps = torch.logit(u)
            b = (logits + eps > 0.).float()
            b_ = (logits - eps > 0.).float()

            # DisARM Estimate
            s = torch.pow(-1, b_) * (~torch.eq(b, b_)).float() * torch.sigmoid(torch.abs(logits))
            fb = f(b)
            fb_ = f(b_)
            reward = 0.5 * (fb - fb_).unsqueeze(-1) * s
        estimate = (reward * logits).sum(-1)
        
        return_dict = {"estimate": estimate}
        if return_samples:
            return_dict["samples"] = torch.stack([b, b_], dim=1)
        if return_scores:
            return_dict["scores"] = torch.stack([fb, fb_], dim=1)
        return return_dict
