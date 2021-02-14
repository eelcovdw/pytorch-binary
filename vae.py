import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import kl_divergence, Normal, Bernoulli, RelaxedBernoulli
import pytorch_lightning as pl

from probabll.dgm import register_conditional_parameterization
from probabll.dgm.prior import PriorLayer
from probabll.dgm.conditional import ConditionalLayer
from probabll.dgm.conditioners import FFConditioner
from probabll.dgm.likelihood import FullyFactorizedLikelihood

from estimators import get_estimator
import distributions


class VAE(pl.LightningModule):
    """
    A simple VAE model using pytorch lightning and dgm.pt
    """
    def __init__(self, x_size, z_size, conditional_x, conditional_z, prior_z, inf_estimator=None, num_valid_samples=10):
        super().__init__()
        self.x_size = x_size
        self.z_size = z_size
        self.q_z = conditional_z
        self.p_x = conditional_x
        self.p_z = prior_z
        self.inf_estimator = inf_estimator
        self.num_valid_samples = num_valid_samples

    def inference_parameters(self):
        return self.conditional_z.parameters()

    def generative_parameters(self):
        return self.conditional_x.parameters()

    def forward(self, x):
        q_z = self.q_z(x)
        if self.inf_estimator is None:
            z = q_z.rsample()
        else:
            z = q_z.sample()

        p_z = self.p_z(x.size(0), x.device)
        p_x = self.p_x(z)
        return z, q_z, p_z, p_x

    def loss(self, x, z, q_z, p_z, p_x):
        return_dict = dict()
        ll = p_x.log_prob(x).sum(-1)
        kl = kl_divergence(q_z, p_z).sum(-1)
        elbo = ll - kl
        loss = -elbo

        # When not using a pathwise estimator, add an additional term to estimate inference network gradients.
        if self.inf_estimator is not None:
            score_fn = lambda b: self.p_x(b).log_prob(x).sum(-1)
            surrogate = self.inf_estimator(q_z, score_fn)["estimate"]
            loss -= surrogate
            return_dict["surrogate"] = surrogate.detach().mean()

        return_dict["loss"] = loss.mean()
        return_dict["ll"] = ll.mean()
        return_dict["kl"] = kl.mean()
        
        return return_dict
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z, q_z, p_z, p_x = self.forward(x)
        loss_dict = self.loss(x, z, q_z, p_z, p_x)

        self.log("ll", loss_dict["ll"], prog_bar=True)
        self.log("kl", loss_dict["kl"], prog_bar=True)
        return loss_dict

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        q_z = self.q_z(x)
        p_z = self.p_z(x.size(0), x.device)

        # multi-sample ll estimate
        z = q_z.sample(torch.Size([self.num_valid_samples]))
        # TODO this only works for fully connected generative models,
        # as we are adding a new dim at position 0. Merge/unmerge along
        # batch dim to make compatible with CNN models.
        p_x = self.p_x(z)
        log_cond_x = p_x.log_prob(x.unsqueeze(0)).sum(-1)
        log_q_z = q_z.log_prob(z).sum(-1)
        log_p_z = p_z.log_prob(z).sum(-1)
        ll = torch.logsumexp(log_cond_x + log_p_z - log_q_z, 0) - math.log(self.num_valid_samples)

        kl = kl_divergence(q_z, p_z).sum(-1)
        elbo = log_cond_x.mean(0) - kl

        return_dict = {
            "val/log_marginal": ll,
            "val/kl": kl,
            "val/elbo": elbo
        }
        return return_dict

    def validation_epoch_end(self, outputs):
        """
        Merge all dicts from self.validation_step, and take mean along batch dim
        """
        if len(outputs) == 0:
            return
        return_dict = dict()
        for key in outputs[0].keys():
            return_dict[key] = torch.cat([o[key] for o in outputs], 0).mean(0)
        self.log_dict(return_dict)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt


def build_model(x_size, latent_size, latent_dist, estimator):
    # Define prior and posterior
    if latent_dist == "normal":
        posterior_type = Normal
        prior_type = Normal
        inf_output_size = latent_size * 2
        prior_params = [0., 1.]
    elif latent_dist == "bernoulli":
        posterior_type = Bernoulli
        prior_type = Bernoulli
        inf_output_size = latent_size
        prior_params = [0.5]
    elif latent_dist == "binary-concrete":
        posterior_type = RelaxedBernoulli
        prior_type = RelaxedBernoulli
        inf_output_size = latent_size
        prior_params = [0.5]
    else:
        raise ValueError(f'Unknown latent distribution: {latent_dist}')

    # Define estimator
    estimator_fn = get_estimator(estimator)

    # Define model components
    conditional_z = ConditionalLayer(
        event_size=latent_size,
        dist_type=posterior_type,
        conditioner=FFConditioner(
            input_size=x_size,
            output_size=inf_output_size,
            hidden_sizes=[200, 200],
            hidden_activation=torch.nn.ReLU()
        )
    )
    prior_z = PriorLayer(
        event_shape=latent_size,
        dist_type=prior_type,
        params=prior_params
    )
    conditional_x = FullyFactorizedLikelihood(
        event_size=x_size,
        dist_type=Bernoulli,
        conditioner=FFConditioner(
            input_size=latent_size,
            output_size=x_size,
            hidden_sizes=[200, 200],
            hidden_activation=torch.nn.ReLU()
        )
    )

    model = VAE(x_size, latent_size, conditional_x, conditional_z, prior_z, estimator_fn)
    return model
