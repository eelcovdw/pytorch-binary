import torch
from torch.distributions import Normal, Bernoulli, RelaxedBernoulli, register_kl
from probabll.dgm import register_conditional_parameterization, register_prior_parameterization

@register_conditional_parameterization(Bernoulli)
def make_bernoulli(inputs, event_size):
    """
    Clamp the Bernoulli logits to avoid numerical instabilities.
    """
    assert inputs.size(-1) == event_size, "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
    return Bernoulli(logits=torch.clamp(inputs, -10, 10))


"""
Concrete distribution.

KL implementation and temperature hyperparameters are taken from [1]

[1] Maddison, C., A. Mnih, and Y. Teh. "The concrete distribution: A continuous relaxation of discrete random variables." 
International Conference on Learning Representations, 2017.
"""

def kl_concrete_concrete(p, q, n_samples=1):
    """
    KL is estimated for the logits of the binary concrete distribution to avoid underflow.
    """
    x_logit = p.base_dist.rsample(torch.Size([n_samples]))
    return (p.base_dist.log_prob(x_logit) - q.base_dist.log_prob(x_logit)).mean(0)

@register_kl(RelaxedBernoulli, RelaxedBernoulli)
def _kl_concrete_concrete(p, q):
    return kl_concrete_concrete(p, q, n_samples=10)


@register_prior_parameterization(RelaxedBernoulli)
def parametrize(batch_shape, event_shape, params, device, dtype):
    if len(params) == 1:
        p = torch.full(batch_shape + event_shape, params[0], device=device, dtype=dtype)
    elif len(params) == event_shape[0]:
        p = torch.Tensor(params).type(dtype).repeat(batch_shape + [1]).to(device)
    temp = torch.Tensor([0.5]).to(device)
    return RelaxedBernoulli(temp, probs=p)


@register_conditional_parameterization(RelaxedBernoulli)
def make_relaxed_bernoulli(inputs, event_size):
    if not inputs.size(-1) == event_size:
        raise ValueError(
            "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
        )
    temp = torch.Tensor([0.66]).to(inputs.device)
    return RelaxedBernoulli(temp, logits=inputs)
