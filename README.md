# pytorch-binary
This repository contains distributions and estimators for deep generative models with binary latent variables in Pytorch.

The estimators only use the `torch.distributions` package, and can be used in any codebase without additional libraries. A simple experiment with a VAE on MNIST is included.

Available estimators:
- Pathwise estimator
- Score function estimator
- ARM estimator
- DisARM estimator

Future additions:
- More binary relaxations
- Control variates for SFE
- Multi-sample estimates
- Straight-through estimator
