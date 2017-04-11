# inverse-elliptical

This is a small framework for the elliptical inverse problem, i.e. given the PDE
-(k(x) * p'(x))' = f(x), which models the pressure p in a material with permeability k and sink/source f, and solely given noisy observations y_k = p(x_k) + e_k, with e_k Gaussian noise, this framework solves the Bayesian inverse problem
y -> k, i.e. we solve for permeability. You can choose between a Gaussian prior with covariance \beta\cdot(-\Delta)^{-\alpha}, a Gaussian prior with a Haar wavelet basis and a Besov prior with a Haar wavelet basis.

For a demo, run 'python -i invProblem.py "1"' or with value 2a, 2b, 3, 4a, 4b instead of 1. Caution: This may take a few minutes.

Regarding parameter values:

1: Gaussian (Fourier) prior, high noise, many observations
2a: Gaussian (Fourier) prior, low noise, many observations
2b: Gaussian (Fourier) prior, low noise, few observations
3: Gaussian (Wavelet) prior, high noise, many observations
4a: Gaussian (Wavelet) prior, low noise, many observations
4b: Gaussian (Wavelet) prior, low noise, few observations
