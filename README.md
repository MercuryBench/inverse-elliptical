# inverse-elliptical

This is a small framework for the elliptical inverse problem, i.e. given the PDE
-(k(x) * p'(x))' = f(x), which models the pressure p in a material with permeability k and sink/source f, and solely given noisy observations y_k = p(x_k) + e_k, with e_k Gaussian noise, this framework solves the Bayesian inverse problem
y -> k, i.e. we solve for permeability. You can choose between a Gaussian prior with covariance \beta\cdot(-\Delta)^{-\alpha}, a Gaussian prior with a Haar wavelet basis and a Besov prior with a Haar wavelet basis.

For a demo, run 'python -i invProblem.py "1"' or with value 2a, 2b, 3, 4a, 4b instead of 1.
