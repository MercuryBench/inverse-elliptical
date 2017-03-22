# inverse-elliptical

This is a small framework for the elliptical inverse problem, i.e. given the PDE
-(k(x) * p'(x))' = f(x), which models the pressure p in a material with permeability k and sink/source f, and solely given noisy observations y_k = p(x_k) + e_k, with e_k Gaussian noise, this framework solves the Bayesian inverse problem
y -> k, i.e. we solve for permeability. You can choose between a Gaussian prior with covariance \beta\cdot(-\Delta)^{-\alpha} and a Besov prior with a Haar wavelet basis.

pressure.png shows a sample output: the black curve is the ground truth pressure with red dot noisy measurements. The algorithm starts with an arbitrary pressure and yields the green curve (as the forward solution of the algorithm's candidate for log-permeability).
Compare this with logpermeability.png: The black curve is the ground truth permeability for which we want to solve. The red curve is the arbitrary initial permeability condition, the green curve is the empirical conditional mean log-permeability. It can be seen that the log-permeability does not need to be visually close to the ground truth in order to be able to fit the data. This is due to the structure of the forward operator.
