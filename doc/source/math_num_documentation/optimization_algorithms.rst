.. _math_num_documentation.optimization_algorithms:

=======================
Optimization Algorithms
=======================

Below is a list of optimization algorithms implemented in `smash`:

- Step-by-Step (SBS): Global optimization algorithm :cite:p:`Michel1989`.
- Nelder-Mead: Derivative-free simplex method for unconstrained optimization :cite:p:`neldermead1965`.
- Powell: Direction-set method for unconstrained optimization without derivatives :cite:p:`powell1964`.
- Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bounded (L-BFGS-B): Quasi-Newton methods for bounded optimization :cite:p:`zhu1994bfgs`.
- Stochastic Gradient Descent (SGD): Iterative optimization using random mini-batches :cite:p:`bottou2012stochastic`.
- Adaptive Moment Estimation (Adam): Adaptive learning rates with momentum for fast convergence :cite:p:`kingma2014adam`.
- Adaptive Gradient (Adagrad): Subgradients-based optimization with adaptive learning rates :cite:p:`duchi2011adaptive`.
- Root Mean Square Propagation (RMSprop): Optimization with squared gradient averaging and adaptive learning rates :cite:p:`graves2013generating`.

The implementations of the Nelder-Mead, Powell, and L-BFGS-B algorithms in `smash` are based on optimization functions provided by the `SciPy <https://scipy.org>`__ library :cite:p:`virtanen2020scipy`.
