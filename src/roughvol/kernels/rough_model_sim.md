# Simulation methods for rough volatility models

This note organizes the main simulation methods for rough volatility models by **model class** and **numerical mechanism** with psuedo-codes. It separates

1. **Gaussian rough-volatility models** such as rough Bergomi, where the main numerical task is to simulate a Gaussian Volterra field efficiently;
2. **Affine Volterra / rough Heston-type models**, where the variance is no longer lognormal Gaussian and positivity becomes a central issue;
3. **Accelerators and wrappers**, which are not new path generators but materially change practical performance.
---

## 1. Model taxonomy

The general rough-vol model is a stochastic volatility model driven by a singular kernel; rough Bergomi is the Gaussian/lognormal special case, while rough Heston is the affine square-root Volterra special case. These volatility models typically take the form

$$
V_t = \Phi(X_t), \qquad X_t = \int_0^t g(t-s) dW_s + \text{drift if necessary}.
$$

### Choice of kernels

To capture **roughness**, canonical choice of the kernel is usually Volterra 

$$
g(u) \sim u^{H-\frac{1}{2}}, \qquad H\in(0,\frac{1}{2}).
$$

Mathematical intuition is the Mandelbrot–Van Ness representation of fBM with Hurst $H$:

$$
B_t^H = \frac{1}{\Gamma(H+1/2)}
\left(
\int_{-\infty}^0 \left((t-s)^{H-1/2}-(-s)^{H-1/2}\right) dW_s
+
\int_0^t (t-s)^{H-1/2} dW_s
\right)
$$

### Stock price modelling
and stock dynamics

$$
\frac{dS_t}{S_t} = \sqrt{V_t} dZ_t, \qquad dZ_t = \rho dW_t + \sqrt{1-\rho^2} dW_t^\perp.
$$

### Canonical Models

The canonical example is **rough Bergomi**:

$$
V_t = \xi_0(t)\exp\left(\eta\widetilde W_t - \frac{\eta^2}{2}t^{2H}\right),
\qquad
\widetilde W_t = \sqrt{2H}\int_0^t (t-s)^{H-\frac{1}{2}}dW_s.
$$

Another class of models are given by the **affine Volterra / rough Heston-type**: These models are usually written as stochastic Volterra equations. The rough Heston model is

$$
\frac{dS_t}{S_t} = \sqrt{V_t} dW_t^{(1)},
\qquad
V_t = V_0 + \int_0^t K(t-s)\Big(\lambda(\theta - V_s) ds + \nu\sqrt{V_s} dW_s^{(2)}\Big),
$$

with

$$
K(t)=\frac{t^{H-\frac{1}{2}}}{\Gamma(H+\frac{1}{2})}, \qquad H\in(0,\frac{1}{2}),
$$

and

$$
d\langle W^{(1)},W^{(2)}\rangle_t = \rho dt.
$$

An equivalent integrated-variance formulation uses

$$
X_t := \int_0^t V_s ds,
$$

and rewrites the dynamics in terms of \(X\) and the martingales

$$
M_t^{(i)} := \int_0^t \sqrt{V_s} dW_s^{(i)}.
$$

### What counts as a distinct simulation method?

A useful classification is by **numerical mechanism**:

- exact or near-exact Gaussian sampling;
- time-domain Volterra discretization;
- tree/random-walk approximations;
- Markovian lift / sum-of-exponentials approximation;
- positivity-preserving implicit or distribution-matching schemes;
- Monte Carlo accelerators such as control variates, QMC, and MLMC.

---

## 2. Gaussian rough-volatility models: methods and formulas

## 2.1 Exact covariance-factorization simulation

### Idea

On a grid \(t_i=i\Delta t\), simulate the Gaussian vector

$$
(\widetilde W_{t_1},\dots,\widetilde W_{t_n}, W_{t_1},\dots,W_{t_n}, W^\perp_{t_1},\dots,W^\perp_{t_n})
$$

from its covariance matrix, then reconstruct \(V\) and \(S\).

### Core formulas for rough Bergomi

$$
\widetilde W_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}dW_s.
$$

With the normalization above,

$$
\mathrm{Var}(\widetilde W_t)=t^{2H},
$$

and for \(s,t\ge 0\),

$$
\mathrm{Cov}(\widetilde W_t,\widetilde W_s)
= \frac12\big(t^{2H}+s^{2H}-|t-s|^{2H}\big).
$$

The stock driver is built from

$$
\Delta Z_i = \rho\,\Delta W_i + \sqrt{1-\rho^2}\,\Delta W_i^\perp.
$$

Then set

$$
V_{t_i}=\xi_0(t_i)\exp\!\left(\eta\widetilde W_{t_i}-\frac{\eta^2}{2}t_i^{2H}\right),
$$

and update the log stock by

$$
\log S_{t_{i+1}} = \log S_{t_i} - \frac12 V_{t_i}\Delta t + \sqrt{V_{t_i}}\,\Delta Z_i.
$$

### Pseudocode

```text
Algorithm ExactGaussian-rBergomi
Input: grid t_0,...,t_n, H, eta, rho, forward variance xi0(.), S0
1. Build covariance matrix C of the Gaussian vector (W-grid, Wtilde-grid, Wperp-grid).
2. Compute a factorization C = L L^T (Cholesky / PCA / SVD).
3. Draw G ~ N(0, I).
4. Set X = L G.
5. Read off W_{t_i}, Wperp_{t_i}, Wtilde_{t_i} from X.
6. For i = 0,...,n:
      V_i = xi0(t_i) * exp(eta * Wtilde_i - 0.5 * eta^2 * t_i^(2H)).
7. For i = 0,...,n-1:
      dW_i     = W_{t_{i+1}} - W_{t_i}
      dWperp_i = Wperp_{t_{i+1}} - Wperp_{t_i}
      dZ_i     = rho * dW_i + sqrt(1-rho^2) * dWperp_i
      logS_{i+1} = logS_i - 0.5 * V_i * dt + sqrt(V_i) * dZ_i
8. Return paths V_i, S_i.
```

### Remarks

- This is the clean benchmark method for rough Bergomi.
- It is accurate but expensive if implemented naively, especially when the covariance matrix is factorized directly.
- It is most useful as a gold standard, for small grids, or for validating faster approximations.

---

## 2.2 Exact stationary Gaussian simulation via circulant embedding / Davies–Harte

### Idea

If the object to be simulated can be reduced to a **stationary Gaussian increment sequence** (for example fractional Gaussian noise), then FFT-based exact simulation is possible.

This is not a separate rough-vol model, but a separate **Gaussian engine**. It is relevant whenever the Volterra driver or an increment representation is stationary or can be embedded into a stationary sequence.

### Fractional Gaussian noise covariance

For fBm \(B^H\), the increments

$$
Y_k = B^H_{k+1} - B^H_k
$$

are stationary with covariance

$$
\gamma(m) = \frac12\Big((m+1)^{2H} - 2m^{2H} + (m-1)^{2H}\Big), \qquad m\ge 1,
$$

and \(\gamma(0)=1\).

### Pseudocode

```text
Algorithm DaviesHarte-fGn
Input: H, number of increments n
1. Form the circulant first row from the target covariance sequence gamma(0),...,gamma(n-1).
2. FFT the circulant row to obtain eigenvalues lambda_k.
3. Draw complex Gaussian variables with Hermitian symmetry.
4. Multiply by sqrt(lambda_k).
5. Apply inverse FFT to obtain a real Gaussian sequence with the desired covariance.
6. Cumulate increments to obtain fBm if needed.
7. Use the resulting Gaussian path inside the rough-volatility model.
```

### Remarks

- This is an exact FFT Gaussian sampler, not a rough-vol path discretization by itself.
- It is especially useful when one chooses a model representation directly in terms of fBm or fractional Gaussian increments.
- In practice, it can replace dense-matrix Cholesky when the covariance structure allows it.

---

## 2.3 The Bennedsen–Lunde–Pakkanen hybrid scheme

### Idea

For a truncated Brownian semistationary / Volterra process

$$
X(t)=\int_0^t g(t-s)\sigma(s)\,dW_s,
$$

with singular kernel \(g(x)\sim x^\alpha L_g(x)\), \(\alpha\in(-1/2,1/2)\setminus\{0\}\), approximate the kernel by

- a **power function near the singularity**, and
- a **step function away from the singularity**.

This is the standard fast simulation method for rough Bergomi-type Gaussian drivers.

### Discrete formula

On a grid \(t_i=i/n\), the hybrid approximation has the form

$$
X_n(t_i)
=
\sum_{k=1}^{\kappa} L_g(k/n)\,\sigma_{i-k}^n\,W_{i-k,k}^n
+
\sum_{k=\kappa+1}^{N_n} g(b_k^*/n)\,\sigma_{i-k}^n\,W_{i-k}^n,
$$

where

$$
W_{i-k,k}^n := \int_{t_{i-k}}^{t_{i-k+1}} (t_i-s)^\alpha \, dW_s,
\qquad
W_{i-k}^n := W_{t_{i-k+1}}-W_{t_{i-k}}.
$$

The first sum handles the kernel singularity accurately; the second sum is a discrete convolution and can be evaluated by FFT.

For rough Bergomi, one applies this to

$$
\widetilde W_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}dW_s,
$$

then sets

$$
V_{t_i} = \xi_0(t_i)\exp\!\left(\eta\widetilde W_{t_i} - \frac{\eta^2}{2}t_i^{2H}\right).
$$

### Pseudocode

```text
Algorithm Hybrid-BLP
Input: grid t_0,...,t_n, singular exponent alpha = H-1/2, cutoff kappa, kernel g
1. Draw Brownian increments dW_j on the grid.
2. For each time index i:
      Near-field term A_i = 0
      For k = 1,...,kappa:
          sample W_{i-k,k}^n = integral over one cell of (t_i-s)^alpha dW_s
          A_i += L_g(k/n) * sigma_{i-k}^n * W_{i-k,k}^n
3. Build the far-field weights a_k = g(b_k^*/n),  k = kappa+1,...,N_n.
4. Compute the far-field convolution
      B_i = sum_{k=kappa+1}^{N_n} a_k * sigma_{i-k}^n * dW_{i-k}
   using FFT.
5. Set X_i = A_i + B_i.
6. In rough Bergomi, define
      V_i = xi0(t_i) * exp(eta * X_i - 0.5 * eta^2 * t_i^(2H)).
7. Simulate the stock with log-Euler using correlated Brownian increments.
8. Return paths.
```

### Remarks

- Complexity is typically \(O(n\log n)\) because of the FFT convolution.
- This is the most common workhorse for rough Bergomi Monte Carlo.
- It is approximate, but usually very accurate for singular Volterra kernels.

---

## 2.4 rDonsker / random-walk approximation (Horvath–Jacquier–Muguruza–Søjmark)

### Idea

Replace Brownian motion by a scaled random walk, then push it through the fractional/Volterra operator. This yields weak convergence and naturally leads to **fractional trees** for early exercise.

### Discrete formulas

On a grid \(t_i=i\Delta t\), let \((\xi_i)\) be i.i.d. centered, variance-one random variables. Define

$$
W_n(t_i)=\sqrt{\Delta t}\sum_{k=1}^i \xi_k.
$$

For a Volterra operator \(\mathcal G^\alpha\), write the discrete convolution

$$
(\mathcal G^\alpha W_n)(t_i)
\approx
\sqrt{\Delta t}\sum_{k=1}^i g(t_{i-k+1})\,\xi_k.
$$

A generic rough-vol approximation then takes the form

$$
X_{i+1}=X_i - \frac12\Phi(Y_i)\Delta t + \sqrt{\Phi(Y_i)}\,\Delta B_i,
\qquad
Y_i=(\mathcal G^\alpha W_n)(t_i).
$$

For rough Bergomi,

$$
V_i = \xi_0(t_i)\exp\!\left(\eta Y_i - \frac{\eta^2}{2}t_i^{2H}\right).
$$

### Pseudocode

```text
Algorithm rDonsker-RoughVol
Input: grid t_0,...,t_n, i.i.d. innovations xi_i, kernel weights g_j
1. Build discrete Brownian path
      Wn_i = sqrt(dt) * sum_{k=1}^i xi_k.
2. Compute the discrete Volterra convolution
      Y_i = sqrt(dt) * sum_{k=1}^i g_{i-k+1} * xi_k.
   (Use FFT if desired.)
3. Convert Y_i to variance, e.g. for rough Bergomi
      V_i = xi0(t_i) * exp(eta * Y_i - 0.5 * eta^2 * t_i^(2H)).
4. Draw a second innovation sequence for the orthogonal Brownian component.
5. Update the stock with correlated increments.
6. Return paths.
```

### Tree version

If the innovations are Bernoulli \(\pm1\), the same construction yields a recombining/nonrecombining fractional tree approximation that can be combined with backward induction for Bermudan or American claims.

### Remarks

- This is a genuine simulation family, not just an analysis device.
- It is especially useful when one wants weak approximation, simple implementation, and early-exercise capability.

---

## 2.5 Direct convolution / Riemann-sum simulation of Gaussian Volterra drivers

### Idea

The most literal discretization of

$$
X_t = \int_0^t g(t-s)\,dW_s
$$

is the left-point or midpoint quadrature

$$
X_{t_i} \approx \sum_{j=0}^{i-1} g(t_i-t_j)\,\Delta W_j.
$$

For rough kernels \(g(u)\sim u^{H-1/2}\), this is the simplest route but performs poorly near the singularity unless the grid is fine or special corrections are added.

### Pseudocode

```text
Algorithm DirectGaussianConvolution
Input: grid, kernel g, Brownian increments dW_j
1. For i = 1,...,n:
      X_i = sum_{j=0}^{i-1} g(t_i - t_j) * dW_j
2. Convert X_i to variance V_i.
3. Simulate S by log-Euler.
```

### Remarks

- This is conceptually basic and often used as a pedagogical or baseline discretization.
- It is slower and less accurate near the singularity than the hybrid scheme.

---

## 2.6 Variance-reduction and quadrature wrappers for Gaussian rough-vol models

These methods do **not** change the path generator but are important enough to list separately.

### 2.6.1 Turbocharging Monte Carlo (McCrickerd–Pakkanen)

Typical wrappers:

- conditional Monte Carlo on partially integrated quantities;
- control variates;
- antithetic coupling of Brownian paths;
- careful discretization of payoffs.

### Generic pseudocode wrapper

```text
Algorithm VarianceReductionWrapper
Input: any path generator producing payoff X
1. Construct antithetic path and payoff X_tilde.
2. Construct control variate Y with known or accurately precomputed expectation E[Y].
3. Return estimator
      0.5*(X + X_tilde) - beta*(Y - E[Y]).
4. Average over Monte Carlo samples.
```

### 2.6.2 Quasi-Monte Carlo / sparse-grid quadrature

Given a deterministic transform

$$
G \mapsto \text{path}(G) \mapsto \text{payoff}(G),
$$

replace pseudorandom Gaussians by low-discrepancy points mapped to Gaussian space. This is especially effective after dimension reduction (PCA / bridge construction).

### 2.6.3 Multilevel Monte Carlo

Given a hierarchy of grid sizes \(\Delta t_\ell\), use

$$
\mathbb E[P_L] = \mathbb E[P_0] + \sum_{\ell=1}^L \mathbb E[P_\ell - P_{\ell-1}],
$$

and estimate each term with a coupled coarse/fine simulation.

### Remarks

- These are best viewed as *performance layers* on top of exact, hybrid, or rDonsker path generation.
- For VIX and other integral functionals in rough Bergomi, MLMC can materially improve complexity.

---

## 3. Affine Volterra / rough Heston-type models: methods and formulas

## 3.1 Direct Euler discretization of the stochastic Volterra equation

### Idea

Discretize the rough Heston variance equation directly in time.

### Model

$$
V_t = V_0 + \int_0^t K(t-s)\Big(\lambda(\theta-V_s)\,ds + \nu\sqrt{V_s}\,dW_s\Big),
\qquad
K(t)=\frac{t^{H-1/2}}{\Gamma(H+1/2)}.
$$

### Explicit time-stepping formula

On a grid \(t_i=i\Delta t\), a left-point Euler-type discretization is

$$
V_{i+1}
=
V_0
+
\sum_{j=0}^{i} K(t_{i+1}-t_j)\lambda(\theta - V_j)\Delta t
+
\sum_{j=0}^{i} K(t_{i+1}-t_j)\nu\sqrt{(V_j)_+}\,\Delta W_j.
$$

The stock is then updated by

$$
\log S_{i+1} = \log S_i - \frac12 V_i\Delta t + \sqrt{V_i}\,\Delta W_i^{(1)}.
$$

### Pseudocode

```text
Algorithm VolterraEuler-RoughHeston
Input: grid, kernel K, parameters lambda, theta, nu, rho, V0, S0
1. Draw correlated Brownian increments dW1_j, dW2_j.
2. Set V_0 = V0, logS_0 = log(S0).
3. For i = 0,...,n-1:
      drift_sum = 0
      diff_sum  = 0
      For j = 0,...,i:
          kij = K(t_{i+1} - t_j)
          drift_sum += kij * lambda * (theta - V_j) * dt
          diff_sum  += kij * nu * sqrt(max(V_j,0)) * dW2_j
      V_{i+1} = V0 + drift_sum + diff_sum
      logS_{i+1} = logS_i - 0.5 * max(V_i,0) * dt + sqrt(max(V_i,0)) * dW1_i
4. Return V_i, S_i.
```

### Remarks

- This is the most direct discrete-time scheme.
- It has quadratic cost because each new time step uses the entire history.
- Positivity is delicate; clipping \((V_j)_+\) is common in implementations.

---

## 3.2 Euler discretization of the integrated-variance formulation

### Idea

Work with

$$
X_t=\int_0^t V_s ds,
$$

rather than directly with \(V\), because some path functionals and convergence arguments are cleaner in this representation.

### Formulation

The integrated rough Heston system can be written in the form

$$
X_t = V_0 t + \int_0^t K(t-s)\big(\theta s - \lambda X_s + \nu M_s\big)ds,
$$

where \(M_t = \int_0^t \sqrt{V_u}\,dW_u\) and \(d\langle M\rangle_t = V_t dt\).

A discrete approximation evolves \(X_i\), then recovers

$$
V_i \approx \frac{X_i - X_{i-1}}{\Delta t}
$$

or a similar finite-difference estimate.

### Pseudocode

```text
Algorithm IntegratedVarianceEuler
Input: grid, kernel K, parameters lambda, theta, nu
1. Initialize X_0 = 0, M_0 = 0.
2. For i = 0,...,n-1:
      Approximate V_i from past X values.
      Update martingale increment
          dM_i = sqrt(max(V_i,0)) * dW_i
      Update X_{i+1} using
          X_{i+1} = V0 * t_{i+1} + sum_{j=0}^i K(t_{i+1}-t_j)
                    * (theta * t_j - lambda * X_j + nu * M_j) * dt
3. Recover V from X.
4. Simulate the stock.
```

### Remarks

- This is less canonical in implementations than direct Volterra Euler, but it is part of the rigorous simulation literature.
- It is mainly useful when integrated variance is the natural state variable for the payoff or for the analysis.

---

## 3.3 Riemann-sum simulation for affine forward variance models

### Idea

Affine forward variance (AFV) models, including rough Heston, can be simulated by discretizing the convolution terms appearing in the forward variance dynamics or in the integrated variance representation.

In practice this means storing the past path and repeatedly recomputing convolution integrals by quadrature.

### Generic form

If the model is represented by a forward variance curve \(\xi_t(u)\), then on a grid one approximates

$$
\int_0^{t_i} K(t_i-s) f(V_s)\,ds
\approx
\sum_{j=0}^{i-1} K(t_i-t_j) f(V_j)\Delta t.
$$

### Pseudocode

```text
Algorithm AFV-RiemannSum
1. Initialize state variables (forward variance curve or equivalent history arrays).
2. For each time step i:
      Recompute each required convolution integral by a Riemann sum over j <= i.
      Update variance-related state variables.
      Update stock.
3. Return the path.
```

### Remarks

- This is a category rather than a single named scheme.
- Computationally it has the same basic weakness as direct Volterra Euler: quadratic cost in the number of time steps.

---

## 3.4 HQE: hybrid quadratic-exponential schemes for rough Heston / AFV models

### Idea

Adapt Andersen’s QE scheme to rough Heston or more general affine forward variance models. At each step, approximate the conditional law of the next variance value using its conditional mean and variance, and then sample from a two-branch approximation.

This is an important practical scheme associated with Gatheral’s AFV simulation line.

### Moment inputs

At time step \(n\), compute or approximate

$$
m_n = \mathbb E[V_{n+1}\mid \mathcal F_{t_n}],
\qquad
s_n^2 = \mathrm{Var}(V_{n+1}\mid \mathcal F_{t_n}),
\qquad
\psi_n = \frac{s_n^2}{m_n^2}.
$$

### QE branch formulas

If \(\psi_n \le \psi_c\), use the quadratic-Gaussian branch

$$
V_{n+1} = a_n (b_n + Z)^2,
\qquad Z\sim N(0,1),
$$

where \(a_n,b_n\) are chosen so that

$$
\mathbb E[V_{n+1}\mid \mathcal F_{t_n}] = m_n,
\qquad
\mathrm{Var}(V_{n+1}\mid \mathcal F_{t_n}) = s_n^2.
$$

If \(\psi_n > \psi_c\), use the atom-plus-exponential branch

$$
V_{n+1} = 0 \quad \text{with probability } p_n,
$$

and otherwise

$$
V_{n+1} = \frac{1}{\beta_n}\log\!\left(\frac{1-p_n}{1-U}\right),
\qquad U\sim \mathrm{Unif}(0,1),
$$

with \(p_n,\beta_n\) chosen to match \(m_n\) and \(s_n^2\).

### Pseudocode

```text
Algorithm HQE-RoughHeston
Input: grid, rough-Heston / AFV parameters
1. Maintain the past path needed to evaluate conditional moments m_n and s_n^2.
2. For n = 0,...,N-1:
      Compute conditional mean m_n of V_{n+1}.
      Compute conditional variance s_n^2 of V_{n+1}.
      psi_n = s_n^2 / m_n^2.
      If psi_n <= psi_c:
          choose a_n, b_n by moment matching
          draw Z ~ N(0,1)
          V_{n+1} = a_n * (b_n + Z)^2
      Else:
          choose p_n, beta_n by moment matching
          draw U ~ Uniform(0,1)
          V_{n+1} = 0 with probability p_n
          otherwise V_{n+1} = (1/beta_n) * log((1-p_n)/(1-U))
      Update stock using correlated Brownian increment / integrated variance approximation.
3. Return path.
```

### Remarks

- This is one of the main viable rough-Heston simulators before the recent Markovian weak schemes.
- It is usually accurate, but retains quadratic cost because conditional moments depend on the whole history.
- It is a *distribution-matching* rather than a direct Euler scheme.

---

## 3.5 Markovian multi-factor approximation (Abi Jaber–El Euch)

### Idea

Approximate the rough kernel by a sum of exponentials:

$$
K(t) \approx K_N(t) = \sum_{m=1}^N w_m e^{-x_m t}.
$$

This lifts the non-Markovian model to an \(N\)-factor Markov diffusion.

### Factor representation

Define factor processes

$$
Y_t^{(m)} = \int_0^t e^{-x_m(t-s)}\Big(\lambda(\theta-V_s)\,ds + \nu\sqrt{V_s}\,dW_s\Big),
$$

and set

$$
V_t \approx V_0 + \sum_{m=1}^N w_m Y_t^{(m)}.
$$

Then

$$
dY_t^{(m)} = \big(-x_m Y_t^{(m)} + \lambda(\theta-V_t)\big)dt + \nu\sqrt{V_t}\,dW_t.
$$

The stock equation becomes a standard Markovian stochastic volatility system:

$$
\frac{dS_t}{S_t}=\sqrt{V_t}\,dW_t^{(1)}.
$$

### Pseudocode

```text
Algorithm MarkovianLift-Euler
Input: weights w_m, mean reversions x_m, parameters lambda, theta, nu, rho
1. Initialize Y_m(0)=0 for m=1,...,N and V_0 = V0.
2. For i = 0,...,n-1:
      Draw correlated Brownian increments dW1_i, dW2_i.
      For m = 1,...,N:
          Y_m^{i+1} = Y_m^i
                      + (-x_m * Y_m^i + lambda * (theta - V_i)) * dt
                      + nu * sqrt(max(V_i,0)) * dW2_i
      V_{i+1} = V0 + sum_{m=1}^N w_m * Y_m^{i+1}
      logS_{i+1} = logS_i - 0.5 * max(V_i,0) * dt + sqrt(max(V_i,0)) * dW1_i
3. Return Y_m, V, S.
```

### Remarks

- This is arguably the most important general-purpose route for rough Heston in current practice.
- Once the lift is in place, one can use many standard Markov diffusion discretizations.
- Complexity becomes linear in the number of time steps, up to the factor dimension \(N\).

---

## 3.6 Weak Markovian simulation schemes (Bayer–Breneis)

### Idea

Start from the multi-factor Markovian lift and then use a **weak simulation scheme** rather than plain Euler. The key point is to approximate the one-step law of the variance update by a nonnegative discrete random variable that matches several conditional moments.

### Generic one-step structure

Given the Markovian state at time \(t_n\), compute conditional moments of the one-step variance proxy and replace the true increment by a discrete random variable \(\zeta_n\) supported on a few nonnegative points, chosen so that

$$
\mathbb E[\zeta_n^k\mid \mathcal F_{t_n}] = \text{target moment } k,
\qquad k=1,\dots,r.
$$

Then update the factors using \(\zeta_n\) instead of a Gaussian/Euler increment.

### Pseudocode

```text
Algorithm WeakMarkovian-RoughHeston
Input: Markovian lift with N factors
1. Initialize factor state Y^i and V_i.
2. For each step i:
      Compute the conditional moments of the next-step variance proxy.
      Choose a small nonnegative support {q_1,...,q_M} and probabilities {p_1,...,p_M}
      so that several moments are matched.
      Draw zeta_i from this discrete law.
      Update factor variables using the weak one-step rule driven by zeta_i.
      Recover V_{i+1} from the factors.
      Update S_{i+1}.
3. Return the path.
```

### Remarks

- This is more specialized than the previous methods, but important in the recent literature.
- Numerical evidence indicates substantially better weak accuracy than Markovian Euler, often at similar or lower total cost for option pricing.
- It is especially attractive for Bermudan pricing because the lifted model is Markovian.

---

## 3.7 Positivity-preserving integrated-process schemes: iVi (Abi Jaber–Attal)

### Idea

Instead of simulating the spot variance directly, simulate the **integrated Volterra square-root process** using an Inverse Gaussian-based construction. This is designed for singular \(L^1\) kernels and is particularly promising in the rough or hyper-rough regime.

### Generic integrated setup

Let

$$
X_t = \int_0^t V_s ds.
$$

The iVi philosophy is to update \(X\) using only integrated-kernel quantities, then recover the variance increment from the increment of \(X\). The scheme is implicit in integrated form and preserves monotonicity of \(X\).

A rough finite-difference recovery is

$$
V_{i+1} \approx \frac{X_{i+1}-X_i}{\Delta t}.
$$

### Pseudocode

```text
Algorithm iVi-IntegratedVolterra
Input: grid, integrated kernel quantities, Volterra square-root parameters
1. Initialize X_0 = 0.
2. For each time step i:
      Compute the local integrated-kernel coefficients required by the scheme.
      Match them to an Inverse-Gaussian update law for DeltaX_i.
      Sample DeltaX_i >= 0 from the corresponding IG distribution.
      Set X_{i+1} = X_i + DeltaX_i.
      Recover a local variance proxy V_{i+1} from DeltaX_i / dt.
      Update the stock if needed.
3. Return X_i and the induced V_i.
```

### Remarks

- This is recent and specialized, but it belongs in a comprehensive list.
- Its appeal is not speed alone; it is designed around positivity and singular kernels.
- It appears especially strong when \(H\downarrow -1/2\).

---

## 3.8 Splitting schemes and higher-order Markovian discretizations

### Idea

Once a rough model has been lifted to a finite-dimensional Markov system, one may apply splitting or high-order schemes to the Markovian approximation instead of plain Euler.

For example, if the lifted generator decomposes as

$$
\mathcal L = \mathcal L_0 + \mathcal L_1,
$$

one may use Lie–Trotter or Strang splitting:

$$
e^{\Delta t \mathcal L}
\approx
 e^{\frac{\Delta t}{2}\mathcal L_0}
 e^{\Delta t\mathcal L_1}
 e^{\frac{\Delta t}{2}\mathcal L_0}.
$$

### Pseudocode

```text
Algorithm SplitMarkovianLift
1. Lift the rough model to an N-factor Markov system.
2. Split the dynamics into subflows that are easy to simulate.
3. For each time step:
      apply half-step of flow A
      apply full-step of flow B
      apply half-step of flow A
4. Reconstruct variance and stock.
```

### Remarks

- This is not yet the default production method for rough Heston, but it is part of the broader simulation landscape.
- It is more a **discretization layer for the Markovian lift** than a distinct rough-vol model simulator.

---

## 4. Which methods are missing if one only names “Friz, Horvath, Bayer”?

A truly broad list should include all of the following categories.

### 4.1 Gaussian rough-volatility methods

1. Dense covariance factorization (Cholesky / PCA / SVD).
2. FFT exact Gaussian simulation when stationary embedding is available (Davies–Harte / circulant embedding).
3. Direct convolution / Riemann-sum discretization of the Volterra driver.
4. The BLP hybrid scheme.
5. rDonsker and fractional-tree approximations.
6. Variance-reduction wrappers (turbocharging, control variates, antithetics).
7. QMC / sparse-grid integration layers.
8. MLMC for path-dependent or integral functionals.

### 4.2 Affine Volterra / rough Heston methods

1. Direct Volterra Euler discretization.
2. Integrated-variance Euler discretization.
3. AFV Riemann-sum history-based schemes.
4. HQE / QE-type moment-matching schemes.
5. Multi-factor Markovian approximation + Euler.
6. Multi-factor Markovian approximation + weak moment-matching schemes.
7. Multi-factor Markovian approximation + splitting / higher-order schemes.
8. Integrated-process positivity-preserving methods such as iVi.

---

## 5. Practical comparison

| Category | Main idea | Typical cost | Positivity handling | Best use case |
|---|---|---:|---|---|
| Exact Gaussian factorization | Simulate full Gaussian vector exactly | high | automatic for lognormal variance | benchmark for rBergomi |
| Davies–Harte / circulant | FFT exact Gaussian increments | low to moderate | automatic for lognormal variance | large Gaussian grids |
| Direct Gaussian convolution | literal Volterra sum | quadratic | automatic for lognormal variance | baseline / pedagogy |
| BLP hybrid | singular near field + FFT far field | \(O(n\log n)\) | automatic for lognormal variance | standard rough Bergomi MC |
| rDonsker | random-walk approximation of Volterra driver | moderate | automatic for lognormal variance | weak convergence, trees |
| Volterra Euler | direct rough Heston discretization | \(O(n^2)\) | clipping/truncation often used | baseline rough Heston paths |
| Integrated Euler | discretize integrated variance form | \(O(n^2)\) | somewhat indirect | integrated-variance payoffs |
| HQE | match conditional moments | \(O(n^2)\) | built into one-step law | practical rough Heston |
| Markovian lift + Euler | sum-of-exponentials kernel approximation | \(O(Nn)\) | via Markovian variance proxy | general-purpose rough Heston |
| Markovian weak scheme | moment-matching on lifted model | \(O(Nn)\) | built into one-step law | option pricing, Bermudans |
| iVi | IG update for integrated process | problem-dependent | designed to preserve monotonicity / positivity | singular kernels, very rough regime |

---

## 6. Minimal implementation templates by model class

## 6.1 Rough Bergomi template

```text
Choose Gaussian engine:
    (A) dense Cholesky/PCA
    (B) Davies-Harte / circulant embedding
    (C) BLP hybrid
    (D) rDonsker
Generate the Volterra Gaussian path X_i approximating Wtilde_{t_i}
Set V_i = xi0(t_i) * exp(eta * X_i - 0.5 * eta^2 * t_i^(2H))
Generate correlated stock increment dZ_i
Update logS_{i+1} = logS_i - 0.5 * V_i * dt + sqrt(V_i) * dZ_i
Apply optional variance reduction / QMC / MLMC
```

## 6.2 Rough Heston template

```text
Choose variance engine:
    (A) direct Volterra Euler
    (B) integrated-variance Euler
    (C) HQE
    (D) Markovian lift + Euler
    (E) Markovian lift + weak scheme
    (F) iVi / integrated-process scheme
Generate V_i (or X_i then recover V_i)
Generate correlated stock increment dW1_i
Update log stock with current variance proxy
Apply optional Longstaff-Schwartz if the lifted model is Markovian and the payoff has early exercise
```

---

## 7. Implementation warnings

1. **Near-singularity handling matters.** Naive Riemann sums for kernels \(u^{H-1/2}\) are often much less accurate than hybrid or Markovian-lift methods.
2. **Do not confuse path generation with variance reduction.** Turbocharging, QMC, sparse grids, and MLMC accelerate Monte Carlo but do not define new rough-vol paths.
3. **Rough Heston positivity is nontrivial.** Simple Euler on \(V\) often needs clipping/truncation and may still perform poorly.
4. **Markovian lifts change the problem class.** After a sum-of-exponentials approximation, standard Markovian tools become available, including regression-based Bermudan pricing.
5. **Model class matters.** Methods that are natural for Gaussian rough-vol models need not transfer directly to affine Volterra models.

---

## 8. References to the main simulation lines

- Bayer, Friz, Gatheral: rough Bergomi / pricing under rough volatility; covariance-based Gaussian simulation benchmark.
- Bennedsen, Lunde, Pakkanen: hybrid scheme for Brownian semistationary processes.
- Horvath, Jacquier, Muguruza, Søjmark: rDonsker approximation and fractional trees.
- McCrickerd, Pakkanen: turbocharging Monte Carlo for rough Bergomi.
- Gatheral, Keller-Ressel, and subsequent AFV simulation work: affine forward variance framework and HQE-style simulation.
- Richard, Tan, Yang: discrete-time simulation of rough Heston via Euler-type schemes.
- Abi Jaber, El Euch: multi-factor / Markovian approximation of rough volatility models.
- Bayer, Breneis: weak Markovian approximations and efficient weak simulation for rough Heston.
- Abi Jaber, Attal: iVi Inverse-Gaussian scheme for integrated Volterra square-root / Volterra Heston models.

---

## 9. Source notes

This note was compiled using the following core sources:

- Bayer, Friz, Gatheral, *Pricing under rough volatility*.
- Bennedsen, Lunde, Pakkanen, *Hybrid scheme for Brownian semistationary processes*.
- Horvath, Jacquier, Muguruza, Søjmark, *Functional central limit theorems for rough volatility*.
- Abi Jaber, El Euch, *Multi-factor approximation of rough volatility models*.
- Richard, Tan, Yang, *On the discrete-time simulation of the rough Heston model*.
- Bayer, Breneis, *Weak Markovian Approximations of Rough Heston* and *Efficient option pricing in the rough Heston model using weak simulation schemes*.
- Abi Jaber, Attal, *Simulating integrated Volterra square-root processes and Volterra Heston models via Inverse Gaussian*.
- McCrickerd, Pakkanen, *Turbocharging Monte Carlo pricing for the rough Bergomi model*.
