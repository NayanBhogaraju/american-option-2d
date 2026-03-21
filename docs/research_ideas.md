# Research Directions: Extending the Monotone Integration Scheme

This document outlines concrete research project ideas building on the
Zhou & Dang (2025) monotone integration framework for two-asset American
options under jump-diffusion.

---

## 1. Extend to the 2-D Kou Jump-Diffusion Model

**Difficulty: Hard | Novelty: High**

The paper explicitly calls this out as future work (Section 3.2). The Kou model
uses double-exponential jumps instead of log-normal, which creates a
combinatorial explosion of piecewise-exponential regions when summing k i.i.d.
jump vectors.

**Approach A — Neural network approximation of the jump PDF:**
- Train a single-hidden-layer NN with Gaussian activations to approximate the
  2-D Kou joint PDF as a Gaussian mixture
- This makes each term in the Green's function series a Gaussian–Gaussian
  convolution (tractable)
- Non-negativity can be enforced during training
- The NN needs to be trained only once (independent of Δτ)

**Approach B — Direct computation:**
- Enumerate the 4^k piecewise regions for k jumps
- Use exponential–Gaussian special functions (generalizing the 1-D Hh family)
- Likely impractical for large k, but could work with small K_ε

**Deliverables:** Implementation, convergence study, comparison with the Merton
results from this paper.


## 2. Higher-Order Time Accuracy via Penalty Methods

**Difficulty: Medium | Novelty: Medium-High**

The current scheme is first-order in time due to the explicit `max(u, payoff)`
enforcement of the early exercise constraint. The paper suggests (Section 4.3)
that a penalty formulation could achieve higher-order accuracy.

**Approach:**
- Replace `max(u, payoff)` with a penalty term: solve
  `u_τ - Lu - Ju + ρ·max(payoff - u, 0) = 0` where ρ → ∞
- Use the Green's function convolution for the L + J part
- Iterate (fixed-point or Newton) within each timestep to handle the penalty
- Analyze whether monotonicity is preserved through the iterations

**Key question:** Can you achieve second-order time convergence while maintaining
the monotonicity guarantee?


## 3. GPU-Accelerated Implementation

**Difficulty: Medium | Novelty: Medium**

The FFT-based convolution is embarrassingly parallelizable. Ghosh & Mishra (2022)
[34] already demonstrated GPU acceleration for finite difference methods on this
problem.

**Approach:**
- Implement the scheme in CuPy or JAX (drop-in NumPy replacements with GPU support)
- The key bottleneck is the 2-D FFT, which maps directly to cuFFT
- Profile and compare wall-clock times against CPU implementation
- Push to refinement level 4 (N=J=4096) which is impractical on CPU

**Deliverables:** GPU implementation, speedup benchmarks, ability to run
convergence studies at the paper's full resolution (2^12 × 2^12).


## 4. Extension to Stochastic Volatility + Jumps (SVJ)

**Difficulty: Hard | Novelty: High**

The current model has constant volatilities σ_x, σ_y. A natural extension is
to allow stochastic volatility (e.g., Heston or SABR) in addition to jumps.

**Challenge:** The Green's function no longer has a closed-form Fourier transform
(the characteristic function of the SVJ model involves the Heston term).

**Possible approaches:**
- Use the Fourier neural network approach from Du & Dang (2025) [27] to
  approximate the unknown Green's function
- Factorize: handle diffusion+jumps via the current method, handle stochastic
  vol via a separate splitting step
- Dimension reduction: for each vol state, compute a conditional Green's function


## 5. Two-Asset Allocation (Continuous Rebalancing)

**Difficulty: Hard | Novelty: Very High**

The paper's methodology was originally motivated by asset allocation problems
(Section 1, referencing [67]). Extending to fully 2-D continuous rebalancing is
explicitly listed as future work.

**Problem setup:**
- Two risky assets (stock + bond, or two stocks) + risk-free asset
- Investor rebalances continuously to maximize utility
- This gives a 2-D HJB equation (not a variational inequality)
- The control is the portfolio weight, optimized at each (x, y, t)

**Approach:**
- Start from discrete rebalancing (where the Green's function approach
  transfers directly)
- Take the limit as rebalancing interval → 0
- Use policy iteration or Howard's algorithm for the HJB equation
- The Green's function convolution handles the time advancement


## 6. Convergence Rate Analysis

**Difficulty: Medium | Novelty: Medium**

The paper proves convergence but does not establish a convergence *rate* for the
full scheme (only that it's first-order in time due to the explicit max).

**Questions to investigate:**
- What is the spatial convergence rate? (The trapezoidal rule gives O(h²) for
  smooth integrands, but the payoff has a kink)
- How does the convergence rate depend on the jump intensity λ?
- Can Richardson extrapolation improve the effective convergence order?
- Compare empirical rates across Cases I, II, III to understand the
  dependence on model parameters

**Deliverables:** Numerical convergence rate study, Richardson extrapolation
experiments, error analysis as a function of λ.


## 7. Comparison with Monte Carlo and Machine Learning Methods

**Difficulty: Medium | Novelty: Medium**

Use the prices from this method as ground truth to benchmark:

**Monte Carlo approaches:**
- Longstaff-Schwartz (least-squares Monte Carlo) for 2-asset American options
- Compare accuracy vs. runtime at various confidence levels
- The jump-diffusion adds complexity to MC path simulation

**Machine learning approaches:**
- Deep BSDE methods (Han, Jentzen, E, 2018)
- Physics-informed neural networks (PINNs) for the variational inequality
- Neural optimal stopping (Becker, Cheridito, Jentzen, 2019)
- Use the MI prices as training targets or validation benchmarks

**Deliverables:** Comprehensive benchmark paper comparing PDE, MC, and ML methods
for this specific problem class.


## 8. Bermudan-to-American Convergence Study

**Difficulty: Low-Medium | Novelty: Low-Medium**

The paper mentions (Section 1.1) that Bermudan options can approximate American
options as the exercise interval shrinks. The COS method [60] has been applied
to 2-D Bermudan options but may produce negative prices.

**Study:**
- Price Bermudan options (exercise only at discrete dates) using both the MI
  method and the COS method
- Track convergence as the number of exercise dates increases
- Document cases where the COS method produces negative prices
- Quantify the monotonicity violation in COS vs. the guaranteed monotonicity of MI

This is a good "first project" — relatively easy to implement and produces
a clear, publishable comparison.


## 9. Sensitivity Analysis and Greeks

**Difficulty: Low-Medium | Novelty: Low**

Compute option sensitivities (Greeks) via finite differences on the pricing grid:

- **Delta** (∂V/∂X, ∂V/∂Y): available directly from the grid
- **Gamma** (∂²V/∂X², etc.): second-order FD on the solution
- **Cross-gamma** (∂²V/∂X∂Y): the mixed derivative
- **Vega** (∂V/∂σ): re-run with perturbed σ
- **Theta** (∂V/∂t): from the time-stepping
- **Jump sensitivity** (∂V/∂λ, ∂V/∂μ̃): re-run with perturbed jump params

**Key insight:** Because the scheme is monotone, the computed Greeks should be
well-behaved (no oscillations). Compare with non-monotone methods.


## 10. Real Market Calibration

**Difficulty: Medium-Hard | Novelty: Medium**

Calibrate the two-asset Merton model to real market data:

- Choose two correlated assets (e.g., SPY + QQQ, or two sector ETFs)
- Obtain market prices of two-asset options (basket options, worst-of options)
  from OTC desks or structured product data
- Calibrate the 11 model parameters (σ_x, σ_y, ρ, λ, μ̃_x, μ̃_y, σ̃_x, σ̃_y, ρ̂, r, T)
  to minimize the pricing error
- Use the calibrated model to price exotic two-asset Americans

**Challenge:** The calibration itself is an optimization problem where each
objective function evaluation requires running the pricer. GPU acceleration
(idea #3) would be very helpful here.

---

## Recommended Starting Points

| Your background | Start with | Then try |
|----------------|------------|----------|
| Strong math/PDE | #2 (penalty methods) | #6 (convergence rates) |
| Strong coding | #3 (GPU) | #7 (MC/ML comparison) |
| Finance focus | #9 (Greeks) | #10 (calibration) |
| ML background | #7 (benchmarking) | #1 (NN for Kou) or #4 (SVJ) |
| First research project | #8 (Bermudan study) | #6 (convergence rates) |
