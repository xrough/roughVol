# Rough Volatility Models
## 1. Stochastic Volatility Models

### 1.1 The Limitations of Black-Scholes
The starting point of modern option pricing is the Black-Scholes (BS) model. While revolutionary, it relies on a strong assumption that contradicts market reality.

**The Constant Volatility Assumption**
In the BS world, the underlying asset price $S_t$ follows a Geometric Brownian Motion:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Here, $\sigma$ (volatility) is assumed to be a **constant**.

* **The Implication:** If the BS model were perfectly correct, the implied volatility derived from market prices of options on the same asset should be identical, regardless of the strike price ($K$) or time to maturity ($T$).
* **The Reality:** This is rarely observed in real markets.

---

### 1.2 Market Reality: The Volatility Smile and Surface
Real-world data "punches back" against the constant volatility assumption. When we invert the BS formula using market prices to solve for $\sigma$, we find that $\sigma$ is not constant.

#### 1.2.1 The Volatility Smile
If we plot Implied Volatility (y-axis) against Strike Price $K$ (x-axis) for a fixed maturity:
* We do not see a flat line.
* We typically see a curve where volatility is lower for At-the-Money (ATM) options and higher for In-the-Money (ITM) or Out-of-the-Money (OTM) options.
* This shape resembles a smile, hence the term **Volatility Smile**.

#### 1.2.2 The Volatility Surface
The "Smile" only accounts for the strike dimension. If we add **Time to Maturity ($T$)** as a third dimension, we get the **Volatility Surface**.
* **X-axis:** Strike ($K$) or Moneyness.
* **Y-axis:** Time to Maturity ($T$).
* **Z-axis:** Implied Volatility ($\sigma$).

**Visual Intuition:** It looks like a "flying carpet." The near-term part of the surface is often jagged and steep (high skew), while the long-term part is usually flatter and smoother.

### 1.2.3 Realized vs. Implied Volatility
It is crucial to distinguish between two concepts:
1. **Realized Volatility (RV):** "Looking Backward." It is the standard deviation of historical returns over a past period (e.g., past 30 days). It is a statistical measurement of what *actually happened*.
2. **Implied Volatility (IV):** "Looking Forward." It is backed out from option prices. It represents the market's expectation of future volatility (and the price of volatility risk).

---

### 1.3 Measuring Realized Volatility
How do we calculate the volatility of a stock given a set of historical prices $\{P_0, P_1, ..., P_n\}$?

**Step 1: Log Returns**
Calculate the natural log returns of the price series:

$$r_i = \ln \left( \frac{P_i}{P_{i-1}} \right)$$

**Step 2: Standard Deviation**
Calculate the standard deviation of these returns. First find the mean $\bar{r}$, then:

$$\text{std} = \sqrt{\frac{\sum(r_i - \bar{r})^2}{n-1}}$$

**Step 3: Annualization**
Volatility is typically quoted in annualized terms. Standard deviation scales with the square root of time. If using daily data (assuming 252 trading days/year):

$$\text{Volatility}_{\text{annualized}} = \text{std} \times \sqrt{252}$$

---

### 1.4 The Stochastic Volatility Framework
To address the Volatility Smile, we move from deterministic volatility to **Stochastic Volatility (SV)**. We follow the framework closely related to Wilmott (2000).

We assume the stock price $S$ and its **variance** $v$ (where volatility $\sigma = \sqrt{v}$) satisfy the following system of Stochastic Differential Equations (SDEs):

#### The System of SDEs

1. **Asset Process:**
   $$dS_t = \mu_t S_t dt + \sqrt{v_t} S_t dZ_1 \quad (1.1)$$

2. **Variance Process:**
   $$dv_t = \alpha(S_t, v_t, t)dt + \eta \beta(S_t, v_t, t)\sqrt{v_t} dZ_2 \quad (1.2)$$

**Key Parameters:**
* $\mu_t$: Instantaneous drift of the stock.
* $\alpha(...)$: Drift of the variance process (e.g., mean reversion).
* $\eta$: The "volatility of volatility" (vol-of-vol).
* **Correlation ($\rho$):** The Wiener processes are correlated:
  $$\langle dZ_1, dZ_2 \rangle = \rho dt.$$

---

### 1.5 Hedging and Deriving the PDE
This is the core of the financial engineering approach. In Black-Scholes, we only have risk from $S$. In SV models, we have risk from both $S$ and $v$.

#### 1.5.1 The Hedging Problem
* **Black-Scholes:** Randomness comes from $dZ_1$ only. We can hedge an option $V$ simply by trading the underlying stock $S$.
* **Stochastic Volatility:** Randomness comes from $dZ_1$ and $dZ_2$.
    * We can hedge $S$-risk using the stock.
    * **Problem:** Volatility is not a traded asset. We cannot "buy variance" directly to hedge the $dv$ term.
    * **Solution:** We must use **another option** (a benchmark instrument) to hedge the volatility risk.

#### 1.5.2 Constructing the Portfolio
We set up a portfolio $\Pi$ containing:
1. One unit of the Option being priced: $V(S, v, t)$.
2. $-\Delta$ units of the Stock $S$.
3. $-\Delta_1$ units of another asset $V_1$ (e.g., a liquid option) to hedge volatility risk.

$$\Pi = V - \Delta S - \Delta_1 V_1$$

#### 1.5.3 Dynamics of the Portfolio
The change in the portfolio value $d\Pi$ is derived using **Itô's Lemma** (details omitted for brevity). The result groups terms by $dt, dS,$ and $dv$:

$$d\Pi = \{ \dots \} dt + \left( \frac{\partial V}{\partial S} - \Delta_1 \frac{\partial V_1}{\partial S} - \Delta \right) dS + \left( \frac{\partial V}{\partial v} - \Delta_1 \frac{\partial V_1}{\partial v} \right) dv$$

#### 1.5.4 Making the Portfolio Risk-Free
To make the portfolio instantaneously risk-free, we must eliminate all stochastic terms ($dS$ and $dv$).

1. **Eliminate $dv$ (Vega Risk):**
   $$\frac{\partial V}{\partial v} - \Delta_1 \frac{\partial V_1}{\partial v} = 0 \implies \Delta_1 = \frac{\partial V / \partial v}{\partial V_1 / \partial v}$$
   *This tells us how many units of the hedging option $V_1$ we need.*

2. **Eliminate $dS$ (Delta Risk):**
   $$\frac{\partial V}{\partial S} - \Delta_1 \frac{\partial V_1}{\partial S} - \Delta = 0$$
   *This tells us the net Delta position required in the stock.*

#### 1.5.5 The General PDE
Once the random terms are removed, the portfolio earns the risk-free rate $r$:
$$d\Pi = r \Pi dt$$

Substituting the expressions back, we find that the equation holds for any derivative only if both sides equal a function of the independent variables. This introduces the **Market Price of Volatility Risk**, often denoted as part of the term $(\alpha - \phi \beta \sqrt{v})$.

Rearranging gives the fundamental **Partial Differential Equation for Stochastic Volatility**:

$$\frac{\partial V}{\partial t} + \frac{1}{2} v S^2 \frac{\partial^2 V}{\partial S^2} + \rho \eta v \beta S \frac{\partial^2 V}{\partial v \partial S} + \frac{1}{2} \eta^2 v \beta^2 \frac{\partial^2 V}{\partial v^2} + r S \frac{\partial V}{\partial S} - rV = -(\alpha - \phi \beta \sqrt{v}) \frac{\partial V}{\partial v}$$

**Interpretation of Terms:**
* $\frac{\partial V}{\partial t}$: Time decay (**Theta**).
* $\frac{1}{2}vS^2 \frac{\partial^2 V}{\partial S^2}$: **Gamma** (standard Black-Scholes term, but with stochastic $v$).
* $\rho \eta v \beta S \frac{\partial^2 V}{\partial v \partial S}$: **Vanna**. The effect of correlation between price and vol changes.
* $\frac{1}{2}\eta^2 v \beta^2 \frac{\partial^2 V}{\partial v^2}$: **Volga** (Vol of Vol). The convexity with respect to variance.
* **RHS Term:** Represents the drift of the variance, adjusted for the market price of risk.

By acknowledging that volatility is stochastic, we move from a world where we only hedge **Delta** (using stock) to a world where we must hedge both **Delta** (Direction) and **Vega** (Volatility). 

## 2. Heston Model: Dynamics and PDE Derivation

### 2.1 Physical Dynamics of the Model (SDEs)
The core idea of the Heston model is to abandon the assumption of "constant volatility" in the Black-Scholes model and instead treat volatility itself as a stochastic process.

Under the **Physical Measure ($\mathbb{P}$)**, the observed asset price $S_t$ and variance $v_t$ follow the stochastic differential equations below:

#### Asset Price Process
$$dS_t = \mu_t S_t dt + \sqrt{v_t} S_t dZ_1$$

* $\mu_t$: The real expected return of the asset (Physical Drift).
* $\sqrt{v_t}$: Instantaneous volatility.
* $dZ_1$: The Wiener process (Brownian motion) driving the stock price.

#### Variance Process (CIR Process)
$$dv_t = -\lambda(v_t - \bar{v})dt + \eta \sqrt{v_t} dZ_2$$

* **$-\lambda(v_t - \bar{v})$: Mean Reversion Term.**
    * $\bar{v}$: Long-run mean variance. In the long run, variance tends to revert to this level.
    * $\lambda$: Speed of reversion. The larger $\lambda$ is, the stronger the force pulling variance back to $\bar{v}$.
    * *(Note: This uses the notation from your image, equivalent to $\lambda(\bar{v} - v_t)$)*
* **$\eta \sqrt{v_t}$: Volatility of Volatility (Vol of Vol).**
    * $\eta$ controls the severity of the oscillation of the variance itself.
    * $\sqrt{v_t}$ ensures that variance remains non-negative (Feller condition).

#### Correlation
$$\langle dZ_1, dZ_2 \rangle = \rho dt$$

* $\rho$: The correlation coefficient between asset price returns and variance changes.
* In equity markets, typically $\rho < 0$ (**Leverage Effect/Fear Effect**), meaning volatility tends to rise when stock prices fall.

---

### 2.2 Derivation of the Pricing Equation (Vega Hedging & PDE)
Since there are two sources of randomness ($dZ_1, dZ_2$) in the market, the risk cannot be fully hedged using only the underlying asset $S$ (**Incomplete Market**). We need to introduce another asset sensitive to volatility (such as a benchmark option $U$) to perform **Vega Hedging**.

#### Constructing the Hedging Portfolio
Construct a portfolio $\Pi$ consisting of the option to be priced $V$, the underlying asset $S$, and the benchmark option $U$:

$$\Pi = V - \Delta S - \psi U$$

The goal is to choose appropriate $\Delta$ and $\psi$ to eliminate all stochastic risks.

#### Applying Itô's Lemma
The change in portfolio value $d\Pi$ contains a deterministic drift term and stochastic diffusion terms:

$$d\Pi = (\dots)dt + \text{Risk}_S \cdot dZ_1 + \text{Risk}_v \cdot dZ_2$$

Where:
* The coefficient for the **Stock Price Risk term ($dZ_1$)** is: $S\sqrt{v}(\frac{\partial V}{\partial S} - \Delta - \psi \frac{\partial U}{\partial S})$
* The coefficient for the **Volatility Risk term ($dZ_2$)** is: $\eta\sqrt{v}(\frac{\partial V}{\partial v} - \psi \frac{\partial U}{\partial v})$

#### Eliminating Risk
Set the coefficients of both risk terms to 0:
1.  **Vega Hedging:** $\psi = \frac{\partial V / \partial v}{\partial U / \partial v}$ (Eliminates $dZ_2$)
2.  **Delta Hedging:** $\Delta = \frac{\partial V}{\partial S} - \psi \frac{\partial U}{\partial S}$ (Eliminates $dZ_1$)

At this point, the portfolio $\Pi$ becomes a risk-free asset.

#### No-Arbitrage Principle & Market Price of Risk
According to the No-Arbitrage Principle, the return of a risk-free portfolio must equal the risk-free rate $r$:
$$d\Pi = r\Pi dt$$

Substituting this relationship and rearranging, we find that for any derivative ($V$ or $U$), the ratio of its **excess return to volatility risk exposure (Vega)** must be equal:

$$\frac{\mathbb{E}[dV] - rV dt}{\partial V / \partial v} = \frac{\mathbb{E}[dU] - rU dt}{\partial U / \partial v} = \text{Constant} \times \eta\sqrt{v}$$

We define the term on the right as: **The Market Price of Volatility Risk ($\phi$)**. For mathematical convenience, it is usually assumed that this price takes the form $\phi(S, v, t) \propto \sqrt{v}$.

#### Deriving the Final PDE
Substituting the above relationship back into the equation for $V$, the physical drift term $-\lambda(v - \bar{v})$ will be modified.

* **Original Physical Drift:** $-\lambda(v - \bar{v})$
* **Drift after deducting Risk Premium:** $-\lambda(v - \bar{v}) - \text{RiskPremium}$

To obtain the PDE, we perform parameter redefinition (**Risk Neutralization**): Assume the modified drift term retains the mean-reverting form, denoted as $\lambda^*(v - \bar{v}^*)$. We can make a simplifying assumption: **Directly set the market price of risk to 0 ($\phi = 0$)**, or in other words, **assume that $\lambda$ and $\bar{v}$ in the equation are already parameters under the Risk-Neutral Measure.**

**The final Heston PDE obtained:**
$$\frac{\partial V}{\partial t} + \frac{1}{2} v S^2 \frac{\partial^2 V}{\partial S^2} + \rho \eta v S \frac{\partial^2 V}{\partial v \partial S} + \frac{1}{2} \eta^2 v \frac{\partial^2 V}{\partial v^2} + r S \frac{\partial V}{\partial S} - r V = \lambda(v - \bar{v}) \frac{\partial V}{\partial v}$$

*(Note: We typically move the term on the right to the left side to treat it as a drift term)*

### 2.3 Deep Dive into Market Price of Risk ($\phi$)

#### Physical Meaning
* **Origin:** Since $dZ_2$ risk cannot be hedged via $S$, investors are forced to bear volatility risk and thus demand an extra expected return.
* **Essence:** $\phi$ represents the market's level of fear regarding uncertainty (**Risk Aversion**).
* **Manifestation:** When the market is in panic, $\phi$ increases, causing option prices to be more expensive than prices calculated purely from physical probabilities.

#### Measurement and Setting
* Since $\phi$ is the result of subjective behavior, we cannot measure it directly with a ruler.
* **The True Meaning of "Set $\phi$ to zero":** This does not mean the market has no risk aversion; rather, it is a **parameterization strategy**. When applying the model, we do not statistically estimate physical parameters and then calculate $\phi$; instead, we **calibrate directly using market data (option prices)**.
* The calibrated $\lambda$ and $\bar{v}$ are parameters that have already been **"distorted by market sentiment"** (i.e., Risk-Neutral parameters). Therefore, in the pricing formula, we do not need to explicitly write a $\phi$ term, because the impact of $\phi$ is already embedded in the calibrated $\lambda$.

---

### 2.4 Semi-Analytical Solution of the Model
The greatest contribution of the Heston model is that, despite being a stochastic volatility model, a semi-analytical solution (Semi-closed form solution) exists for European options.

#### Form of the Solution (Ansatz)
Similar to the Black-Scholes formula, the call option price is conjectured to be:

$$C(S, v, t) = SP_1 - Ke^{-r\tau}P_2$$

* $SP_1$: The expected present value of receiving the stock.
* $Ke^{-r\tau}P_2$: The expected present value of paying the strike price.
* $P_1, P_2$: Two probability terms (probabilities of the option being in-the-money under different measures).

#### Characteristic Function and Integration
The probabilities $P_j (j = 1, 2)$ cannot be solved directly but can be obtained via the inverse Fourier transform of their **characteristic functions** $f_j$:

$$P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \text{Re} \left[ \frac{e^{-i\phi \ln K} f_j(x, v, \tau; \phi)}{i\phi} \right] d\phi$$

Where the characteristic function is assumed to have an **Exponential Affine Form**:

$$f_j(x, v, \tau; \phi) = \exp(C_j(\tau, \phi) + D_j(\tau, \phi)v + i\phi x)$$

#### Rigorous Expression of Coefficients $C_j$ and $D_j$
By substituting into the PDE, we transform the problem into solving **Riccati** ordinary differential equations. Define auxiliary variables:

* $u_1 = 1/2, \quad u_2 = -1/2$
* $b_1 = \lambda - \rho \eta, \quad b_2 = \lambda$
* $d_j = \sqrt{(\rho \eta \phi i - b_j)^2 - \eta^2(2u_j \phi i - \phi^2)}$
* $g_j = \frac{b_j - \rho \eta \phi i + d_j}{b_j - \rho \eta \phi i - d_j}$

Solving yields:

$$D_j(\tau, \phi) = \frac{b_j - \rho \eta \phi i + d_j}{\eta^2} \left( \frac{1 - e^{d_j \tau}}{1 - g_j e^{d_j \tau}} \right)$$

$$C_j(\tau, \phi) = r \phi i \tau + \frac{\lambda \bar{v}}{\eta^2} \left[ (b_j - \rho \eta \phi i + d_j)\tau - 2 \ln \left( \frac{1 - g_j e^{d_j \tau}}{1 - g_j} \right) \right]$$

---

### 2.5 Volatility Smile & Skew
The Heston model explains why the Black-Scholes implied volatility curve is not a straight line via parameters $\rho$ and $\eta$.

According to small-time asymptotic expansion, the implied volatility near ATM is approximately:

$$\sigma_{imp}(k) \approx \sigma_0 + \underbrace{\frac{\rho \eta}{4 \sigma_0} k}_{\text{Slope (Skew)}} + \underbrace{\frac{\eta^2(2 - 3\rho^2)}{24 \sigma_0^3} k^2}_{\text{Curvature (Smile)}}$$

*(Where $k$ is the log-strike price)*

#### $\rho$ (Correlation) and Skew
* **Role:** Controls the **Slope** of the curve.
* **Phenomenon:** Since $\rho < 0$ in equity markets (**Leverage/Panic Effect**), the slope is negative.
* **Explanation:** When stock prices fall, volatility rises, making low-strike Put options extremely expensive. This elevates the implied volatility on the left side, forming a **Smirk (downward slope to the right)**.

#### $\eta$ (Vol of Vol) and Smile
* **Role:** Controls the **Curvature/Convexity** of the curve.
* **Phenomenon:** $\eta^2$ is always positive, causing both ends of the curve to turn upwards.
* **Explanation:** The larger $\eta$ is, the more severe the variance oscillation, and the higher the probability of extreme market moves (**Fat-tailed distribution**). This makes deep ITM and deep OTM options more expensive than predicted by the BS model, forming a **Smile**.

## 3. Rough Heston Model: From Microstructure to Macro Pricing

### 3.1 Model Dynamics under Physical Measure ($\mathbb{P}$)
The core discovery of the Rough Heston model is that the trajectory of volatility is not a smooth Brownian motion, but a process with "rough" characteristics. Under the **Physical Measure ($\mathbb{P}$)**, preserving the real drift term $\mu_t$, the dynamics of the model are as follows:

#### Asset Price Process (Standard SDE)
The asset price is still driven by a standard Brownian motion:

$$dS_t = \mu_t S_t dt + \sqrt{v_t} S_t dZ_1(t)$$

* **$\mu_t$**: The real expected return of the asset.
* **$dZ_1$**: Standard Brownian motion.

#### Rough Variance Process (Volterra Integral Equation)
The variance process is no longer a simple differential equation, but a path-dependent **Volterra Integral Equation**:

$$v_t = v_0 + \frac{1}{\Gamma(H+\frac{1}{2})} \int_0^t (t-s)^{H-\frac{1}{2}} \left[ -\lambda(v_s - \bar{v}) ds + \eta \sqrt{v_s} dW(s) \right]$$

* **Integral Kernel $K(t - s) \propto (t - s)^{H - \frac{1}{2}}$**: This is the soul of the model.
    * As $s \to t$ (the immediate past), the weight tends to infinity. This means current volatility is extremely sensitive to "very short-term history."
* **$dW(s)$**: Standard Brownian motion is used here.
* **Correlation**: $\langle dZ_1, dW \rangle = \rho dt$.

#### Heuristic SDE Form
Although mathematically non-rigorous (because fractional Brownian motion is not a semimartingale), for physical intuition, we can understand it as an SDE driven by **Fractional Brownian Motion (fBm)**:

$$dv_t \approx -\lambda(v_t - \bar{v})dt + \eta \sqrt{v_t} \circ dZ^H(t)$$

Where $Z^H(t)$ is a fractional Brownian motion with Hurst exponent $H$.

---

### 3.2 Fractional Brownian Motion (fBm)
The foundation of Rough Heston is fractional Brownian motion. As a Gaussian process, it is uniquely determined by its mean and covariance function.

#### Strict Definition (Covariance Definition)
Let $Z_t^H$ be a standard fractional Brownian motion satisfying:

1. **Mean**: $\mathbb{E}[Z_t^H] = 0$
2. **Variance**: $\mathbb{E}[(Z_t^H)^2] = |t|^{2H}$
3. **Covariance Structure (Core Formula)**:

$$\mathbb{E}[Z_t^H Z_s^H] = \frac{1}{2} (|t|^{2H} + |s|^{2H} - |t - s|^{2H})$$

#### Physical Meaning of Hurst Exponent ($H$)
* **$H = 1/2$ (Standard Heston)**: The covariance degenerates to $\min(t, s)$, which is standard Brownian motion. Increments are independent.
* **$H > 1/2$ (Smooth/Persistent)**: Increments are positively correlated, and the path is smoother than Brownian motion (Long Memory).
* **$H < 1/2$ (Rough/Anti-persistent)**: The case for **Rough Heston ($H \approx 0

---

### 3.3 Increment Correlation & Roughness
Rather than looking at the covariance formula, it is more intuitive to look at the correlation of increments over two non-overlapping small intervals.

Consider increments $\Delta Z_t$ and $\Delta Z_s$ over an extremely short time $\delta$ at two time points separated by $\tau$:

$$\mathbb{E}[\Delta Z_t \cdot \Delta Z_s] \approx H(2H-1) \cdot \tau^{2H-2} \cdot \delta^2$$

$$\text{Corr}(\Delta Z_t, \Delta Z_s) \sim H(2H - 1) \cdot \tau^{2H - 2}$$

#### Microscopic Origin of Mean Reversion ($H < 0.5$)
* The coefficient **$H(2H - 1)$ is negative**.
* **Physical Meaning**: If there was an upward jump at past time $s$, there is a tendency to jump downward at future time $t$.
* This "push and pull" at the microscopic, high-frequency level leads to the jagged characteristics of the path (**Roughness**).
* The decay term **$\tau^{2H - 2}$** (e.g., $\tau^{-1.8}$) indicates that this negative correlation decays extremely fast with distance.

---

### 3.4 From Volterra Equations to the Markovian Lift

#### The Asset Price Process

In the Rough Heston model, the asset price dynamics follow a standard Stochastic Differential Equation (SDE), similar to the classical Heston model. The asset price $S_t$ is driven by:

$$dS_t = \mu_t S_t dt + \sqrt{v_t} S_t dZ_1(t)$$

Where:

* $\mu_t$: The real expected return of the asset.

* $dZ_1(t)$: A standard Brownian motion.

* $v_t$: The instantaneous variance (squared volatility) of the asset.

#### The Rough Variance Process

The key difference—and the mathematical complexity—lies in how the variance $v_t$ evolves. Unlike the standard Heston model, which uses a simple SDE, the Rough Heston model describes variance using a path-dependent Volterra Integral Equation:

$$v_t = v_0 + \frac{1}{\Gamma(H + \frac{1}{2})} \int_0^t (t-s)^{H-\frac{1}{2}} \left[ -\lambda(v_s - \bar{v})ds + \eta \sqrt{v_s} dW(s) \right]$$

Key Components:

* The Integral Kernel $K(t-s) \propto (t-s)^{H-\frac{1}{2}}$: This is the "soul" of the model. As $s \to t$ (looking at the immediate past), the weight tends to infinity (singular kernel). This implies that current volatility is extremely sensitive to very short-term history, creating "rough" trajectories.

* Correlation: $\langle dZ_1, dW \rangle = \rho dt$.

#### Step 1: Identifying the "Culprit" (Non-Markovian Nature)

Why is the equation above difficult to solve or simulate? The problem lies in the Power Law Kernel:

$$K(t) = t^{H-\frac{1}{2}}$$

This power function prevents us from using standard SDE techniques because it is Non-Markovian. To understand this, let's compare it to an exponential function.

* The Exponential Case (Markovian): If the kernel were $e^{-\lambda(t-s)}$, we could use the property $e^{-\lambda t} \cdot e^{\lambda s}$. The term $e^{-\lambda t}$ can be factored out of the integral. The remaining part depends only on $s$. This separability allows us to write it as a differential equation (Markovian).

* The Power Law Case (Non-Markovian): For $(t-s)^{\alpha}$, you cannot factor out $t$. This means that at every moment $t$, to calculate the variance, you must "look back" at the entire history from $0$ to $t$ and re-weight it. The future depends on the entire past path, not just the current value.

#### Step 2: The Core Idea — "Disguising" the Kernel

Engineering Insight: If we cannot handle a complex function (the power law), we approximate it using a sum of simple functions that we can handle (exponentials).

Mathematically (based on the Bernstein theorem), a completely monotone function like our kernel can be represented as a weighted integral of exponentials. For engineering approximation, we use a finite sum:

$$(t-s)^{H-\frac{1}{2}} \approx \sum_{i=1}^n c_i e^{-x_i(t-s)}$$

Where:

* $c_i$: Weights.

* $x_i$: Mean Reversion Speeds.

Intuition: We are simulating a "Long Memory" process (power law decay) by superimposing a stack of "Short Memory" processes (exponential decays) with different speeds.

#### Step 3: Substitution and the "Markovian Lift"

Let's substitute our approximation back into the original Volterra integral equation. To simplify notation, we define the "Driving Term" $dY_s$:

$$dY_s = \left[ -\lambda(v_s - \bar{v})ds + \eta \sqrt{v_s} dW(s) \right]$$

Substituting the sum of exponentials:

$$v_t \approx v_0 + \int_0^t \left( \sum_{i=1}^n c_i e^{-x_i(t-s)} \right) dY_s$$

By linearity of integration, we can pull the summation outside the integral:

$$v_t \approx v_0 + \sum_{i=1}^n c_i \underbrace{\left( \int_0^t e^{-x_i(t-s)} dY_s \right)}_{\text{Define this term as } U_t^{(i)}}$$

#### Step 4: Witnessing the Miracle (Returning to Markov)

Focus on the term inside the brackets, $U_t^{(i)}$:

$$U_t^{(i)} = \int_0^t e^{-x_i(t-s)} dY_s = e^{-x_i t} \int_0^t e^{x_i s} dY_s$$

This is the exact solution form of an Ornstein-Uhlenbeck (OU) process. If we differentiate this (or apply Itô's Lemma), we find that $U_t^{(i)}$ satisfies a very simple Stochastic Differential Equation (SDE):

$$dU_t^{(i)} = -x_i U_t^{(i)} dt + dY_t$$

Expanding $dY_t$ back, we get a system of $n$ equations:

$$dU_t^{(i)} = -x_i U_t^{(i)} dt + \left( -\lambda(v_t - \bar{v})dt + \eta \sqrt{v_t} dW_t \right)$$

#### Step 5: The Final Model Construction

By performing this operation, we have transformed the "scary" path-dependent integral equation into a system of $n$ standard factors. This is called the Markovian Lift.

The System Structure:

* Total Variance: The variance is simply the weighted sum of these factors (plus initial variance):

$$v_t = v_0 + \sum_{i=1}^n c_i U_t^{(i)}$$

* Factor Dynamics: Each factor $U_t^{(i)}$ is a standard Markov process (SDE):

$$dU_t^{(i)} = -x_i U_t^{(i)} dt + \left( -\lambda(v_t - \bar{v})dt + \eta \sqrt{v_t} dW_t \right)$$

#### Mathematical Justification: The Laplace Transform Identity

How do we justify approximating $t^{H-1/2}$ with exponentials? We use a famous identity involving the Gamma function and Laplace transforms:

$$t^{\alpha - 1} = \frac{1}{\Gamma(1-\alpha)} \int_0^\infty e^{-xt} x^{-\alpha} dx$$

In Rough Heston, our power is $H - 1/2$.
Let $\alpha = H + 1/2$. (Note that $1-\alpha = 1/2 - H$).
Substituting this into the identity:

$$t^{H-\frac{1}{2}} = \frac{1}{\Gamma(\frac{1}{2}-H)} \int_0^\infty e^{-xt} \underbrace{x^{-(H+\frac{1}{2})}}_{w(x)} dx$$

Interpretation:

* Left Side: The target Power Law Kernel.

* Right Side: An integral (continuous sum) of exponentials $e^{-xt}$ weighted by a density $w(x)$.

* Task: Discretize the continuous integral $\int_0^\infty$ into a finite sum $\sum_{i=1}^n$.

#### Discretization: The Geometric Grid

To implement this in engineering/coding, we must discretize the integration variable $x$ (which corresponds to the mean reversion speed).

Because $w(x) = x^{-(H+1/2)}$ spans a huge magnitude (from near 0 to infinity), a Linear Grid will fail efficiently capturing the behavior. We must use a Geometric Grid to capture both short-term memory (large $x$) and long-term memory (small $x$).

Algorithm to Determine $x_i$ and $c_i$:

1. Define Grid Boundaries:

* Set number of factors $n$ (e.g., $n=20$).

* Set lower bound $r_{\min}$ and upper bound $r_{\max}$ (e.g., $0.0001$ to $1000$).

2. Calculate Geometric Growth Rate ($\eta$):

$$\eta = \left( \frac{r_{\max}}{r_{\min}} \right)^{\frac{1}{n-1}}$$

3. Calculate Speeds ($x_i$):

$$x_i = r_{\min} \cdot \eta^{i-1}, \quad \text{for } i = 1, \dots, n$$

4. Calculate Weights ($c_i$):
We need to assign the mass of the continuous density $w(x) = x^{-(H+\frac{1}{2})}$ to each point $x_i$.

* A. Explicit Approximation (Recommended by Jim Gatheral):
A commonly used explicit formula derived from the integral is:$$c_i = \frac{r_{\min}^{1-\alpha}(\eta^{1-\alpha}-1)}{\Gamma(1-\alpha)(1-\alpha)} \cdot \eta^{(i-1)(1-\alpha)}$$ Where $\alpha = H + 1/2$ (thus $1-\alpha = 1/2 - H$).

* B. Intuitive Numerical Integration Form:
Alternatively, we can view $c_i$ as the integral over the interval assigned to $x_i$ (denoted as "$\text{Interval}_i$"): $$c_i = \frac{1}{\Gamma(\frac{1}{2}-H)} \int_{\text{Interval}_i} x^{-(H+\frac{1}{2})} dx$$ (Note: The integration intervals are usually defined by partition points $k_i = \sqrt{x_i x_{i+1}}$ between the speeds).

---

### 3.5 Deriving the Pricing PDE (The Hedging Argument)

We successfully "lifted" the non-Markovian Rough Heston model into a high-dimensional Markovian system using $n$ factors $U^{(i)}$. Now, we derive the Partial Differential Equation (PDE) governing the price of an option $V(t, S, U^{(1)}, \dots, U^{(n)})$ in this system.

#### Step 1: Setting the Stage

To price the option, we use a replication/hedging argument. We assume a market with three tradeable assets:

1. The Underlying Asset $S_t$ (Used to hedge Delta risk): $$dS_t = \mu S_t dt + \sqrt{v_t} S_t dZ_1(t)$$

2. The Target Option $V(t, S, U^{(1)}, \dots, U^{(n)})$:
The derivative we want to price.

3. A Hedging Instrument $V_1(t, S, U^{(1)}, \dots, U^{(n)})$ (Used to hedge Vega risk):
Another liquid option on the same asset. We need this because volatility is stochastic and not directly tradeable.

#### The Factor Dynamics

Recall from Part 1 that our $n$ factors $U_t^{(i)}$ follow:


$$dU_t^{(i)} = \underbrace{[-x_i U_t^{(i)} - \lambda(v_t - \bar{v})]}_{\text{Drift}_i} dt + \eta \sqrt{v_t} dW(t)$$


Crucial Note: All factors share the same driving noise $dW(t)$. This means the "volatility risk" is driven by a single source of uncertainty, even though there are $n$ factors.

#### Step 2: Dynamics of the Options (Itô's Lemma)

To simplify the messy algebra of Itô's Lemma for a function of $n+2$ variables, we define the differential operator $\mathcal{A}$ which collects all the deterministic ($dt$) terms:

$$\mathcal{A}V = \frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2}v S^2 \frac{\partial^2 V}{\partial S^2} + \sum_{i=1}^n \text{Drift}_i \frac{\partial V}{\partial U_i} + \frac{1}{2}\eta^2 v \sum_{i,j} \frac{\partial^2 V}{\partial U_i \partial U_j} + \rho \eta v S \sum_{i=1}^n \frac{\partial^2 V}{\partial S \partial U_i}$$

Now, the dynamics of $V$ and $V_1$ can be written as "Deterministic Drift + Two Risk Sources":

For the Target Option $V$:

$$dV = (\mathcal{A}V)dt + \underbrace{\frac{\partial V}{\partial S}\sqrt{v}S dZ_1}_{\text{Delta Risk}} + \underbrace{\left(\sum_{i=1}^n \frac{\partial V}{\partial U_i}\right) \eta \sqrt{v} dW}_{\text{Vega Risk}}$$

For the Hedging Instrument $V_1$:

$$dV_1 = (\mathcal{A}V_1)dt + \frac{\partial V_1}{\partial S}\sqrt{v}S dZ_1 + \left(\sum_{i=1}^n \frac{\partial V_1}{\partial U_i}\right) \eta \sqrt{v} dW$$

Notation: Let's define the Total Vega Sensitivity as $\mathcal{V}_{sens}(V) = \sum_{i=1}^n \frac{\partial V}{\partial U_i}$.

#### Step 3: Constructing the Portfolio

We construct a portfolio $\Pi$ to eliminate all risk:

* Long 1 unit of Target Option $V$.

* Short $\Delta$ units of Underlying $S$.

* Short $\phi$ units of Hedging Instrument $V_1$.

$$\Pi = V - \Delta S - \phi V_1$$

The change in portfolio value is:

$$d\Pi = dV - \Delta dS - \phi dV_1$$

Substituting the dynamics from Step 2:

$$\begin{aligned}
d\Pi &= [\mathcal{A}V - \phi \mathcal{A}V_1 - \Delta \mu S] dt \\
     &+ \left[ \frac{\partial V}{\partial S} - \Delta - \phi \frac{\partial V_1}{\partial S} \right] \sqrt{v} S dZ_1 \quad \leftarrow \text{(1) Price Risk} \\
     &+ \left[ \mathcal{V}_{sens}(V) - \phi \mathcal{V}_{sens}(V_1) \right] \eta \sqrt{v} dW \quad \leftarrow \text{(2) Vol Risk}
\end{aligned}$$

#### Step 4: The Hedging Strategy (Risk Elimination)

To make the portfolio risk-free, the coefficients of $dZ_1$ and $dW$ must be zero.

1. Eliminate Volatility Risk (Find $\phi$):

$$\mathcal{V}_{sens}(V) - \phi \mathcal{V}_{sens}(V_1) = 0 \implies \phi = \frac{\mathcal{V}_{sens}(V)}{\mathcal{V}_{sens}(V_1)}$$

Interpretation: The ratio of hedging instruments needed is the ratio of their total sensitivities to the variance factors.

2. Eliminate Price Risk (Find $\Delta$):

$$\frac{\partial V}{\partial S} - \Delta - \phi \frac{\partial V_1}{\partial S} = 0 \implies \Delta = \frac{\partial V}{\partial S} - \phi \frac{\partial V_1}{\partial S}$$

Interpretation: Your Delta hedge is modified. It's not just $\frac{\partial V}{\partial S}$; you must subtract the Delta exposure already introduced by your volatility hedge $\phi V_1$.

#### Step 5: No Arbitrage Condition

Since the portfolio is now risk-free, it must earn the risk-free rate $r$:

$$d\Pi = r \Pi dt$$

Substituting the $dt$ terms (since random terms are 0) and the expression for $\Pi$:

$$\mathcal{A}V - \phi \mathcal{A}V_1 - \Delta \mu S = r(V - \Delta S - \phi V_1)$$

Now, we perform the Separation of Variables. We substitute $\Delta$ and $\phi$ back into this equation and rearrange terms so that everything related to $V$ is on the left, and everything related to $V_1$ is on the right.

After some algebra (combining $S \frac{\partial V}{\partial S}$ terms which have coefficient $r-\mu$), we get:

$$\frac{\mathcal{A}V - (\mu - r)S \frac{\partial V}{\partial S} - rV}{\eta \sqrt{v} \mathcal{V}_{sens}(V)} = \frac{\mathcal{A}V_1 - (\mu - r)S \frac{\partial V_1}{\partial S} - rV_1}{\eta \sqrt{v} \mathcal{V}_{sens}(V_1)}$$

(Note: We divided by $\eta \sqrt{v}$ to normalize the risk quantity).

#### Step 6: Market Price of Volatility Risk

Observe the equation above:

* The LHS depends only on $V$.

* The RHS depends only on $V_1$.

* $V$ and $V_1$ can be any two derivatives.

For this equality to hold for any contract, both sides must equal a common function that depends only on the state variables $(t, S, v)$. We call this function the Market Price of Volatility Risk, denoted by $-\lambda_{mpr}(t, S, v)$.

$$\frac{\mathcal{A}V - (\mu - r)S \frac{\partial V}{\partial S} - rV}{\eta \sqrt{v} \sum \frac{\partial V}{\partial U_i}} = -\lambda_{mpr}$$

#### Step 7: The Final Pricing PDE

Now, multiply the denominator to the right side and expand the operator $\mathcal{A}V$.

$$\mathcal{A}V - (\mu - r)S \frac{\partial V}{\partial S} - rV = -\lambda_{mpr} \eta \sqrt{v} \sum_{i=1}^n \frac{\partial V}{\partial U_i}$$

Let's write out $\mathcal{A}V$ fully to see the magic cancellation:

$$\underbrace{\frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2}vS^2 \frac{\partial^2 V}{\partial S^2} + \sum \text{Drift}_i \frac{\partial V}{\partial U_i} + \dots}_{\mathcal{A}V} - \mu S \frac{\partial V}{\partial S} + rS \frac{\partial V}{\partial S} - rV = -\lambda_{mpr} \eta \sqrt{v} \sum \frac{\partial V}{\partial U_i}$$

Key Observation 1: Cancellation of $\mu$
The operator $\mathcal{A}V$ naturally contains a drift term for $S$ inside the definition of dynamics if we were to write it fully, but specifically here, look at the $\frac{\partial V}{\partial S}$ terms:
We have $-(\mu - r)S \frac{\partial V}{\partial S}$.
In the standard Black-Scholes or Heston derivation, the physical drift $\mu$ disappears and is replaced by $r$. The equation rearranges to use $rS \frac{\partial V}{\partial S}$.

Key Observation 2: The New Drift for Factors
We combine the original drift of the factors with the market price of risk term.
Original term in $\mathcal{A}V$: $\sum \text{Drift}_i \frac{\partial V}{\partial U_i}$
New term from RHS moved to Left: $+ \lambda_{mpr} \eta \sqrt{v} \sum \frac{\partial V}{\partial U_i}$

Grouping the terms for $\frac{\partial V}{\partial U_i}$:

$$\text{New Drift}_i = \text{Drift}_i + \lambda_{mpr} \eta \sqrt{v}$$

Substituting $\text{Drift}_i = -x_i U_i - \lambda(v - \bar{v})$:

$$\text{Risk Neutral Drift}_i = -x_i U_i - \lambda(v - \bar{v}) + \lambda_{mpr} \eta \sqrt{v}$$

#### Conclusion: The Complete Pricing PDE

The value of the option $V$ satisfies the following partial differential equation:

$$\begin{aligned}
\frac{\partial V}{\partial t} &+ \frac{1}{2} v S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV \\
&+ \sum_{i=1}^n \left[ -x_i U_i - \lambda(v - \bar{v}) + \lambda_{mpr} \eta \sqrt{v} \right] \frac{\partial V}{\partial U_i} \\
&+ \frac{1}{2} \eta^2 v \sum_{i,j} \frac{\partial^2 V}{\partial U_i \partial U_j} \\
&+ \rho \eta v S \sum_{i=1}^n \frac{\partial^2 V}{\partial S \partial U_i} = 0
\end{aligned}$$

This is a high-dimensional PDE (dimension $n+2$). While difficult to solve using Finite Difference methods (due to the "Curse of Dimensionality"), the affine structure of this equation (linear in $v$ and $U_i$) allows us to solve it semi-analytically using Fourier Transforms / Characteristic Functions, just like the standard Heston model.





### 📚 References & Further Reading
For a deeper dive into the dynamics of the volatility surface and the practical implementation of these models, the following resource is highly recommended:

* **Gatheral, J. (2006).** *The Volatility Surface: A Practitioner's Guide*. Hoboken, NJ: John Wiley & Sons.
* **Eduardo Abi Jaber (2019) .** *Lifting the Heston model*. Quantitative Finance
