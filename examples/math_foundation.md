# Mathematical and Statistical Foundation of BOCPD

Here we provide a rigorous derivation of the Bayesian Online Changepoint Detection algorithm from first principles.

**Note:** This is theoretical in nature. Understanding this material is **not required** to use the library effectively, but provides deep insight into how BOCPD works and why design choices were made.

**Who should read this:**
- Those wanting to understand the mathematical foundations
- Researchers extending or modifying the algorithm
- Users who want to see the connection between theory and implementation

**You can skip this if:** You just want to use BOCPD for applications (start with tutorial notebooks).

---

## Topics Covered

1. **Notation:** Mathematical objects and their meaning
2. **Derivation:** Step-by-step posterior distribution recursion
3. **Assumptions:** What makes BOCPD tractable
4. **Usage:** How to extract information from the posterior
5. **Implementation:** How the recursion is actually computed
6. **Connection:** Link to the original Adams & MacKay (2007) paper

---

## 1. Notation for mathematical objects in BOCPD
- **Obeservations:**   $x_1, x_2,...,x_t$
  - Time series observations that may contain changepoints or non-overlapping product patitions.
    <br /><br />
    
- **Product Partitions:**   $p_1, p_2,...p_n$
  - Time axis divided into non-overlapping contiguous segments
  - Represents the different runs/segments/regimes we can split data into by changepoint.
    <br /><br />

- **Data distribution inside a segment:**
  - All observations in a particular segment/partition is assumed to be *iid*
  - All observations in a particular segment/partition are generated from some distribution $P(x_t|n_p)$ where $n_p$ denotes the latent parameters of that segment
    <br /><br />
    
- **Contiguous Subspaces:** $x_{a:b} = x_a, x_{a+1},...x_b$
  - Denotes a segment of observations.
  - We also use the notation $x_t^{(r)} = x_{t-r+1:t}$ to denote the set of observations in the current segment of length **r** ending at time **t**.
    <br /><br />

- **Gap Distribution:** $P_{gap}(g)$
  - Prior distribution over segment lengths.
  - Interpretable as prior probability that the segment lasts exactly **g** time steps.
    <br /><br />

- **Run length:** $r_t$
  - Run length at time **t**.
  - $r_t$ = number of observations since last changepoint.
  - Changepoint at time $t \quad \xrightarrow \quad r_t = 0$
  - Current segment start at time $s \quad \xrightarrow \quad r_t = t-s$
    <br /><br />
    
- **BOCPD Goal:** $P(r_t | x_{1:t})$
  - We want to calculate the posterior distribution of run length at time t.
  - This tells us the probability of how long a regime has lasted, thus we can see how likely it is that a changepoint just happened.

---

## 2. Derivation of the posterior distribution over run length

In this section we will derive how to recursively express the computation of $P(r_t | x_{1:t})$ in **7** simple steps.

### Step-by-Step Derivation

**Goal:** Compute $P(r_t | x_{1:t})$ recursively using previously computed quantities.

$P(r_t | x_{1:t}) = \frac {P(r_t, x_{1:t})}{P(x_{1:t})}$ &emsp; *(1) Definition of conditional probability*

*where:* <br />
${P(x_{1:t})} = \sum_{r_t}P(r_t, x_{1:t})$ &emsp; *(2) Marginalization over run lengths*

So we focus on computing the joint probability: <br />

$P(r_t, x_{1:t}) = \sum_{r_{t-1}} P(r_t, r_{t-1}, x_{1:t})$ &emsp; *(3) Marginalization over* $r_{t-1}$ <br />

$= \sum_{r_{t-1}} P(r_t, x_t | r_{t-1}, x_{1:t-1})P(r_{t-1}, x_{1:t-1})$ &emsp; *(4) Product rule* <br />

$= \sum_{r_{t-1}} P(r_t|r_{t-1}, x_{1:t-1}, x_t)P(x_t|r_{t-1}, x_{1:t-1})P(r_{t-1}, x_{1:t-1})$ &emsp; *(5) Chain rule* <br />

$= \sum_{r_{t-1}} P(r_t|r_{t-1})P(x_t|r_{t-1}, x_{1:t-1})P(r_{t-1}, x_{1:t-1})$ &emsp; *(6) By Assumption A (Markov property)* <br />

$= \sum_{r_{t-1}} P(r_t|r_{t-1})P(x_t|r_{t-1}, x_t^{(r_{t-1})})P(r_{t-1}, x_{1:t-1})$ &emsp; *(7) By Assumption B (Segment independence)* <br />

### Components of the Recursion

The final expression (7) decomposes into three interpretable components:

1. **Change Prior:** $P(r_t|r_{t-1})$ 
   - Probability of transitioning from run length $r_{t-1}$ to $r_t$
   - Determined by the hazard function

2. **Predictive Likelihood:** $P(x_t|r_{t-1}, x_t^{(r_{t-1})})$ 
   - Probability of observing $x_t$ given current segment data
   - Integrates out unknown parameters

3. **Previous Joint:** $P(r_{t-1}, x_{1:t-1})$ 
   - Joint from previous time step (recursion anchor)
   - Already computed in previous iteration

### Mathematical Notes

**Step 3:** We introduce $r_{t-1}$ by marginalizing over all possible previous run lengths. This is valid because $\sum_{r_{t-1}} P(A, r_{t-1}) = P(A)$ for any event $A$.

**Step 4:** We apply the product rule (chain rule of probability): $P(A, B) = P(A|B)P(B)$ with $A = (r_t, x_t)$ and $B = (r_{t-1}, x_{1:t-1})$.

**Step 6:** We drop the conditioning on $x_{1:t}$ from the change prior because run length evolution is assumed independent of observed values (Assumption A).

---

### Predictive Likelihood Detail

The predictive likelihood marginalizes over unknown parameters $\theta$:

**After observing segment data** (when $r_{t-1} > 0$): <br />
$ P(x_t|r_{t-1}, x_t^{(r_{t-1})}) = \int p(x_t|\theta)p(\theta|x_t^{(r_{t-1})})d\theta$ &emsp; *(i) Posterior predictive*

**At start of new segment** (when $r_{t-1} = 0$, implying $r_t = 1$ after increment): <br />
$ P(x_t|r_{t-1}=0) = \int p(x_t|\theta)p(\theta)d\theta$ &emsp; *(ii) Prior predictive*

**Key insight:** By using conjugate priors, these integrals have closed-form solutions, enabling efficient online computation.

---

## 3. Assumptions of BOCPD

The BOCPD algorithm relies on several key assumptions that make the problem tractable while remaining realistic for many applications.

### Assumption A: Markov Run Length Process

**Formal statement:**

$$P(r_t | r_{t-1}, x_{1:t}) = P(r_t | r_{t-1})$$

**Interpretation:** The evolution of the run length depends **only** on the previous run length, not on the observed data values.

**Why this makes sense:**
- The hazard function $H(\tau)$ determines changepoint probability based on **time** (how long since last changepoint)
- Once we know the current segment has lasted $r_{t-1}$ steps, the **probability** it ends now is governed by the gap distribution
- The actual **values** of observations don't directly control when changepoints occur (they only provide evidence through the likelihood)

**Example:** If changepoints occur roughly every 100 steps (constant hazard), the probability of a changepoint at step 50 is the same whether observations are high or low.

**Mathematical consequence:** This allows us to separate $P(r_t | r_{t-1})$ from the data likelihood in the recursion.

---

### Assumption B: Segment Independence (Conditional on Changepoints)

**Formal statement:**

$$P(x_t | r_{t-1}, x_{1:t-1}) = P(x_t | x_t^{(r_{t-1})})$$

where $x_t^{(r_{t-1})} = x_{t-r_{t-1}+1:t}$ denotes observations in the current segment.

**Interpretation:** Given the current segment, $x_t$ depends **only** on data since the last changepoint, not on earlier segments.

**Why this makes sense:**
- Changepoints partition the time series into independent regimes
- Each segment has its own parameters $\theta$ drawn from the prior
- Observations in different segments are conditionally independent given the changepoint locations

**Example:** If a changepoint occurred at $t=50$, then $x_{100}$ depends on $x_{51:99}$ (current segment) but not on $x_{1:50}$ (previous segment).

**Mathematical consequence:** We only need to maintain sufficient statistics for the **current segment**, not the entire history.

---

### Assumption C: Independent and Identically Distributed Within Segments

**Formal statement:**

$$x_t | \theta, r_t \sim p(x | \theta) \quad \text{i.i.d. within each segment}$$

**Interpretation:** Within a segment (between changepoints), observations are:
- **Independent:** $p(x_i, x_j | \theta) = p(x_i | \theta)p(x_j | \theta)$
- **Identically distributed:** Same parameters $\theta$ throughout the segment

**Why this makes sense:**
- Segments represent "stationary regimes" where the data-generating process is stable
- Changes in the underlying process manifest as changepoints, creating new segments

**Limitations:** This assumption rules out:
- Autocorrelation within segments (e.g., AR processes)
- Gradual drift (slow parameter changes)
- Seasonal patterns within segments

**Extensions:** More sophisticated BOCPD variants relax this to allow autoregressive models.

---

### Assumption D: Conjugate Prior-Likelihood Pairs (Implementation Choice)

**Formal statement:**

If likelihood is $p(x | \theta)$ and prior is $p(\theta)$, then posterior $p(\theta | x_{1:r})$ has the same functional form as the prior.

**Examples:**
- Gaussian likelihood → Normal-Inverse-Gamma prior (GaussianNIG)
- Poisson likelihood → Gamma prior (Poisson-Gamma)
- Bernoulli likelihood → Beta prior (Bernoulli-Beta)

**Why this is crucial:**
- Enables **closed-form** posterior updates: $p(\theta | x_{1:r})$ can be computed exactly
- Allows **closed-form** predictive distributions: $\int p(x | \theta)p(\theta | x_{1:r})d\theta$ has analytical solution
- Makes algorithm **computationally tractable** for online use

**Not a fundamental limitation:** BOCPD can work with non-conjugate models using approximations (e.g., particle filters, variational inference), but at higher computational cost.

---

**These assumptions are reasonable for many real-world changepoint detection problems, especially when:**
- Changepoints represent true regime shifts
- Segments are relatively stationary
- Computational efficiency is important

---

## 4. Using the Run Length Posterior

Once we have $P(r_t | x_{1:t})$, we can extract various quantities for changepoint detection and analysis.

### 4.1 Changepoint Probability

**Definition:**

$$P(\text{changepoint at } t) = P(r_t = 0 | x_{1:t})$$

**Interpretation:** Probability that a changepoint just occurred.

**Usage:** Simple threshold-based detection:
```python
if P(r_t = 0 | x_1:t) > threshold:
    detect_changepoint()
```

**Advantages:** Intuitive, easy to implement  
**Disadvantages:** Requires manual threshold tuning

---

### 4.2 Maximum A Posteriori (MAP) Run Length

**Definition:**

$$r_{MAP}(t) = \arg\max_r P(r_t = r | x_{1:t})$$

**Interpretation:** Most likely run length given the data.

**Detection rule:**
- If $r_{MAP}(t) = 0$: Changepoint is the most probable explanation
- If $r_{MAP}(t) > 0$: Currently in a segment that started $r_{MAP}$ steps ago

**Usage in library:**
```python
map_r = bocpd.get_map_run_length()
if map_r == 0:
    changepoint_detected()
```

**Advantages:**
- No threshold needed (uses mode of distribution)
- Provides estimate of segment age
- Natural for decision-making

---

### 4.3 Confidence in MAP Estimate

**Definition:**

$$\text{Confidence} = P(r_t = r_{MAP}(t) | x_{1:t})$$

**Interpretation:** How concentrated is the posterior around the MAP estimate?

**Usage:** Filter low-confidence detections:
```python
map_r = bocpd.get_map_run_length()
confidence = bocpd.get_map_confidence()

if map_r == 0 and confidence > 0.3:
    high_confidence_changepoint()
```

**Typical values:**
- **> 0.7:** High confidence, peaked posterior
- **0.3 - 0.7:** Moderate confidence
- **< 0.3:** Low confidence, diffuse posterior

---

### 4.4 Posterior Predictive Distribution

**Definition:**

$$P(x_{t+1} | x_{1:t}) = \sum_{r_t} P(x_{t+1} | r_t, x_{1:t}) P(r_t | x_{1:t})$$

**Interpretation:** Predicted distribution for next observation, averaging over run length uncertainty.

**Components:**
- $P(x_{t+1} | r_t, x_{1:t})$: Prediction given specific run length (from observation model)
- $P(r_t | x_{1:t})$: Run length posterior (from BOCPD)

**Usage:** Anomaly detection, forecasting

**Example (GaussianNIG):**

For each $r_t$, the predictive is Student-t:
$$P(x_{t+1} | r_t = r, x_{1:t}) = \text{Student-t}(\nu_r, \mu_r, \sigma_r^2)$$

where parameters $(\nu_r, \mu_r, \sigma_r^2)$ are updated from data in current segment.

---

### 4.5 Expected Run Length

**Definition:**

$$\mathbb{E}[r_t | x_{1:t}] = \sum_{r=0}^{R_{max}} r \cdot P(r_t = r | x_{1:t})$$

**Interpretation:** Average run length weighted by posterior probabilities.

**Difference from MAP:**
- MAP = mode (most likely value)
- Mean = weighted average

**When to use:**
- MAP: When you need a single point estimate for decisions
- Mean: When you want expected value (e.g., for cost calculations)

---

### 4.6 Credible Intervals

**Definition:** Find interval $[r_{low}, r_{high}]$ such that:

$$\sum_{r=r_{low}}^{r_{high}} P(r_t = r | x_{1:t}) \geq 1 - \alpha$$

**Interpretation:** With probability $1-\alpha$, the true run length is in this interval.

**Example (95% credible interval):** $\alpha = 0.05$

**Usage:** Uncertainty quantification, especially when posterior is multimodal.

---

## 5. Implementation Details

Understanding how the algorithm is implemented helps optimize performance and debug issues.

### 5.1 The Recursion in Practice

**What we actually compute:**

At each time $t$, maintain:
```
posterior_r[r] = P(r_t = r, x_{1:t})  # Unnormalized joint
```

**Update step:**

```python
# For each possible run length r_{t}
for r_t in range(max_run_length + 1):
    if r_t == 0:
        # Changepoint: sum over all previous run lengths
        posterior_r[0] = sum(
            posterior_prev[r] * hazard(r) * pred_prob(x_t, r)
            for r in range(max_run_length + 1)
        )
    else:
        # Growth: came from r_{t-1} = r_t - 1
        r_prev = r_t - 1
        posterior_r[r_t] = (
            posterior_prev[r_prev] 
            * (1 - hazard(r_prev))  # Survival probability
            * pred_prob(x_t, r_prev)
        )
```

**Normalization:**

```python
Z = sum(posterior_r)  # Evidence P(x_{1:t})
posterior_r /= Z      # Normalize to get P(r_t | x_{1:t})
cp_prob = posterior_r[0]  # Changepoint probability
```

---

### 5.2 Computational Complexity

**Per observation:**
- **Time:** $O(R_{max})$ for updating posterior
- **Space:** $O(R_{max})$ to store posterior distribution

**For T observations:**
- **Time:** $O(T \cdot R_{max})$
- **Space:** $O(R_{max})$ (constant, only store current posterior)

**Why this is efficient:**
- Linear in $R_{max}$ (not quadratic)
- Constant space (don't store full history)
- Online: process data as it arrives

---

### 5.3 Numerical Stability

**Challenge:** Probabilities can underflow (become too small to represent).

**Solution in this library:** Work in C with double precision, but can use log-space if needed:

```c
// Instead of: p = p1 * p2 * p3
// Use: log_p = log_p1 + log_p2 + log_p3
```

**Normalization trick:**
```python
# Prevent underflow by working with relative probabilities
log_posterior_r = log_posterior_r - log_sum_exp(log_posterior_r)
```

---

### 5.4 Efficient Sufficient Statistics

For conjugate models, we don't store all data—only **sufficient statistics**.

**Example (GaussianNIG):**

Instead of storing all $x_1, x_2, \ldots, x_r$ in current segment, store:
- $n$: count of observations
- $\bar{x}$: sample mean
- $S$: sum of squared deviations

**Update rule (recursive):**

```python
# When new observation x arrives
n_new = n + 1
mean_new = (n * mean + x) / n_new
S_new = S + (x - mean) * (x - mean_new)
```

**Complexity:** $O(1)$ per observation (constant time update)

**Memory:** $O(R_{max})$ to store statistics for each possible run length

---

### 5.5 The Change Prior: $P(r_t | r_{t-1})$

This transition probability has special structure:

$$P(r_t | r_{t-1}) = \begin{cases}
H(r_{t-1}) & \text{if } r_t = 0 \quad \text{(changepoint occurs)} \\
1 - H(r_{t-1}) & \text{if } r_t = r_{t-1} + 1 \quad \text{(segment grows)} \\
0 & \text{otherwise}
\end{cases}$$

**Interpretation:**
- Run length either **resets to 0** (changepoint) or **increments by 1** (growth)
- Probability of reset governed by hazard function $H(r_{t-1})$

**Constant Hazard Example:**

$$H(\tau) = \frac{1}{\lambda} \implies \begin{cases}
P(r_t = 0 | r_{t-1}) = \frac{1}{\lambda} \\
P(r_t = r_{t-1} + 1 | r_{t-1}) = 1 - \frac{1}{\lambda}
\end{cases}$$

**Implementation:** Vectorized operations for all run lengths simultaneously.

---

## 6. Connection to the Original BOCPD Paper

This implementation follows the framework of:

> **Adams, R. P., & MacKay, D. J. (2007).** *Bayesian Online Changepoint Detection.* arXiv:0710.3742

**Key contributions of the paper:**
1. Recursive formulation enabling online inference
2. Use of message passing for efficient computation
3. Conjugate prior framework for closed-form updates
4. Connection between hazard functions and gap distributions

**Our implementation:**
- Stays true to the mathematical framework
- Optimized C implementation for speed
- Extended with utilities like `OnlineChangeDetector`
- Designed for production use while maintaining theoretical correctness

---

## Summary

### The BOCPD Algorithm in One Equation

$$P(r_t | x_{1:t}) \propto \sum_{r_{t-1}} P(r_t | r_{t-1}) \cdot P(x_t | r_{t-1}, x_t^{(r_{t-1})}) \cdot P(r_{t-1} | x_{1:t-1})$$

**Components:**
1. **Changepoint Prior** $P(r_t | r_{t-1})$: Hazard function
2. **Predictive Likelihood** $P(x_t | r_{t-1}, x_t^{(r_{t-1})})$: Observation model
3. **Previous Posterior** $P(r_{t-1} | x_{1:t-1})$: Recursive anchor

### Key Insights

- **Elegant recursion:** Only need previous posterior, not full history
- **Bayesian framework:** Naturally handles uncertainty
- **Modular design:** Swap observation models and hazard functions
- **Computationally tractable:** $O(R_{max})$ per observation

### Practical Takeaways

- Posterior gives complete information about run length uncertainty
- Multiple ways to use it: CP probability, MAP, prediction, etc.
- Assumptions enable efficiency but have limitations
- Implementation optimizes for speed while staying mathematically sound

---