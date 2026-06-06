# ternary-hmm

**Hidden Markov Models with Ternary States and Emissions**

A complete HMM implementation where both hidden states and observable emissions are ternary values {-1, 0, +1}. Includes forward algorithm, backward algorithm, Viterbi decoding, Baum-Welch training, and state prediction (filtering and smoothing).

---

## Why Ternary HMM?

Hidden Markov Models are traditionally defined over discrete symbol alphabets or continuous distributions. When the underlying system naturally produces ternary signals — {-1, 0, +1} — a ternary HMM captures the structure directly:

- **Balanced ternary computing**: model state transitions in ternary processors
- **Ternary neural activations**: hidden states in quantized RNNs
- **Signal processing**: ternary quantization of continuous signals (negative/zero/positive)
- **Financial modeling**: market states (bear/neutral/bull) with ternary indicators
- **Natural language**: sentiment flow (negative/neutral/positive) through text

This crate provides a self-contained, well-tested HMM with exactly 3 states and 3 emission symbols, constrained to the ternary domain.

---

## Model Definition

A Ternary HMM λ = (π, A, B) has:

- **States**: S = {-1, 0, +1}
- **Emissions**: O = {-1, 0, +1}
- **Initial distribution**: π = [P(state₀ = -1), P(state₀ = 0), P(state₀ = +1)]
- **Transition matrix**: A[i][j] = P(state_{t+1} = j | state_t = i)
- **Emission matrix**: B[i][k] = P(emission_t = k | state_t = i)

All probabilities are parameterized — there is no assumption that states "prefer" matching emissions.

---

## Quick Start

```rust
use ternary_hmm::TernaryHMM;

// Create a model with specific parameters
let pi = [0.2, 0.5, 0.3];
let a = [
    [0.7, 0.2, 0.1], // from state -1
    [0.1, 0.8, 0.1], // from state  0
    [0.1, 0.2, 0.7], // from state +1
];
let b = [
    [0.8, 0.1, 0.1], // state -1 emits -1 mostly
    [0.1, 0.8, 0.1], // state  0 emits  0 mostly
    [0.1, 0.1, 0.8], // state +1 emits +1 mostly
];
let hmm = TernaryHMM::with_params(pi, a, b).unwrap();

let obs = vec![1, 1, -1, 0, 1];

// --- Forward algorithm ---
let (alpha, likelihood) = hmm.forward(&obs).unwrap();
// alpha[t][i] = P(O_0..O_t, state_t = i)
// likelihood = P(O)

// --- Backward algorithm ---
let beta = hmm.backward(&obs).unwrap();
// beta[t][i] = P(O_{t+1}..O_{T-1} | state_t = i)

// --- Sequence likelihood ---
let p = hmm.sequence_likelihood(&obs).unwrap();

// --- Viterbi decoding ---
let (path, log_prob) = hmm.viterbi(&obs).unwrap();
// path: most likely state sequence
// log_prob: log P(path, O)

// --- State prediction (filtering) ---
let state = hmm.predict_state(&obs, 2).unwrap(); // most likely state at t=2 given O_0..O_2

// --- State smoothing ---
let state = hmm.smooth_state(&obs, 2).unwrap(); // most likely state at t=2 given ALL obs

// --- Baum-Welch training ---
let mut hmm2 = TernaryHMM::new(); // start uniform
let training_obs = vec![1, 1, 1, 0, 0, -1, -1, -1, 1, 1];
let likelihoods = hmm2.baum_welch(&training_obs, 100, 1e-8).unwrap();
// likelihoods: log-likelihood at each iteration (should be non-decreasing)
```

---

## Algorithm Details

### Forward Algorithm
Computes α[t][i] = P(O_0, ..., O_t, q_t = S_i) using dynamic programming:
- **Init**: α[0][i] = π[i] · B[i][O_0]
- **Step**: α[t][j] = Σ_i α[t-1][i] · A[i][j] · B[j][O_t]
- **Result**: P(O) = Σ_i α[T-1][i]

Time complexity: O(T · N²) = O(T · 9) since N=3 is fixed.

### Backward Algorithm
Computes β[t][i] = P(O_{t+1}, ..., O_{T-1} | q_t = S_i):
- **Init**: β[T-1][i] = 1
- **Step**: β[t][i] = Σ_j A[i][j] · B[j][O_{t+1}] · β[t+1][j]

### Viterbi Algorithm
Finds the single most likely state sequence using log-space dynamic programming:
- Avoids numerical underflow for long sequences
- Returns both the optimal path and its log-probability

### Baum-Welch (EM)
Iteratively re-estimates parameters to maximize P(O):
1. **E-step**: Compute γ (state posteriors) and ξ (transition posteriors) using forward-backward
2. **M-step**: Update π, A, B from expected counts
3. **Converge**: Stop when |ΔP(O)| < tolerance

Guaranteed to converge to a local maximum. Run with multiple random initializations for better results.

### Filtering vs Smoothing
- **Filtering** (predict_state): P(q_t | O_0..O_t) — uses only past observations
- **Smoothing** (smooth_state): P(q_t | O_0..O_{T-1}) — uses all observations

---

## Mathematical Verification

The crate is tested with manually computed values:

**Example**: HMM with π = [0.5, 0.3, 0.2], observation O = [+1]:
```
α[0] = [π[0]·B[0][2], π[1]·B[1][2], π[2]·B[2][2]]
     = [0.5·0.1, 0.3·0.1, 0.2·0.7]
     = [0.05, 0.03, 0.14]
P(O) = 0.22
```

These exact values are verified in the test suite, ensuring numerical correctness.

---

## Consistency Guarantees

- Forward and backward algorithms produce the same P(O)
- Baum-Welch likelihoods are monotonically non-decreasing
- Viterbi returns a valid ternary state sequence
- All probabilities remain non-negative throughout training

---

## Research Applications

- **Ternary logic circuits**: model sequential ternary logic with hidden states
- **Quantized speech recognition**: ternary acoustic features with HMM decoding
- **Computational biology**: ternary-encoded genomic signals (e.g., methylation states)
- **Financial time series**: hidden regime detection with ternary indicators
- **Reinforcement learning**: ternary action/state spaces with partially observable environments

---

## API Reference

| Method | Description |
|---|---|
| `TernaryHMM::new()` | Uniform initialization |
| `TernaryHMM::with_params(pi, a, b)` | Custom parameters with validation |
| `forward(&self, obs)` | Forward algorithm → (alpha, likelihood) |
| `backward(&self, obs)` | Backward algorithm → beta |
| `sequence_likelihood(&self, obs)` | P(O) |
| `viterbi(&self, obs)` | Most likely path → (states, log_prob) |
| `baum_welch(&mut self, obs, max_iter, tol)` | EM training → likelihoods per iteration |
| `predict_state(&self, obs, t)` | Filtering: P(q_t \| O_0..O_t) |
| `smooth_state(&self, obs, t)` | Smoothing: P(q_t \| O_0..O_{T-1}) |

---

## License

MIT
