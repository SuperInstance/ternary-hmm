# ternary-hmm

Hidden Markov Models with ternary states and emissions {-1, 0, +1} — forward-backward algorithms, Viterbi decoding, Baum-Welch training, filtering and smoothing.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why this exists

Hidden Markov Models model sequential data where the true state is hidden and you only observe noisy emissions. Classic HMMs use arbitrary discrete symbols or continuous Gaussians. When your underlying system naturally produces ternary signals — market sentiment (bear/neutral/bull), quantized neural activations (−1/0/+1), ternary processor states — a ternary HMM captures the structure directly without any encoding overhead.

The ternary constraint is more than a convenience: with exactly 3 states and 3 emissions, all the HMM matrices are 3×3. That's 9 transition probabilities, 9 emission probabilities, and 3 initial probabilities — 21 parameters total. You can train a usable model from surprisingly few observations.

## The key insight

An HMM over ternary symbols has a natural symmetry: state −1 should emit −1, state 0 should emit 0, state +1 should emit +1. The emission matrix B is approximately diagonal when the model is well-specified. This means you can initialize Baum-Welch with a nearly-diagonal B and converge in fewer iterations — a trick that doesn't generalize to larger symbol alphabets.

## Quick Start

```rust
use ternary_hmm::TernaryHMM;

// Define the model: states {-1, 0, +1}, emissions {-1, 0, +1}
let pi = [0.2, 0.5, 0.3];           // initial state distribution
let a = [                              // transition matrix
    [0.7, 0.2, 0.1],  // from −1: likely to stay −1
    [0.1, 0.8, 0.1],  // from  0: very sticky
    [0.1, 0.2, 0.7],  // from +1: likely to stay +1
];
let b = [                              // emission matrix
    [0.8, 0.1, 0.1],  // state −1 → emits −1 mostly
    [0.1, 0.8, 0.1],  // state  0 → emits  0 mostly
    [0.1, 0.1, 0.8],  // state +1 → emits +1 mostly
];
let hmm = TernaryHMM::with_params(pi, a, b).unwrap();

let obs = vec![1, 1, -1, 0, 1];

// ── Forward algorithm ──
let (alpha, likelihood) = hmm.forward(&obs).unwrap();
// alpha[t][i] = P(O_0..O_t, state_t = i)
// likelihood  = P(O)

// ── Viterbi: most likely state sequence ──
let (path, log_prob) = hmm.viterbi(&obs).unwrap();

// ── Filtering: P(state_t | O_0..O_t) ──
let state = hmm.predict_state(&obs, 2).unwrap(); // state at t=2 given observations up to t=2

// ── Smoothing: P(state_t | ALL observations) ──
let state = hmm.smooth_state(&obs, 2).unwrap(); // state at t=2 given all observations

// ── Baum-Welch training ──
let mut hmm2 = TernaryHMM::new(); // start uniform
let training_obs = vec![1, 1, 1, 0, 0, -1, -1, -1, 1, 1];
let likelihoods = hmm2.baum_welch(&training_obs, 100, 1e-8).unwrap();
// likelihoods: monotonically non-decreasing (guaranteed by EM)
```

## Architecture

```
                 TernaryHMM λ = (π, A, B)
                 ┌─────────────────────────┐
                 │ π: initial state probs   │  [f64; 3]
                 │ A: transition matrix     │  [[f64; 3]; 3]
                 │ B: emission matrix       │  [[f64; 3]; 3]
                 └─────────┬───────────────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    forward()         viterbi()       baum_welch()
    α[t][i]           δ[t][i], ψ[t][i]  E-step: γ, ξ
    P(O)              best path        M-step: update π,A,B
           │               │               │
           ▼               ▼               ▼
    sequence_likelihood  (path, log_p)   improved model
    predict_state (filter)
    smooth_state (forward-backward)
```

All algorithms work on fixed-size 3×3 arrays — no heap allocation for the core DP tables (only for the time dimension, which scales with observation length T).

## API Reference

### Model Construction

```rust
let hmm = TernaryHMM::new();  // uniform initialization
let hmm = TernaryHMM::with_params(pi: [f64; 3], a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> Result<_, String>;
```

Parameters are validated: each row must sum to 1.0 (within 1e-6 tolerance).

### Inference

| Method | Returns | Complexity |
|--------|---------|-----------|
| `forward(&obs)` | `(alpha: Vec<[f64; 3]>, likelihood: f64)` | O(T·9) |
| `backward(&obs)` | `beta: Vec<[f64; 3]>` | O(T·9) |
| `sequence_likelihood(&obs)` | `P(O): f64` | O(T·9) |
| `viterbi(&obs)` | `(path: Vec<Trit>, log_prob: f64)` | O(T·9) |
| `predict_state(&obs, t)` | `Trit` (filtering) | O(T·9) |
| `smooth_state(&obs, t)` | `Trit` (smoothing) | O(T·9) |

### Training

```rust
hmm.baum_welch(&obs, max_iter: usize, tol: f64) -> Result<Vec<f64>, String>
// Returns per-iteration likelihoods (should be non-decreasing)
```

### Utilities

```rust
fn trit_to_index(t: Trit) -> usize;   // {-1, 0, 1} → {0, 1, 2}
fn index_to_trit(i: usize) -> Trit;   // {0, 1, 2} → {-1, 0, 1}
fn validate_ternary(seq: &[Trit]) -> Result<(), String>;
```

## Real-world example

A quantized trading firm models market regimes as ternary HMM states: −1 (bear), 0 (neutral), +1 (bull). Observable emissions are daily sentiment signals: negative news flow, flat, or positive news flow.

With 252 trading days of observations, Baum-Welch converges in ~30 iterations. The trained model reveals:

- **High self-transition probabilities** (A[i][i] > 0.7): regimes persist for weeks
- **Asymmetric transitions** (A[−1][+1] < A[+1][−1]): bear→bull transitions are rarer than bull→bear
- **Noisy emissions** (B[i][i] ≈ 0.75): sentiment signals are right 75% of the time

Smoothing (using all 252 days) gives much better regime estimates at the boundaries than filtering (using only past data). The Viterbi path provides a single best-guess regime sequence for backtesting.

## Ecosystem connections

- **[`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize)** — produces ternary observations from continuous signals
- **[`ternary-knn`](https://github.com/SuperInstance/ternary-knn)** — non-temporal alternative (ignores sequence ordering)
- **[`ternary-svm`](https://github.com/SuperInstance/ternary-svm)** — frame-by-frame classification (no temporal model)
- **[`ternary-transformer`](https://github.com/SuperInstance/ternary-transformer)** — attention-based alternative to HMMs for sequences

## Performance

| Operation | Time | Space |
|-----------|------|-------|
| Forward/Backward | O(T·9) | O(T·3) |
| Viterbi | O(T·9) | O(T·3) |
| Baum-Welch iteration | O(T·27) | O(T·9) |

Since N=3 is fixed, the N² factor is always 9. The algorithms scale linearly with observation length T. For T=10,000, a single forward pass takes microseconds.

## Consistency guarantees

- Forward and backward algorithms produce the same P(O) (verified in tests to 1e-10 tolerance)
- Baum-Welch likelihoods are monotonically non-decreasing
- Viterbi always returns a valid ternary state sequence
- All probabilities remain non-negative throughout training

## Open questions

- **Multiple sequences**: Baum-Welch currently trains on a single observation sequence. Concatenating multiple sequences with artificial transitions is the standard workaround, but a native multi-sequence API would be cleaner.
- **Regularization**: No prior on transition or emission matrices. With sparse data, some probabilities may collapse to zero — should we add Laplace smoothing?
- **Continuous-time variant**: For irregularly spaced observations, a continuous-time ternary HMM with exponential holding times might be more appropriate.
- **Hierarchical HMMs**: Nested ternary HMMs where the macro-state is also ternary — useful for multi-scale temporal modeling.

## Testing

```bash
cargo test
```

13 tests: forward/backward consistency (P(O) matches to 1e-10), manually computed alpha values, Viterbi on unambiguous sequences, Baum-Welch monotonic likelihood, known-sequence decoding, filtering/smoothing agreement, parameter validation, edge cases (empty sequences, invalid trits).

## License

MIT
