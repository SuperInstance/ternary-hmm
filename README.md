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
use ternary_hmm::Ternary::{Negative, Neutral, Positive};

// Define the model: states {-1, 0, +1}, emissions {-1, 0, +1}
let pi = [0.2, 0.5, 0.3]; // initial state distribution
let a = [
    // transition matrix
    [0.7, 0.2, 0.1], // from -1: likely to stay -1
    [0.1, 0.8, 0.1], // from  0: very sticky
    [0.1, 0.2, 0.7], // from +1: likely to stay +1
];
let b = [
    // emission matrix
    [0.8, 0.1, 0.1], // state -1 -> emits -1 mostly
    [0.1, 0.8, 0.1], // state  0 -> emits  0 mostly
    [0.1, 0.1, 0.8], // state +1 -> emits +1 mostly
];
let hmm = TernaryHMM::with_params(pi, a, b).unwrap();

let obs = vec![Positive, Positive, Negative, Neutral, Positive];

// -- Forward algorithm --
// alpha[t][i] = P(state_t = i | O_0..O_t)  (the scaled filtering posterior;
//               each row sums to 1 and never underflows, see "Numerical notes")
// likelihood  = P(O)
let (alpha, likelihood) = hmm.forward(&obs).unwrap();

// -- Viterbi: most likely state sequence --
let (path, log_prob) = hmm.viterbi(&obs).unwrap();

// -- Filtering: most likely state at t=2 given observations up to t=2 --
let state = hmm.predict_state(&obs, 2).unwrap();

// -- Smoothing: most likely state at t=2 given ALL observations --
let state = hmm.smooth_state(&obs, 2).unwrap();

// -- Baum-Welch training --
let mut hmm2 = TernaryHMM::new(); // start uniform
let training_obs = vec![
    Positive, Positive, Positive, Neutral, Neutral, Negative, Negative, Negative, Positive,
    Positive,
];
let likelihoods = hmm2.baum_welch(&training_obs, 100, 1e-8).unwrap();
// likelihoods: per-iteration P(O), non-decreasing by the EM guarantee
let _ = (alpha, likelihood, path, log_prob, state, likelihoods);
```

`Ternary` (re-exported from `ternary-types`) is the observation/state type — the three
variants are `Negative` (-1), `Neutral` (0) and `Positive` (+1).

## Architecture

```text
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

## Numerical notes

The forward and backward recursions use **Rabiner scaling** (L. R. Rabiner, 1989): each
time step is normalized by its own scaling coefficient `c[t]`. This is essential because the
unscaled joint probability `P(O_0..O_t, state_t)` is a product of many sub-unit terms that
collapses to `0.0` after only a few hundred observations — silently breaking training and
smoothing.

With matching scales:

- `alpha[t][i] = P(state_t = i | O_0..O_t)` — the **filtering** posterior (row-stochastic).
- `alpha[t][i] * beta[t][i] = P(state_t = i | O)` — the **smoothing** posterior, which sums
  to exactly 1 over `i` for every `t` (this is what the forward/backward consistency test
  checks). `smooth_state` therefore never divides by `P(O)`, avoiding the division-by-zero
  that the previous unscaled code hit when `P(O)` underflowed.
- Baum-Welch drives its convergence decision from the **log-likelihood** `Σ ln(c[t])`, which
  stays finite when `P(O) = Π c[t]` underflows on long sequences.

## API Reference

### Model Construction

```rust
# use ternary_hmm::TernaryHMM;
let hmm = TernaryHMM::new(); // uniform initialization
let hmm = TernaryHMM::with_params(
    [0.2, 0.5, 0.3],
    [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
    [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
)
.unwrap();
// Parameters are validated: each row must sum to 1.0 (within 1e-6), and every
// entry must be finite and non-negative (a row like [2, -0.5, -0.5] sums to 1
// but would yield ln(negative) = NaN in Viterbi).
```

### Inference

| Method              | Returns                                          | Complexity |
| ------------------- | ------------------------------------------------ | ---------- |
| `forward(&obs)`     | `(alpha: Vec<[f64; 3]>, likelihood: f64)`        | O(T·9)     |
| `backward(&obs)`    | `beta: Vec<[f64; 3]>`                            | O(T·9)     |
| `sequence_likelihood(&obs)` | `P(O): f64`                             | O(T·9)     |
| `viterbi(&obs)`     | `(path: Vec<Ternary>, log_prob: f64)`            | O(T·9)     |
| `predict_state(&obs, t)` | `Ternary` (filtering)                        | O(T·9)     |
| `smooth_state(&obs, t)` | `Ternary` (smoothing)                        | O(T·9)     |

`alpha[t][i]` and `beta[t][i]` are the scaled values described under "Numerical notes".
Float comparisons use `f64::total_cmp`, so the arg-max in filtering/Viterbi never panics on
`NaN`.

### Training

```rust
# use ternary_hmm::TernaryHMM;
# use ternary_hmm::Ternary::{Negative, Neutral, Positive};
# let mut hmm = TernaryHMM::new();
# let obs = vec![Positive, Neutral, Negative];
let likelihoods = hmm.baum_welch(&obs, 100, 1e-8).unwrap();
// Returns per-iteration likelihoods P(O) (non-decreasing by the EM guarantee).
// A perfectly uniform model is a fixed point (the three states are
// indistinguishable), so start from a symmetry-broken / perturbed init when you
// want transitions to be learned.
```

### Utilities

```rust
# use ternary_hmm::{index_to_trit, trit_to_index, validate_ternary, Ternary};
let idx = trit_to_index(Ternary::Positive); // -> 2
let t = index_to_trit(0); // -> Ternary::Negative
let _ = validate_ternary(&[Ternary::Positive, Ternary::Neutral]); // always Ok
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

| Operation          | Time     | Space   |
| ------------------ | -------- | ------- |
| Forward/Backward   | O(T·9)   | O(T·3)  |
| Viterbi            | O(T·9)   | O(T·3)  |
| Baum-Welch iteration | O(T·27) | O(T·9)  |

Since N=3 is fixed, the N² factor is always 9. The algorithms scale linearly with observation length T. For T=10,000, a single forward pass takes microseconds.

## Consistency guarantees

- Forward `alpha` and backward `beta` are scaled consistently, so `alpha[t][i] * beta[t][i]` is the exact smoothing posterior and sums to 1 over `i` for every `t` (verified in tests).
- Baum-Welch likelihoods are monotonically non-decreasing (tested with a strict-improvement assertion that fails on an inert M-step).
- Viterbi's returned path is a true maximum-probability path, cross-checked against brute-force enumeration of all `3^T` paths for small T.
- All probabilities remain non-negative throughout training; `with_params` rejects NaN/infinite/negative entries.

## Open questions

- **Multiple sequences**: Baum-Welch currently trains on a single observation sequence. Concatenating multiple sequences with artificial transitions is the standard workaround, but a native multi-sequence API would be cleaner.
- **Regularization**: No prior on transition or emission matrices. With sparse data, some probabilities may collapse to zero — should we add Laplace smoothing?
- **Continuous-time variant**: For irregularly spaced observations, a continuous-time ternary HMM with exponential holding times might be more appropriate.
- **Hierarchical HMMs**: Nested ternary HMMs where the macro-state is also ternary — useful for multi-scale temporal modeling.

## Testing

```bash
cargo test
```

The README code blocks above are compiled and run as doctests. There are 14 unit tests: forward/backward posterior consistency, a hand-computed forward likelihood, Viterbi on unambiguous sequences, Viterbi cross-checked against brute-force enumeration of all paths, Baum-Welch monotonic + strictly-increasing likelihood, Baum-Welch stability on a 700-observation sequence (no underflow), known-sequence decoding, filtering/smoothing agreement, parameter validation (row sums plus negative/NaN/infinity rejection), and edge cases (empty sequences).

## License

MIT
