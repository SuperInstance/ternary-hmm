//! # ternary-hmm
//!
//! Hidden Markov Model with ternary states {-1, 0, +1} and ternary emissions.
//! Implements forward algorithm, backward algorithm, Viterbi decoding,
//! Baum-Welch training (ternary-constrained), and sequence likelihood.
//!
//! The forward and backward recursions use **Rabiner scaling** (each time-step
//! is normalized by its own scaling coefficient) so that the filtering and
//! smoothing distributions never underflow to zero, even for long observation
//! sequences. With raw (unscaled) probabilities the joint `P(O_0..O_t, state_t)`
//! is a product of many sub-unit terms and collapses to `0.0` for sequences of a
//! few hundred observations, which silently breaks training and smoothing.
//!
//! Uses [`Ternary`] from `ternary-types` as the shared fleet type.

use ternary_types::Ternary;
use ternary_types::Ternary::{Negative, Neutral, Positive};

/// The three possible ternary states.
pub const STATES: [Ternary; 3] = [Negative, Neutral, Positive];

/// Map a trit to an index (0, 1, 2).
pub fn trit_to_index(t: Ternary) -> usize {
    match t {
        Negative => 0,
        Neutral => 1,
        Positive => 2,
    }
}

/// Map an index back to a trit.
pub fn index_to_trit(i: usize) -> Ternary {
    match i {
        0 => Negative,
        1 => Neutral,
        2 => Positive,
        _ => panic!("Invalid index: {i}, expected 0..2"),
    }
}

/// Validate that a slice contains only valid trits — all Ternary variants are valid.
pub fn validate_ternary(_seq: &[Ternary]) -> Result<(), String> {
    Ok(())
}

/// Output of the scaled forward recursion: scaled alpha, per-step scales, and P(O).
type ScaledForward = (Vec<[f64; 3]>, Vec<f64>, f64);

/// A Hidden Markov Model with ternary states and emissions.
#[derive(Debug, Clone)]
pub struct TernaryHMM {
    /// Initial state probabilities: pi[i] = P(state_0 = STATES[i]).
    pub pi: [f64; 3],
    /// Transition probabilities: a[i][j] = P(state_{t+1} = STATES[j] | state_t = STATES[i]).
    pub a: [[f64; 3]; 3],
    /// Emission probabilities: b[i][k] = P(emission = STATES[k] | state = STATES[i]).
    pub b: [[f64; 3]; 3],
}

impl TernaryHMM {
    /// Create a new HMM with uniform probabilities.
    pub fn new() -> Self {
        Self {
            pi: [1.0 / 3.0; 3],
            a: [[1.0 / 3.0; 3]; 3],
            b: [[1.0 / 3.0; 3]; 3],
        }
    }

    /// Create an HMM with specific parameters.
    ///
    /// Each row of `pi`, `a` and `b` must sum to 1.0 (within `1e-6`), and every
    /// entry must be finite and non-negative — a row such as `[2.0, -0.5, -0.5]`
    /// sums to one but contains a negative "probability", which would corrupt the
    /// recursions (e.g. `ln` of a negative is `NaN`).
    pub fn with_params(pi: [f64; 3], a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> Result<Self, String> {
        if let Some(msg) = check_prob_slice(&pi, "Initial probabilities") {
            return Err(msg);
        }
        let pi_sum: f64 = pi.iter().sum();
        if (pi_sum - 1.0).abs() > 1e-6 {
            return Err(format!("Initial probabilities must sum to 1, got {pi_sum}"));
        }
        for (i, row) in a.iter().enumerate() {
            if let Some(msg) = check_prob_slice(row, &format!("Transition row {i}")) {
                return Err(msg);
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("Transition row {i} must sum to 1, got {sum}"));
            }
        }
        for (i, row) in b.iter().enumerate() {
            if let Some(msg) = check_prob_slice(row, &format!("Emission row {i}")) {
                return Err(msg);
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("Emission row {i} must sum to 1, got {sum}"));
            }
        }
        Ok(Self { pi, a, b })
    }

    /// Scaled forward recursion (Rabiner, 1989).
    ///
    /// Returns:
    /// * `alpha` — the **scaled** filtering distribution `alpha[t][i] =
    ///   P(state_t = i | O_0..O_t)` (each row sums to 1).
    /// * `scales` — per-step normalization coefficients `c[t]`; the likelihood is
    ///   `prod(c[t])`.
    /// * `likelihood` — `P(O) = prod(c[t])`.
    fn forward_scaled(&self, obs: &[Ternary]) -> Result<ScaledForward, String> {
        let t = obs.len();
        if t == 0 {
            return Err("Observation sequence is empty".into());
        }

        let mut alpha = vec![[0.0f64; 3]; t];
        let mut scales = vec![0.0f64; t];

        // Initialization.
        let o0 = trit_to_index(obs[0]);
        let c0: f64 = (0..3).map(|i| self.pi[i] * self.b[i][o0]).sum();
        scales[0] = c0;
        if c0 > 0.0 {
            for (i, slot) in alpha[0].iter_mut().enumerate() {
                *slot = self.pi[i] * self.b[i][o0] / c0;
            }
        }

        // Induction.
        for t_idx in 1..t {
            let o = trit_to_index(obs[t_idx]);
            let prev = alpha[t_idx - 1];
            let mut c = 0.0;
            for (j, slot) in alpha[t_idx].iter_mut().enumerate() {
                let s: f64 = (0..3).map(|i| prev[i] * self.a[i][j]).sum();
                let val = s * self.b[j][o];
                *slot = val;
                c += val;
            }
            scales[t_idx] = c;
            if c > 0.0 {
                for slot in alpha[t_idx].iter_mut() {
                    *slot /= c;
                }
            }
        }

        let likelihood: f64 = scales.iter().product();
        Ok((alpha, scales, likelihood))
    }

    /// Scaled backward recursion (Rabiner, 1989), using the same `scales`
    /// produced by [`forward_scaled`].
    ///
    /// With matching scales, `alpha[t][i] * beta[t][i]` is exactly the smoothing
    /// posterior `P(state_t = i | O)` (it sums to 1 over `i` for every `t`).
    fn backward_scaled(&self, obs: &[Ternary], scales: &[f64]) -> Result<Vec<[f64; 3]>, String> {
        let t = obs.len();
        if t == 0 {
            return Err("Observation sequence is empty".into());
        }

        let mut beta = vec![[1.0f64; 3]; t];

        for t_idx in (0..t - 1).rev() {
            let o_next = trit_to_index(obs[t_idx + 1]);
            let next = beta[t_idx + 1];
            let c = scales[t_idx + 1];
            for (i, row) in beta[t_idx].iter_mut().enumerate() {
                let s: f64 = (0..3)
                    .map(|j| self.a[i][j] * self.b[j][o_next] * next[j])
                    .sum();
                *row = if c > 0.0 { s / c } else { 0.0 };
            }
        }

        Ok(beta)
    }

    /// Forward algorithm.
    ///
    /// Returns `(alpha, likelihood)` where:
    /// * `alpha[t][i] = P(state_t = i | O_0..O_t)` — the scaled filtering
    ///   distribution (row-stochastic, never underflows).
    /// * `likelihood = P(O)`.
    pub fn forward(&self, obs: &[Ternary]) -> Result<(Vec<[f64; 3]>, f64), String> {
        let (alpha, _scales, likelihood) = self.forward_scaled(obs)?;
        Ok((alpha, likelihood))
    }

    /// Backward algorithm: compute scaled `beta[t][i]`.
    ///
    /// `alpha[t][i] * beta[t][i]` (from [`forward`]) equals the smoothing
    /// posterior `P(state_t = i | O)`.
    pub fn backward(&self, obs: &[Ternary]) -> Result<Vec<[f64; 3]>, String> {
        let (_alpha, scales, _likelihood) = self.forward_scaled(obs)?;
        self.backward_scaled(obs, &scales)
    }

    /// Compute the sequence likelihood P(O) using the forward algorithm.
    pub fn sequence_likelihood(&self, obs: &[Ternary]) -> Result<f64, String> {
        let (_, likelihood) = self.forward(obs)?;
        Ok(likelihood)
    }

    /// Viterbi algorithm: find the most likely state sequence.
    pub fn viterbi(&self, obs: &[Ternary]) -> Result<(Vec<Ternary>, f64), String> {
        let t = obs.len();
        if t == 0 {
            return Err("Observation sequence is empty".into());
        }

        let neg_inf = f64::NEG_INFINITY;
        let mut delta = vec![[neg_inf; 3]; t];
        let mut psi = vec![[0usize; 3]; t];

        let o0 = trit_to_index(obs[0]);

        for (i, row) in delta[0].iter_mut().enumerate() {
            let p = self.pi[i] * self.b[i][o0];
            *row = if p > 0.0 { p.ln() } else { neg_inf };
        }

        for t_idx in 1..t {
            let o = trit_to_index(obs[t_idx]);
            let prev = delta[t_idx - 1];
            for (j, slot) in delta[t_idx].iter_mut().enumerate() {
                let b_j = self.b[j][o];
                if b_j <= 0.0 {
                    *slot = neg_inf;
                    continue;
                }
                let ln_b = b_j.ln();
                let mut best_val = neg_inf;
                let mut best_i = 0;
                for (i, &d) in prev.iter().enumerate() {
                    if d == neg_inf {
                        continue;
                    }
                    let val = d + self.a[i][j].ln() + ln_b;
                    if val > best_val {
                        best_val = val;
                        best_i = i;
                    }
                }
                *slot = best_val;
                psi[t_idx][j] = best_i;
            }
        }

        let (best_i, &best_val) = delta[t - 1]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty final delta row");
        let _ = best_val;

        let mut path = vec![0usize; t];
        path[t - 1] = best_i;
        for t_idx in (0..t - 1).rev() {
            path[t_idx] = psi[t_idx + 1][path[t_idx + 1]];
        }

        let state_path: Vec<Ternary> = path.iter().map(|&i| index_to_trit(i)).collect();
        Ok((state_path, best_val))
    }

    /// Baum-Welch training with ternary constraints.
    pub fn baum_welch(
        &mut self,
        obs: &[Ternary],
        max_iter: usize,
        tol: f64,
    ) -> Result<Vec<f64>, String> {
        let t_len = obs.len();
        if t_len == 0 {
            return Err("Observation sequence is empty".into());
        }

        let mut likelihoods = Vec::new();
        let mut prev_log_lik: Option<f64> = None;

        for _iter in 0..max_iter {
            let (alpha, scales, likelihood) = self.forward_scaled(obs)?;
            let beta = self.backward_scaled(obs, &scales)?;
            // Log-likelihood = sum of log(scales) stays finite even when P(O) =
            // prod(scales) underflows to 0 for long sequences; it drives the
            // convergence/exit decisions below.
            let log_lik: f64 = scales
                .iter()
                .map(|&c| if c > 0.0 { c.ln() } else { f64::NEG_INFINITY })
                .sum();
            likelihoods.push(likelihood);

            // A -inf log-likelihood means an observation is genuinely impossible
            // (a zero-probability emission along every reachable path); there is
            // nothing to re-estimate.
            if !log_lik.is_finite() {
                break;
            }

            // gamma[t][i] = P(state_t = i | O). With matching scales this is simply
            // alpha[t][i] * beta[t][i] and already sums to 1 over i.
            let gamma: Vec<[f64; 3]> = alpha
                .iter()
                .zip(beta.iter())
                .map(|(a, b)| {
                    let g = [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
                    let s = g[0] + g[1] + g[2];
                    if s > 0.0 {
                        [g[0] / s, g[1] / s, g[2] / s]
                    } else {
                        g
                    }
                })
                .collect();

            // xi[t][i][j] = P(state_t=i, state_{t+1}=j | O), normalized per t.
            let mut xi = vec![[[0.0f64; 3]; 3]; t_len - 1];
            for t_idx in 0..t_len - 1 {
                let o_next = trit_to_index(obs[t_idx + 1]);
                let next_beta = beta[t_idx + 1];
                let mut sum = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        xi[t_idx][i][j] =
                            alpha[t_idx][i] * self.a[i][j] * self.b[j][o_next] * next_beta[j];
                        sum += xi[t_idx][i][j];
                    }
                }
                if sum > 0.0 {
                    for row in xi[t_idx].iter_mut() {
                        for slot in row.iter_mut() {
                            *slot /= sum;
                        }
                    }
                }
            }

            for (i, pi_i) in self.pi.iter_mut().enumerate() {
                *pi_i = gamma[0][i];
            }

            for (i, a_row) in self.a.iter_mut().enumerate() {
                let denom: f64 = gamma.iter().take(t_len - 1).map(|g| g[i]).sum();
                if denom > 0.0 {
                    for (j, a_ij) in a_row.iter_mut().enumerate() {
                        let numer: f64 = xi.iter().take(t_len - 1).map(|x| x[i][j]).sum();
                        *a_ij = numer / denom;
                    }
                }
            }

            for (i, b_row) in self.b.iter_mut().enumerate() {
                let denom: f64 = gamma.iter().map(|g| g[i]).sum();
                if denom > 0.0 {
                    for (k, b_ik) in b_row.iter_mut().enumerate() {
                        let numer: f64 = gamma
                            .iter()
                            .zip(obs.iter())
                            .filter(|(_, &o)| trit_to_index(o) == k)
                            .map(|(g, _)| g[i])
                            .sum();
                        *b_ik = numer / denom;
                    }
                }
            }

            if let Some(prev) = prev_log_lik {
                if (log_lik - prev).abs() < tol {
                    break;
                }
            }
            prev_log_lik = Some(log_lik);
        }

        Ok(likelihoods)
    }

    /// Filtering: predict state at time t given obs up to t.
    pub fn predict_state(&self, obs: &[Ternary], t: usize) -> Result<Ternary, String> {
        let (alpha, _) = self.forward(obs)?;
        if t >= alpha.len() {
            return Err(format!("t={t} out of range (len={})", alpha.len()));
        }
        // `total_cmp` never panics, including on NaN, unlike `partial_cmp().unwrap()`.
        let best = alpha[t]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("state array is non-empty");
        Ok(index_to_trit(best))
    }

    /// Smoothing: predict state at time t given ALL observations.
    pub fn smooth_state(&self, obs: &[Ternary], t: usize) -> Result<Ternary, String> {
        let (alpha, scales, _likelihood) = self.forward_scaled(obs)?;
        if t >= alpha.len() {
            return Err(format!("t={t} out of range (len={})", alpha.len()));
        }
        let beta = self.backward_scaled(obs, &scales)?;
        // With matching scales, alpha[t][i] * beta[t][i] = P(state_t = i | O), so
        // argmax over i is the smoothed state. No division by P(O) is needed,
        // avoiding the div-by-zero that occurred when the (unscaled) likelihood
        // underflowed to 0.0.
        let mut best_i = 0;
        let mut best_prob = f64::NEG_INFINITY;
        for i in 0..3 {
            let prob = alpha[t][i] * beta[t][i];
            if prob > best_prob {
                best_prob = prob;
                best_i = i;
            }
        }
        Ok(index_to_trit(best_i))
    }
}

/// Verify a probability slice is finite and non-negative. Returns `Some(msg)` on
/// the first offending entry.
fn check_prob_slice(p: &[f64], label: &str) -> Option<String> {
    for (k, &v) in p.iter().enumerate() {
        if v.is_nan() {
            return Some(format!("{label} contains NaN at index {k}"));
        }
        if v.is_infinite() {
            return Some(format!("{label} contains infinity at index {k}"));
        }
        if v < 0.0 {
            return Some(format!("{label} contains negative value {v} at index {k}"));
        }
    }
    None
}

impl Default for TernaryHMM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_hmm() -> TernaryHMM {
        let pi = [0.2, 0.5, 0.3];
        let a = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]];
        let b = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]];
        TernaryHMM::with_params(pi, a, b).unwrap()
    }

    #[test]
    fn test_forward_probabilities_sum() {
        let hmm = make_simple_hmm();
        let obs = vec![Positive, Positive, Negative, Neutral, Positive];
        let (alpha, likelihood) = hmm.forward(&obs).unwrap();
        assert!(likelihood > 0.0, "Likelihood should be positive");
        assert!(likelihood < 1.0, "Likelihood should be < 1");
        // Scaled alpha rows are filtering posteriors and must sum to 1.
        for (t, row) in alpha.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-12,
                "alpha row {t} should sum to 1, got {s}"
            );
        }
    }

    #[test]
    fn test_forward_backward_consistency() {
        let hmm = make_simple_hmm();
        let obs = vec![Positive, Neutral, Negative, Neutral, Positive];
        let (alpha, likelihood) = hmm.forward(&obs).unwrap();
        let beta = hmm.backward(&obs).unwrap();

        // With Rabiner scaling, alpha[t][i] * beta[t][i] is the exact smoothing
        // posterior P(state_t = i | O), so it must sum to 1 over i for every t.
        for (t, (a, b)) in alpha.iter().zip(beta.iter()).enumerate() {
            let s: f64 = (0..3).map(|i| a[i] * b[i]).sum();
            assert!(
                (s - 1.0).abs() < 1e-9,
                "posterior at t={t} should sum to 1, got {s}"
            );
        }
        assert!(likelihood > 0.0);
    }

    #[test]
    fn test_viterbi_most_likely_path() {
        let hmm = make_simple_hmm();
        let obs = vec![Positive, Positive, Positive, Positive];
        let (path, _log_prob) = hmm.viterbi(&obs).unwrap();
        for (t, &state) in path.iter().enumerate() {
            assert_eq!(state, Positive, "State at t={t} should be Positive");
        }
    }

    #[test]
    fn test_viterbi_switching() {
        let hmm = make_simple_hmm();
        let obs = vec![Negative, Negative, Negative, Positive, Positive, Positive];
        let (path, _) = hmm.viterbi(&obs).unwrap();
        assert_eq!(path[0], Negative, "First state should be Negative");
        assert_eq!(path[5], Positive, "Last state should be Positive");
    }

    #[test]
    fn test_baum_welch_improves_likelihood() {
        let mut hmm = TernaryHMM::new();
        let obs = vec![
            Positive, Positive, Positive, Neutral, Neutral, Negative, Negative, Negative, Positive,
            Positive,
        ];
        let initial_likelihood = hmm.sequence_likelihood(&obs).unwrap();
        let likelihoods = hmm.baum_welch(&obs, 50, 1e-8).unwrap();
        assert!(likelihoods.len() > 1, "Should have multiple iterations");
        let final_likelihood = *likelihoods.last().unwrap();
        // STRICT improvement: an M-step that updates nothing leaves the likelihood
        // unchanged, so a non-strict `>=` test would pass on a broken trainer.
        assert!(
            final_likelihood > initial_likelihood,
            "Likelihood must strictly increase: initial={initial_likelihood}, final={final_likelihood}"
        );
    }

    #[test]
    fn test_known_sequence_decoding() {
        let pi = [1.0 / 3.0; 3];
        let a = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]];
        let b = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]];
        let hmm = TernaryHMM::with_params(pi, a, b).unwrap();
        let obs = vec![Negative, Neutral, Positive];
        let (path, _) = hmm.viterbi(&obs).unwrap();
        assert_eq!(path, vec![Negative, Neutral, Positive]);
    }

    #[test]
    fn test_manual_verification_small() {
        let pi = [0.5, 0.3, 0.2];
        let a = [[0.6, 0.2, 0.2], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]];
        let b = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]];
        let hmm = TernaryHMM::with_params(pi, a, b).unwrap();
        let obs = vec![Positive];
        let (_alpha, likelihood) = hmm.forward(&obs).unwrap();
        let expected = 0.5 * 0.1 + 0.3 * 0.1 + 0.2 * 0.7;
        assert!(
            (likelihood - expected).abs() < 1e-10,
            "likelihood = {likelihood}, expected {expected}"
        );
    }

    #[test]
    fn test_empty_observation_error() {
        let hmm = make_simple_hmm();
        assert!(hmm.forward(&[]).is_err());
        assert!(hmm.backward(&[]).is_err());
        assert!(hmm.viterbi(&[]).is_err());
    }

    #[test]
    fn test_hmm_validation() {
        assert!(
            TernaryHMM::with_params([1.0, 0.0, 0.5], [[1.0 / 3.0; 3]; 3], [[1.0 / 3.0; 3]; 3])
                .is_err()
        );
        assert!(TernaryHMM::with_params(
            [0.5, 0.3, 0.2],
            [[0.6, 0.2, 0.2], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]],
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
        )
        .is_ok());
    }

    #[test]
    fn test_baum_welch_convergence() {
        let mut hmm = TernaryHMM::new();
        let obs = vec![
            Negative, Negative, Negative, Neutral, Neutral, Positive, Positive, Positive, Positive,
            Positive,
        ];
        let likelihoods = hmm.baum_welch(&obs, 100, 1e-12).unwrap();
        // Monotonic non-decreasing across all iterations ...
        for w in likelihoods.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-10,
                "Likelihood decreased: {} -> {}",
                w[0],
                w[1]
            );
        }
        // ... but NOT all-equal: a no-op M-step leaves every likelihood identical,
        // so requiring at least one strict increase defeats that fake-green case.
        let any_strict_increase = likelihoods.windows(2).any(|w| w[1] > w[0] + 1e-12);
        assert!(
            any_strict_increase,
            "Likelihood never strictly increased across {} iterations — EM M-step may be inert",
            likelihoods.len()
        );
    }

    /// Independent verification of Viterbi: brute-force enumerate ALL 3^T state
    /// paths, compute each path's joint probability with the observations, and
    /// confirm Viterbi returns a maximum-probability path with matching log-prob.
    #[test]
    fn test_viterbi_matches_brute_force() {
        let pi = [0.5, 0.3, 0.2];
        let a = [[0.6, 0.2, 0.2], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]];
        let b = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]];
        let hmm = TernaryHMM::with_params(pi, a, b).unwrap();
        let obs = vec![Positive, Negative, Positive, Neutral]; // T=4 -> 81 paths
        let o: Vec<usize> = obs.iter().map(|&t| trit_to_index(t)).collect();
        let t = o.len();

        // Brute force over every state-index sequence.
        let mut best_prob = -1.0_f64;
        for code in 0..3usize.pow(t as u32) {
            let mut path = vec![0usize; t];
            let mut c = code;
            for slot in path.iter_mut().rev() {
                *slot = c % 3;
                c /= 3;
            }
            let mut p = pi[path[0]] * b[path[0]][o[0]];
            for k in 1..t {
                p *= a[path[k - 1]][path[k]] * b[path[k]][o[k]];
            }
            if p > best_prob {
                best_prob = p;
            }
        }

        let (vpath, vlog) = hmm.viterbi(&obs).unwrap();
        // The probability of the Viterbi-returned path must equal the brute-force max.
        let vidx: Vec<usize> = vpath.iter().map(|&t| trit_to_index(t)).collect();
        let mut vprob = pi[vidx[0]] * b[vidx[0]][o[0]];
        for k in 1..t {
            vprob *= a[vidx[k - 1]][vidx[k]] * b[vidx[k]][o[k]];
        }
        assert!(
            (vprob - best_prob).abs() < 1e-12,
            "Viterbi path prob {vprob} != brute-force max {best_prob}"
        );
        assert!(
            (vlog - best_prob.ln()).abs() < 1e-9,
            "Viterbi log-prob {vlog} != ln(brute-force max) {}",
            best_prob.ln()
        );
    }

    #[test]
    fn test_with_params_rejects_invalid_probabilities() {
        // Negative entry whose row still sums to 1 (would yield ln(negative)=NaN).
        assert!(TernaryHMM::with_params(
            [2.0, -0.5, -0.5],
            [[1.0 / 3.0; 3]; 3],
            [[1.0 / 3.0; 3]; 3]
        )
        .is_err());
        // Negative value inside a transition row.
        assert!(TernaryHMM::with_params(
            [1.0 / 3.0; 3],
            [[1.4, -0.2, -0.2], [1.0 / 3.0; 3], [1.0 / 3.0; 3]],
            [[1.0 / 3.0; 3]; 3]
        )
        .is_err());
        // NaN (note: its row sum is NaN, which the sum check alone would miss
        // because `(NaN - 1.0).abs() > 1e-6` is false).
        assert!(TernaryHMM::with_params(
            [f64::NAN, 0.0, 1.0],
            [[1.0 / 3.0; 3]; 3],
            [[1.0 / 3.0; 3]; 3]
        )
        .is_err());
        // Infinity.
        assert!(TernaryHMM::with_params(
            [1.0 / 3.0; 3],
            [[1.0 / 3.0; 3]; 3],
            [
                [f64::INFINITY, f64::NEG_INFINITY, 1.0],
                [1.0 / 3.0; 3],
                [1.0 / 3.0; 3]
            ]
        )
        .is_err());
        // A valid model still constructs.
        assert!(make_simple_hmm().pi.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_filtering_smoothing_agree_on_clean_signal() {
        // Strongly diagonal model + clean observations: filtering (past only) and
        // smoothing (all observations) should agree at interior time steps.
        let pi = [1.0 / 3.0; 3];
        let a = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]];
        let b = [
            [0.95, 0.025, 0.025],
            [0.025, 0.95, 0.025],
            [0.025, 0.025, 0.95],
        ];
        let hmm = TernaryHMM::with_params(pi, a, b).unwrap();
        let obs = vec![Negative, Negative, Neutral, Positive, Positive];
        for t in 1..obs.len() - 1 {
            let filt = hmm.predict_state(&obs, t).unwrap();
            let smooth = hmm.smooth_state(&obs, t).unwrap();
            assert_eq!(filt, smooth, "filtering != smoothing at t={t}");
            assert_eq!(filt, obs[t], "decoded state != observed at t={t}");
        }
    }

    #[test]
    fn test_baum_welch_long_sequence_stable() {
        // T=700: the OLD unscaled forward underflowed P(O) to 0.0, so baum_welch
        // hit `likelihood == 0.0` and silently broke after one iteration. With
        // scaling + log-space convergence, training proceeds normally.
        //
        // A perfectly uniform start is a Baum-Welch fixed point (the three states
        // are indistinguishable), so we start from a symmetry-broken, diagonal-
        // biased init on a blocky sequence whose sticky structure is learnable.
        let mut hmm = TernaryHMM::with_params(
            [0.34, 0.33, 0.33],
            [[0.36, 0.32, 0.32], [0.32, 0.36, 0.32], [0.32, 0.32, 0.36]],
            [[0.36, 0.32, 0.32], [0.32, 0.36, 0.32], [0.32, 0.32, 0.36]],
        )
        .unwrap();
        let init_a = hmm.a;
        // Long runs of each symbol => sticky transition structure to learn.
        let obs: Vec<_> = (0..700usize)
            .map(|i| match (i / 70) % 3 {
                0 => Negative,
                1 => Neutral,
                _ => Positive,
            })
            .collect();
        let likelihoods = hmm.baum_welch(&obs, 50, 1e-10).unwrap();
        // Multiple real iterations => no immediate underflow exit.
        assert!(
            likelihoods.len() >= 2,
            "baum_welch exited immediately (underflow?)"
        );
        // Transition matrix actually moved => M-step ran with correctly-scaled
        // (non-underflowed) gamma.
        let moved = hmm
            .a
            .iter()
            .flatten()
            .zip(init_a.iter().flatten())
            .any(|(&x, &u)| (x - u).abs() > 1e-9);
        assert!(
            moved,
            "baum_welch did not update transition matrix on T=700"
        );
        // Inference on the long sequence must not panic or produce NaN.
        assert!(hmm.predict_state(&obs, 500).is_ok());
        assert!(hmm.smooth_state(&obs, 500).is_ok());
    }
}
