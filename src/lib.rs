//! # ternary-hmm
//!
//! Hidden Markov Model with ternary states {-1, 0, +1} and ternary emissions.
//! Implements forward algorithm, backward algorithm, Viterbi decoding,
//! Baum-Welch training (ternary-constrained), and sequence likelihood.
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
        Neutral  => 1,
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
    pub fn with_params(
        pi: [f64; 3],
        a: [[f64; 3]; 3],
        b: [[f64; 3]; 3],
    ) -> Result<Self, String> {
        let pi_sum: f64 = pi.iter().sum();
        if (pi_sum - 1.0).abs() > 1e-6 {
            return Err(format!("Initial probabilities must sum to 1, got {pi_sum}"));
        }
        for (i, row) in a.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("Transition row {i} must sum to 1, got {sum}"));
            }
        }
        for (i, row) in b.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("Emission row {i} must sum to 1, got {sum}"));
            }
        }
        Ok(Self { pi, a, b })
    }

    /// Forward algorithm: compute alpha[t][i] = P(O_0..O_t, state_t = i).
    pub fn forward(&self, obs: &[Ternary]) -> Result<(Vec<[f64; 3]>, f64), String> {
        let t = obs.len();
        if t == 0 {
            return Err("Observation sequence is empty".into());
        }

        let mut alpha = vec![[0.0f64; 3]; t];
        let o0 = trit_to_index(obs[0]);

        // Initialization
        for i in 0..3 {
            alpha[0][i] = self.pi[i] * self.b[i][o0];
        }

        // Induction
        for t_idx in 1..t {
            let o = trit_to_index(obs[t_idx]);
            for j in 0..3 {
                let mut sum = 0.0;
                for i in 0..3 {
                    sum += alpha[t_idx - 1][i] * self.a[i][j];
                }
                alpha[t_idx][j] = sum * self.b[j][o];
            }
        }

        // Termination: P(O) = sum of alpha[T-1][i]
        let likelihood: f64 = alpha[t - 1].iter().sum();
        Ok((alpha, likelihood))
    }

    /// Backward algorithm: compute beta[t][i] = P(O_{t+1}..O_{T-1} | state_t = i).
    pub fn backward(&self, obs: &[Ternary]) -> Result<Vec<[f64; 3]>, String> {
        let t = obs.len();
        if t == 0 {
            return Err("Observation sequence is empty".into());
        }

        let mut beta = vec![[0.0f64; 3]; t];

        // Initialization: beta[T-1][i] = 1
        for i in 0..3 {
            beta[t - 1][i] = 1.0;
        }

        // Induction
        for t_idx in (0..t - 1).rev() {
            let o_next = trit_to_index(obs[t_idx + 1]);
            for i in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    sum += self.a[i][j] * self.b[j][o_next] * beta[t_idx + 1][j];
                }
                beta[t_idx][i] = sum;
            }
        }

        Ok(beta)
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

        for i in 0..3 {
            let p = self.pi[i] * self.b[i][o0];
            delta[0][i] = if p > 0.0 { p.ln() } else { neg_inf };
        }

        for t_idx in 1..t {
            let o = trit_to_index(obs[t_idx]);
            for j in 0..3 {
                let b_j = self.b[j][o];
                if b_j <= 0.0 {
                    delta[t_idx][j] = neg_inf;
                    continue;
                }
                let ln_b = b_j.ln();
                let mut best_val = neg_inf;
                let mut best_i = 0;
                for i in 0..3 {
                    if delta[t_idx - 1][i] == neg_inf {
                        continue;
                    }
                    let val = delta[t_idx - 1][i] + self.a[i][j].ln() + ln_b;
                    if val > best_val {
                        best_val = val;
                        best_i = i;
                    }
                }
                delta[t_idx][j] = best_val;
                psi[t_idx][j] = best_i;
            }
        }

        let mut best_val = neg_inf;
        let mut best_i = 0;
        for i in 0..3 {
            if delta[t - 1][i] > best_val {
                best_val = delta[t - 1][i];
                best_i = i;
            }
        }

        let mut path = vec![0usize; t];
        path[t - 1] = best_i;
        for t_idx in (0..t - 1).rev() {
            path[t_idx] = psi[t_idx + 1][path[t_idx + 1]];
        }

        let state_path: Vec<Ternary> = path.iter().map(|&i| index_to_trit(i)).collect();
        Ok((state_path, best_val))
    }

    /// Baum-Welch training with ternary constraints.
    pub fn baum_welch(&mut self, obs: &[Ternary], max_iter: usize, tol: f64) -> Result<Vec<f64>, String> {
        let t_len = obs.len();
        if t_len == 0 {
            return Err("Observation sequence is empty".into());
        }

        let mut likelihoods = Vec::new();

        for _iter in 0..max_iter {
            let (alpha, likelihood) = self.forward(obs)?;
            let beta = self.backward(obs)?;
            likelihoods.push(likelihood);

            if likelihood == 0.0 {
                break;
            }

            let mut gamma = vec![[0.0f64; 3]; t_len];
            for t_idx in 0..t_len {
                let mut sum = 0.0;
                for i in 0..3 {
                    gamma[t_idx][i] = alpha[t_idx][i] * beta[t_idx][i];
                    sum += gamma[t_idx][i];
                }
                if sum > 0.0 {
                    for i in 0..3 {
                        gamma[t_idx][i] /= sum;
                    }
                }
            }

            let mut xi = vec![[[0.0f64; 3]; 3]; t_len - 1];
            for t_idx in 0..t_len - 1 {
                let o_next = trit_to_index(obs[t_idx + 1]);
                let mut sum = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        xi[t_idx][i][j] = alpha[t_idx][i] * self.a[i][j]
                            * self.b[j][o_next] * beta[t_idx + 1][j];
                        sum += xi[t_idx][i][j];
                    }
                }
                if sum > 0.0 {
                    for i in 0..3 {
                        for j in 0..3 {
                            xi[t_idx][i][j] /= sum;
                        }
                    }
                }
            }

            for i in 0..3 {
                self.pi[i] = gamma[0][i];
            }

            for i in 0..3 {
                let denom: f64 = (0..t_len - 1).map(|t_idx| gamma[t_idx][i]).sum();
                if denom > 0.0 {
                    for j in 0..3 {
                        let numer: f64 = (0..t_len - 1).map(|t_idx| xi[t_idx][i][j]).sum();
                        self.a[i][j] = numer / denom;
                    }
                }
            }

            for i in 0..3 {
                let denom: f64 = (0..t_len).map(|t_idx| gamma[t_idx][i]).sum();
                if denom > 0.0 {
                    for k in 0..3 {
                        let numer: f64 = (0..t_len)
                            .filter(|&t_idx| trit_to_index(obs[t_idx]) == k)
                            .map(|t_idx| gamma[t_idx][i])
                            .sum();
                        self.b[i][k] = numer / denom;
                    }
                }
            }

            if likelihoods.len() >= 2 {
                let prev = likelihoods[likelihoods.len() - 2];
                if (likelihood - prev).abs() < tol {
                    break;
                }
            }
        }

        Ok(likelihoods)
    }

    /// Filtering: predict state at time t given obs up to t.
    pub fn predict_state(&self, obs: &[Ternary], t: usize) -> Result<Ternary, String> {
        let (alpha, _) = self.forward(obs)?;
        if t >= alpha.len() {
            return Err(format!("t={t} out of range (len={})", alpha.len()));
        }
        let best = alpha[t]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        Ok(index_to_trit(best))
    }

    /// Smoothing: predict state at time t given ALL observations.
    pub fn smooth_state(&self, obs: &[Ternary], t: usize) -> Result<Ternary, String> {
        let (alpha, likelihood) = self.forward(obs)?;
        let beta = self.backward(obs)?;
        if t >= alpha.len() {
            return Err(format!("t={t} out of range (len={})", alpha.len()));
        }
        let mut best_i = 0;
        let mut best_prob = 0.0;
        for i in 0..3 {
            let prob = alpha[t][i] * beta[t][i] / likelihood;
            if prob > best_prob {
                best_prob = prob;
                best_i = i;
            }
        }
        Ok(index_to_trit(best_i))
    }
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
        let a = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ];
        let b = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ];
        TernaryHMM::with_params(pi, a, b).unwrap()
    }

    #[test]
    fn test_forward_probabilities_sum() {
        let hmm = make_simple_hmm();
        let obs = vec![Positive, Positive, Negative, Neutral, Positive];
        let (_alpha, likelihood) = hmm.forward(&obs).unwrap();
        assert!(likelihood > 0.0, "Likelihood should be positive");
        assert!(likelihood < 1.0, "Likelihood should be < 1");
    }

    #[test]
    fn test_forward_backward_consistency() {
        let hmm = make_simple_hmm();
        let obs = vec![Positive, Neutral, Negative, Neutral, Positive];
        let (alpha, fwd_likelihood) = hmm.forward(&obs).unwrap();
        let beta = hmm.backward(&obs).unwrap();

        let o0 = trit_to_index(obs[0]);
        let bwd_likelihood: f64 = (0..3)
            .map(|i| hmm.pi[i] * hmm.b[i][o0] * beta[0][i])
            .sum();

        assert!(
            (fwd_likelihood - bwd_likelihood).abs() < 1e-10,
            "Forward ({}) and backward ({}) likelihoods should match",
            fwd_likelihood, bwd_likelihood
        );
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
        let obs = vec![Positive, Positive, Positive, Neutral, Neutral, Negative, Negative, Negative, Positive, Positive];
        let initial_likelihood = hmm.sequence_likelihood(&obs).unwrap();
        let likelihoods = hmm.baum_welch(&obs, 50, 1e-8).unwrap();
        assert!(likelihoods.len() > 1, "Should have multiple iterations");
        let final_likelihood = *likelihoods.last().unwrap();
        assert!(
            final_likelihood >= initial_likelihood - 1e-10,
            "Likelihood should not decrease: initial={initial_likelihood}, final={final_likelihood}"
        );
    }

    #[test]
    fn test_known_sequence_decoding() {
        let pi = [1.0 / 3.0; 3];
        let a = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ];
        let b = [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ];
        let hmm = TernaryHMM::with_params(pi, a, b).unwrap();
        let obs = vec![Negative, Neutral, Positive];
        let (path, _) = hmm.viterbi(&obs).unwrap();
        assert_eq!(path, vec![Negative, Neutral, Positive]);
    }

    #[test]
    fn test_manual_verification_small() {
        let pi = [0.5, 0.3, 0.2];
        let a = [
            [0.6, 0.2, 0.2],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6],
        ];
        let b = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ];
        let hmm = TernaryHMM::with_params(pi, a, b).unwrap();
        let obs = vec![Positive];
        let (_alpha, likelihood) = hmm.forward(&obs).unwrap();
        let expected = 0.5 * 0.1 + 0.3 * 0.1 + 0.2 * 0.7;
        assert!((likelihood - expected).abs() < 1e-10, "likelihood = {likelihood}, expected {expected}");
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
        assert!(TernaryHMM::with_params(
            [1.0, 0.0, 0.5],
            [[1.0/3.0; 3]; 3],
            [[1.0/3.0; 3]; 3]
        ).is_err());
        assert!(TernaryHMM::with_params(
            [0.5, 0.3, 0.2],
            [[0.6, 0.2, 0.2], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]],
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
        ).is_ok());
    }

    #[test]
    fn test_baum_welch_convergence() {
        let mut hmm = TernaryHMM::new();
        let obs = vec![Negative, Negative, Negative, Neutral, Neutral, Positive, Positive, Positive, Positive, Positive];
        let likelihoods = hmm.baum_welch(&obs, 100, 1e-12).unwrap();
        for w in likelihoods.windows(2) {
            assert!(w[1] >= w[0] - 1e-10, "Likelihood decreased: {} -> {}", w[0], w[1]);
        }
    }
}
