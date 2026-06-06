//! # ternary-hmm
//!
//! Hidden Markov Model with ternary states {-1, 0, +1} and ternary emissions.
//! Implements forward algorithm, backward algorithm, Viterbi decoding,
//! Baum-Welch training (ternary-constrained), and sequence likelihood.

/// A ternary value: -1, 0, or +1.
pub type Trit = i8;

/// The three possible states.
pub const STATES: [Trit; 3] = [-1, 0, 1];

/// Map a trit to an index (0, 1, 2).
pub fn trit_to_index(t: Trit) -> usize {
    match t {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => panic!("Invalid trit: {}", t),
    }
}

/// Map an index back to a trit.
pub fn index_to_trit(i: usize) -> Trit {
    STATES[i]
}

/// Validate that a slice contains only valid trits.
pub fn validate_ternary(seq: &[Trit]) -> Result<(), String> {
    for (i, &t) in seq.iter().enumerate() {
        if t != -1 && t != 0 && t != 1 {
            return Err(format!("Invalid trit {} at position {}", t, i));
        }
    }
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
        // Validate probabilities sum to ~1
        let pi_sum: f64 = pi.iter().sum();
        if (pi_sum - 1.0).abs() > 1e-6 {
            return Err(format!("Initial probabilities must sum to 1, got {}", pi_sum));
        }
        for (i, row) in a.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("Transition row {} must sum to 1, got {}", i, sum));
            }
        }
        for (i, row) in b.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("Emission row {} must sum to 1, got {}", i, sum));
            }
        }
        Ok(Self { pi, a, b })
    }

    /// Forward algorithm: compute alpha[t][i] = P(O_0..O_t, state_t = i).
    /// Returns the forward matrix and the sequence likelihood P(O).
    pub fn forward(&self, obs: &[Trit]) -> Result<(Vec<[f64; 3]>, f64), String> {
        validate_ternary(obs)?;
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
    pub fn backward(&self, obs: &[Trit]) -> Result<Vec<[f64; 3]>, String> {
        validate_ternary(obs)?;
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
    pub fn sequence_likelihood(&self, obs: &[Trit]) -> Result<f64, String> {
        let (_, likelihood) = self.forward(obs)?;
        Ok(likelihood)
    }

    /// Viterbi algorithm: find the most likely state sequence.
    /// Returns (path, log_probability).
    pub fn viterbi(&self, obs: &[Trit]) -> Result<(Vec<Trit>, f64), String> {
        validate_ternary(obs)?;
        let t = obs.len();
        if t == 0 {
            return Err("Observation sequence is empty".into());
        }

        let neg_inf = f64::NEG_INFINITY;
        let mut delta = vec![[neg_inf; 3]; t];
        let mut psi = vec![[0usize; 3]; t];

        let o0 = trit_to_index(obs[0]);

        // Initialization (log space)
        for i in 0..3 {
            let p = self.pi[i] * self.b[i][o0];
            delta[0][i] = if p > 0.0 { p.ln() } else { neg_inf };
        }

        // Recursion
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

        // Termination
        let mut best_val = neg_inf;
        let mut best_i = 0;
        for i in 0..3 {
            if delta[t - 1][i] > best_val {
                best_val = delta[t - 1][i];
                best_i = i;
            }
        }

        // Backtrack
        let mut path = vec![0usize; t];
        path[t - 1] = best_i;
        for t_idx in (0..t - 1).rev() {
            path[t_idx] = psi[t_idx + 1][path[t_idx + 1]];
        }

        let state_path: Vec<Trit> = path.iter().map(|&i| index_to_trit(i)).collect();
        Ok((state_path, best_val))
    }

    /// Baum-Welch training with ternary constraints.
    /// Runs EM iterations to maximize the likelihood of the observation sequence.
    /// Transitions and emissions are constrained to ternary states.
    pub fn baum_welch(&mut self, obs: &[Trit], max_iter: usize, tol: f64) -> Result<Vec<f64>, String> {
        validate_ternary(obs)?;
        let t_len = obs.len();
        if t_len == 0 {
            return Err("Observation sequence is empty".into());
        }

        let mut likelihoods = Vec::new();

        for _iter in 0..max_iter {
            // E-step
            let (alpha, likelihood) = self.forward(obs)?;
            let beta = self.backward(obs)?;
            likelihoods.push(likelihood);

            if likelihood == 0.0 {
                break;
            }

            // Compute gamma[t][i] = P(state_t = i | O)
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

            // Compute xi[t][i][j] = P(state_t = i, state_{t+1} = j | O)
            let mut xi = vec![[[0.0f64; 3]; 3]; t_len - 1];
            for t_idx in 0..t_len - 1 {
                let o_next = trit_to_index(obs[t_idx + 1]);
                let mut sum = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        xi[t_idx][i][j] = alpha[t_idx][i] * self.a[i][j]
                            * self.b[j][o_next]
                            * beta[t_idx + 1][j];
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

            // M-step
            // Update pi
            for i in 0..3 {
                self.pi[i] = gamma[0][i];
            }

            // Update transition matrix
            for i in 0..3 {
                let denom: f64 = (0..t_len - 1).map(|t_idx| gamma[t_idx][i]).sum();
                if denom > 0.0 {
                    for j in 0..3 {
                        let numer: f64 = (0..t_len - 1).map(|t_idx| xi[t_idx][i][j]).sum();
                        self.a[i][j] = numer / denom;
                    }
                }
            }

            // Update emission matrix
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

            // Check convergence
            if likelihoods.len() >= 2 {
                let prev = likelihoods[likelihoods.len() - 2];
                if (likelihood - prev).abs() < tol {
                    break;
                }
            }
        }

        Ok(likelihoods)
    }

    /// Predict the most likely state at time t given observations up to time t.
    /// Uses forward probabilities (filtering).
    pub fn predict_state(&self, obs: &[Trit], t: usize) -> Result<Trit, String> {
        let (alpha, _) = self.forward(obs)?;
        if t >= alpha.len() {
            return Err(format!("t={} out of range (len={})", t, alpha.len()));
        }
        let best = alpha[t]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        Ok(index_to_trit(best))
    }

    /// Smoothing: predict the most likely state at time t given ALL observations.
    /// Uses forward-backward (gamma).
    pub fn smooth_state(&self, obs: &[Trit], t: usize) -> Result<Trit, String> {
        let (alpha, likelihood) = self.forward(obs)?;
        let beta = self.backward(obs)?;
        if t >= alpha.len() {
            return Err(format!("t={} out of range (len={})", t, alpha.len()));
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_hmm() -> TernaryHMM {
        // A biased coin-like model:
        // State -1: prefers emitting -1
        // State 0: prefers emitting 0
        // State +1: prefers emitting +1
        // With high self-transition probability
        let pi = [0.2, 0.5, 0.3];
        let a = [
            [0.7, 0.2, 0.1], // from -1
            [0.1, 0.8, 0.1], // from 0
            [0.1, 0.2, 0.7], // from +1
        ];
        let b = [
            [0.8, 0.1, 0.1], // state -1 emits -1 mostly
            [0.1, 0.8, 0.1], // state 0 emits 0 mostly
            [0.1, 0.1, 0.8], // state +1 emits +1 mostly
        ];
        TernaryHMM::with_params(pi, a, b).unwrap()
    }

    #[test]
    fn test_forward_probabilities_sum() {
        let hmm = make_simple_hmm();
        let obs = vec![1, 1, -1, 0, 1];
        let (alpha, likelihood) = hmm.forward(&obs).unwrap();

        // Likelihood should be > 0
        assert!(likelihood > 0.0, "Likelihood should be positive");
        // Likelihood should be < 1
        assert!(likelihood < 1.0, "Likelihood should be < 1");
        // All alpha values should be non-negative
        for row in &alpha {
            for &v in row {
                assert!(v >= 0.0, "Alpha values should be non-negative");
            }
        }
    }

    #[test]
    fn test_forward_backward_consistency() {
        let hmm = make_simple_hmm();
        let obs = vec![1, 0, -1, 0, 1];
        let (alpha, fwd_likelihood) = hmm.forward(&obs).unwrap();
        let beta = hmm.backward(&obs).unwrap();

        // P(O) computed via backward: sum_i pi[i] * b[i][o0] * beta[0][i]
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
        // Emit all +1: most likely state is +1 throughout
        let obs = vec![1, 1, 1, 1];
        let (path, _log_prob) = hmm.viterbi(&obs).unwrap();

        for (t, &state) in path.iter().enumerate() {
            assert_eq!(state, 1, "State at t={} should be +1, got {}", t, state);
        }
    }

    #[test]
    fn test_viterbi_switching() {
        let hmm = make_simple_hmm();
        // Emit -1 then +1: states should switch
        let obs = vec![-1, -1, -1, 1, 1, 1];
        let (path, _) = hmm.viterbi(&obs).unwrap();

        assert_eq!(path[0], -1, "First state should be -1");
        assert_eq!(path[5], 1, "Last state should be +1");
    }

    #[test]
    fn test_baum_welch_improves_likelihood() {
        let mut hmm = TernaryHMM::new(); // Start with uniform
        let obs = vec![1, 1, 1, 0, 0, -1, -1, -1, 1, 1];
        let initial_likelihood = hmm.sequence_likelihood(&obs).unwrap();

        let likelihoods = hmm.baum_welch(&obs, 50, 1e-8).unwrap();

        assert!(likelihoods.len() > 1, "Should have multiple iterations");
        let final_likelihood = *likelihoods.last().unwrap();
        assert!(
            final_likelihood >= initial_likelihood - 1e-10,
            "Likelihood should not decrease: initial={}, final={}",
            initial_likelihood, final_likelihood
        );
    }

    #[test]
    fn test_known_sequence_decoding() {
        // Simple model where state = emission with high probability
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

        let obs = vec![-1, 0, 1];
        let (path, _) = hmm.viterbi(&obs).unwrap();

        // With strong diagonal emission, state should match emission
        assert_eq!(path, vec![-1, 0, 1]);
    }

    #[test]
    fn test_manual_verification_small() {
        // Manually verifiable HMM
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

        // Single observation: o = 1 (index 2)
        let obs = vec![1];
        let (alpha, likelihood) = hmm.forward(&obs).unwrap();

        // alpha[0][i] = pi[i] * b[i][2]
        let expected_alpha = [
            pi[0] * b[0][2], // 0.5 * 0.1 = 0.05
            pi[1] * b[1][2], // 0.3 * 0.1 = 0.03
            pi[2] * b[2][2], // 0.2 * 0.7 = 0.14
        ];
        let expected_likelihood = 0.05 + 0.03 + 0.14; // 0.22

        for i in 0..3 {
            assert!(
                (alpha[0][i] - expected_alpha[i]).abs() < 1e-10,
                "alpha[0][{}] = {}, expected {}",
                i, alpha[0][i], expected_alpha[i]
            );
        }
        assert!(
            (likelihood - expected_likelihood).abs() < 1e-10,
            "likelihood = {}, expected {}",
            likelihood, expected_likelihood
        );
    }

    #[test]
    fn test_manual_two_step() {
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

        // obs = [0, 1] => indices [1, 2]
        let obs = vec![0, 1];
        let (alpha, _likelihood) = hmm.forward(&obs).unwrap();

        // alpha[0][i] = pi[i] * b[i][1]
        let alpha0 = [
            0.5 * 0.2,  // 0.10
            0.3 * 0.8,  // 0.24
            0.2 * 0.2,  // 0.04
        ];

        for i in 0..3 {
            assert!((alpha[0][i] - alpha0[i]).abs() < 1e-10);
        }

        // alpha[1][j] = sum_i alpha[0][i] * a[i][j] * b[j][2]
        // j=0: (0.10*0.6 + 0.24*0.3 + 0.04*0.1) * 0.1 = (0.06+0.072+0.004)*0.1 = 0.0136
        // j=1: (0.10*0.2 + 0.24*0.4 + 0.04*0.3) * 0.1 = (0.02+0.096+0.012)*0.1 = 0.0128
        // j=2: (0.10*0.2 + 0.24*0.3 + 0.04*0.6) * 0.7 = (0.02+0.072+0.024)*0.7 = 0.0812
        let expected = [0.0136, 0.0128, 0.0812];
        for j in 0..3 {
            assert!(
                (alpha[1][j] - expected[j]).abs() < 1e-10,
                "alpha[1][{}] = {}, expected {}",
                j, alpha[1][j], expected[j]
            );
        }
    }

    #[test]
    fn test_predict_state() {
        let hmm = make_simple_hmm();
        let obs = vec![1, 1, 1];
        // After seeing all +1 emissions, filtered state should be +1
        assert_eq!(hmm.predict_state(&obs, 2).unwrap(), 1);
    }

    #[test]
    fn test_smooth_state() {
        let hmm = make_simple_hmm();
        let obs = vec![-1, -1, 0, 1, 1];
        // With full observation, smoothing at t=2 (emit 0) should favor state 0
        let state = hmm.smooth_state(&obs, 2).unwrap();
        assert_eq!(state, 0);
    }

    #[test]
    fn test_sequence_likelihood_single() {
        let hmm = make_simple_hmm();
        let obs = vec![1];
        let ll = hmm.sequence_likelihood(&obs).unwrap();
        // P(+1) = sum_i pi[i] * b[i][2] = 0.2*0.1 + 0.5*0.1 + 0.3*0.8 = 0.02+0.05+0.24 = 0.31
        assert!((ll - 0.31).abs() < 1e-10, "Expected 0.31, got {}", ll);
    }

    #[test]
    fn test_empty_observation_error() {
        let hmm = make_simple_hmm();
        assert!(hmm.forward(&[]).is_err());
        assert!(hmm.backward(&[]).is_err());
        assert!(hmm.viterbi(&[]).is_err());
    }

    #[test]
    fn test_invalid_trit_error() {
        assert!(validate_ternary(&[2]).is_err());
        assert!(validate_ternary(&[1, -1, 0]).is_ok());
    }

    #[test]
    fn test_hmm_validation() {
        // Bad pi
        assert!(TernaryHMM::with_params([1.0, 0.0, 0.5], [[1.0/3.0; 3]; 3], [[1.0/3.0; 3]; 3]).is_err());
        // Good params
        assert!(TernaryHMM::with_params([0.5, 0.3, 0.2],
            [[0.6, 0.2, 0.2], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]],
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]).is_ok());
    }

    #[test]
    fn test_baum_welch_convergence() {
        // Train a model and check that likelihoods are monotonically non-decreasing
        let mut hmm = TernaryHMM::new();
        let obs = vec![-1, -1, -1, 0, 0, 1, 1, 1, 1, 1];
        let likelihoods = hmm.baum_welch(&obs, 100, 1e-12).unwrap();

        for w in likelihoods.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-10,
                "Likelihood decreased: {} -> {}",
                w[0], w[1]
            );
        }
    }
}
