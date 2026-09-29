"""
RL decision policy: replaces the closed-form "expected F0.5" cutoff rule
with a learned sequential stopping policy, trained by policy gradient
(REINFORCE with a self-critical baseline) to directly optimize the actual,
non-differentiable macro F0.5 metric.

Why RL fits this sub-problem specifically: macro F0.5 is computed per S1
entity from a *discrete* accepted set, with sharp special cases (T empty &
P empty -> 1.0; T empty & P non-empty -> 0.0). That reward is not a smooth
function of any single candidate's score, so it can't be backpropagated
through the way BCE can. Framing "how many of the sorted candidates to
keep" as a sequential decision -- accept the next candidate or stop --
turns this into a standard episodic RL problem where the *true* metric is
the reward, no surrogate loss required.

Episode (per S1 entity):
  - Candidates sorted by DL-matcher probability, descending, capped at
    MAX_STEPS.
  - At step t the agent observes state_t (rank, this candidate's prob,
    running totals, gaps -- see `build_state`) and samples an action:
    CONTINUE (accept candidate t, move to t+1) or STOP (stop now, reject
    candidate t and everything after it).
  - Terminal reward = the true F0.5 of the accepted set for that S1
    (computed against ground truth during training).
  - Policy: a tiny 2-layer MLP (numpy) outputting P(continue | state_t).

Training: for each S1 in the training batch, roll out N_ROLLOUTS stochastic
episodes, use the batch's mean reward as a self-critical baseline (as in
Rennie et al.'s SCST) to reduce variance, and take a REINFORCE gradient
step. An entropy bonus keeps exploration alive early on.

At inference time the policy is run greedily (argmax action at every step,
i.e. threshold the continue-probability at 0.5) for a deterministic,
reproducible cutoff per S1.
"""
import numpy as np

from nn_numpy import Dense, Adam

MAX_STEPS = 15
N_ROLLOUTS = 6
ENTROPY_COEF = 0.01


def f_beta(pred_ids: set, true_ids: set, beta=0.5) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_ids)
    rec = tp / len(true_ids)
    b2 = beta ** 2
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def build_state(probs_sorted: np.ndarray, t: int) -> np.ndarray:
    """State features for step t (0-indexed) given the full sorted prob
    array for this S1's candidates (already capped/padded to MAX_STEPS)."""
    n = len(probs_sorted)
    p_t = probs_sorted[t] if t < n else 0.0
    p_next = probs_sorted[t + 1] if t + 1 < n else 0.0
    p_top = probs_sorted[0] if n > 0 else 0.0
    cum = probs_sorted[:t].sum() if t > 0 else 0.0
    remaining = probs_sorted[t:].sum() if t < n else 0.0
    return np.array([
        p_t,
        t / MAX_STEPS,
        cum / max(t, 1),
        remaining,
        p_top - p_t,
        p_t - p_next,
        n / MAX_STEPS,
    ], dtype=np.float32)


STATE_DIM = 7


class StoppingPolicy:
    def __init__(self, hidden=32, lr=1e-2):
        self.fc1 = Dense(STATE_DIM, hidden, act="relu")
        self.fc2 = Dense(hidden, 1, act="sigmoid")
        self.opt = Adam(lr=lr)

    def continue_prob(self, states: np.ndarray) -> np.ndarray:
        h = self.fc1.forward(states)
        p = self.fc2.forward(h)[:, 0]
        return p, h

    def _backward(self, d_p, h_cache_valid=True):
        d_h, grad2 = self.fc2.backward(d_p[:, None])
        _, grad1 = self.fc1.backward(d_h)
        return {
            "fc1.W": (self.fc1.W, grad1["W"]), "fc1.b": (self.fc1.b, grad1["b"]),
            "fc2.W": (self.fc2.W, grad2["W"]), "fc2.b": (self.fc2.b, grad2["b"]),
        }

    def rollout(self, probs_sorted: np.ndarray, greedy=False, rng=None):
        """Run one episode over this S1's sorted candidate probabilities.
        Returns: accepted_count, list of (state, action, continue_prob)."""
        rng = rng or np.random.default_rng()
        n = min(len(probs_sorted), MAX_STEPS)
        trace = []
        for t in range(n):
            state = build_state(probs_sorted, t)
            cp, _ = self.continue_prob(state[None, :])
            cp = float(cp[0])
            if greedy:
                action = 1 if cp >= 0.5 else 0
            else:
                action = 1 if rng.random() < cp else 0
            trace.append((state, action, cp))
            if action == 0:
                return t, trace  # stop before accepting candidate t
        return n, trace  # accepted everything up to the cap

    def train_step(self, episodes):
        """
        episodes: list of dicts with keys
          'trace' (list of (state, action, cp)), 'reward' (float)
        Applies REINFORCE with a batch-mean self-critical baseline.
        """
        baseline = np.mean([e["reward"] for e in episodes])
        states, d_ps = [], []
        for ep in episodes:
            advantage = ep["reward"] - baseline
            for state, action, cp in ep["trace"]:
                # d(-log pi(a|s))/d(cp): for a=1 (continue), logp = log(cp);
                # for a=0 (stop), logp = log(1-cp).
                cp_c = min(max(cp, 1e-6), 1 - 1e-6)
                if action == 1:
                    dlogp_dcp = 1.0 / cp_c
                else:
                    dlogp_dcp = -1.0 / (1 - cp_c)
                # gradient ASCENT on advantage-weighted logprob + entropy bonus
                entropy_grad = ENTROPY_COEF * (-(1 - 2 * cp_c))
                grad_cp = -(advantage * dlogp_dcp) - entropy_grad
                states.append(state)
                d_ps.append(grad_cp)
        if not states:
            return 0.0
        states = np.stack(states)
        d_ps = np.array(d_ps, dtype=np.float32) / len(states)
        # re-run forward to populate layer caches consistently with d_ps order
        self.continue_prob(states)
        grads = self._backward(d_ps)
        self.opt.step(grads)
        return float(baseline)

    def save_state(self):
        return {"fc1": self.fc1.params(), "fc2": self.fc2.params()}

    def load_state(self, state):
        self.fc1.set_params(state["fc1"])
        self.fc2.set_params(state["fc2"])


def train_policy(matcher_probs_by_s1: dict, true_by_s1: dict, other_ids_by_s1: dict,
                  epochs=25, seed=0, verbose=True):
    """
    matcher_probs_by_s1: {s1_id: np.array of candidate probs, sorted desc}
    other_ids_by_s1:      {s1_id: list of candidate ids, in the SAME sorted order}
    true_by_s1:           {s1_id: set of true matching ids}
    """
    policy = StoppingPolicy()
    rng = np.random.default_rng(seed)
    s1_ids = list(matcher_probs_by_s1.keys())

    for ep in range(epochs):
        rng.shuffle(s1_ids)
        epoch_rewards = []
        batch_episodes = []
        for s1 in s1_ids:
            probs = matcher_probs_by_s1[s1]
            ids = other_ids_by_s1[s1]
            true_set = true_by_s1.get(s1, set())
            rollouts = []
            for _ in range(N_ROLLOUTS):
                k, trace = policy.rollout(probs, greedy=False, rng=rng)
                pred_set = set(ids[:k])
                reward = f_beta(pred_set, true_set)
                rollouts.append({"trace": trace, "reward": reward})
                epoch_rewards.append(reward)
            batch_episodes.extend(rollouts)
            if len(batch_episodes) >= 64:
                policy.train_step(batch_episodes)
                batch_episodes = []
        if batch_episodes:
            policy.train_step(batch_episodes)
        if verbose:
            print(f"[rl_decision] epoch {ep+1}/{epochs}  "
                  f"mean rollout F0.5={np.mean(epoch_rewards):.4f}")
    return policy


def evaluate_policy(policy: StoppingPolicy, matcher_probs_by_s1: dict,
                     true_by_s1: dict, other_ids_by_s1: dict):
    scores = []
    for s1, probs in matcher_probs_by_s1.items():
        ids = other_ids_by_s1[s1]
        true_set = true_by_s1.get(s1, set())
        k, _ = policy.rollout(probs, greedy=True)
        pred_set = set(ids[:k])
        scores.append(f_beta(pred_set, true_set))
    # include S1 ids that had zero candidates at all (not in the dicts above)
    return float(np.mean(scores)) if scores else 0.0
