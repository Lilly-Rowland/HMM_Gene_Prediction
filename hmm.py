import math
from utils import DNA_ALPHABET, NEG_INF, normalize_probs


class HMMModel:
    def __init__(self, name, states, start_probs, transition_probs, emission_probs, allowed_state_for_base=None):
        self.name = name
        self.states = states

        # Normalize start probabilities so they sum to 1
        self.start_probs = normalize_probs(start_probs)
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs

        # Optional rules for restricting certain states at certain positions
        self.allowed_state_for_base = allowed_state_for_base

        for s in self.states:
            # Make sure every state has transition probabilities
            if s not in self.transition_probs:
                raise ValueError(f"Missing transition probabilities for state {s}")
            self.transition_probs[s] = normalize_probs(self.transition_probs[s])

            # Make sure every state has emission probabilities
            if s not in self.emission_probs:
                raise ValueError(f"Missing emission probabilities for state {s}")
            self.emission_probs[s] = normalize_probs(self.emission_probs[s])

        # Precompute log probabilities for numerical stability
        self.log_start = {s: math.log(self.start_probs.get(s, 1e-15)) for s in self.states}
        self.log_trans = {
            s: {t: math.log(self.transition_probs[s].get(t, 1e-15)) for t in self.states}
            for s in self.states
        }
        self.log_emit = {
            s: {b: math.log(self.emission_probs[s].get(b, 1e-15)) for b in DNA_ALPHABET}
            for s in self.states
        }

    def emission_logp(self, state, base, position, sequence):
        # If state has positional/base constraints, enforce them here
        if self.allowed_state_for_base is not None and state in self.allowed_state_for_base:
            ok = self.allowed_state_for_base[state](position, sequence)
            if not ok:
                return NEG_INF  # impossible state at this position
        return self.log_emit[state][base]

    def viterbi(self, sequence):
        n = len(sequence)
        if n == 0:
            return []

        # dp[i][s] = best log-prob of ending in state s at position i
        dp = [{s: NEG_INF for s in self.states} for _ in range(n)]

        # back[i][s] stores best previous state leading to state s at position i
        back = [{s: None for s in self.states} for _ in range(n)]

        # Initialize first position using start probabilities
        base0 = sequence[0]
        for s in self.states:
            emit = self.emission_logp(s, base0, 0, sequence)
            dp[0][s] = self.log_start[s] + emit if emit > NEG_INF / 2 else NEG_INF

        # Fill DP table left to right
        for i in range(1, n):
            base = sequence[i]
            for curr in self.states:
                emit = self.emission_logp(curr, base, i, sequence)

                # Skip impossible emissions
                if emit <= NEG_INF / 2:
                    dp[i][curr] = NEG_INF
                    back[i][curr] = None
                    continue

                best_score = NEG_INF
                best_prev = None

                # Check all possible previous states
                for prev in self.states:
                    score = dp[i - 1][prev] + self.log_trans[prev][curr] + emit
                    if score > best_score:
                        best_score = score
                        best_prev = prev

                dp[i][curr] = best_score
                back[i][curr] = best_prev

        # Choose best ending state
        last_state = max(self.states, key=lambda s: dp[n - 1][s])

        # Backtrack to recover best state path
        path = [last_state]
        for i in range(n - 1, 0, -1):
            prev = back[i][path[-1]]
            if prev is None:
                prev = self.states[0]  # fallback if something is missing
            path.append(prev)

        path.reverse()
        return path