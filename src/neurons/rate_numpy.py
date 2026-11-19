"""Two-layer NumPy Elman recurrence and analytical BPTT from grow_rnn."""

import numpy as np


def orth_rho(n, rho, rng):
    """Draw an orthogonal matrix with initial spectral radius rho."""
    matrix = rng.standard_normal((n, n))
    q, r = np.linalg.qr(matrix)
    q = q * np.sign(np.diag(r))
    return q * rho


class NumpyElman2:
    """Two Elman layers of the same width followed by a linear readout.

    Matrices are [receiver, sender]. The given generator controls initialization
    in the same draw order as the original four-action prototype.
    """

    def __init__(self, rng, input_size, hidden_size, output_size, rho0=0.9):
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("model dimensions must be positive")
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.W1 = rng.standard_normal((hidden_size, input_size)) / np.sqrt(input_size)
        self.U1 = orth_rho(hidden_size, rho0, rng)
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.standard_normal((hidden_size, hidden_size)) / np.sqrt(hidden_size)
        self.U2 = orth_rho(hidden_size, rho0, rng)
        self.b2 = np.zeros(hidden_size)
        self.W3 = rng.standard_normal((output_size, hidden_size)) / np.sqrt(hidden_size)
        self.b3 = np.zeros(output_size)

    def keys(self):
        return ('W1', 'U1', 'b1', 'W2', 'U2', 'b2', 'W3', 'b3')

    def params(self):
        return [getattr(self, name) for name in self.keys()]

    def spectral_radius(self):
        r1 = float(np.abs(np.linalg.eigvals(self.U1)).max())
        r2 = float(np.abs(np.linalg.eigvals(self.U2)).max())
        return max(r1, r2), r1, r2

    def step(self, X, H1, H2, t):
        """Write the new hidden states at t+1 and return the tick's readout."""
        a1 = self.W1 @ X[t] + self.U1 @ H1[t] + self.b1
        H1[t + 1] = np.tanh(a1)
        a2 = self.W2 @ H1[t + 1] + self.U2 @ H2[t] + self.b2
        H2[t + 1] = np.tanh(a2)
        return self.W3 @ H2[t + 1] + self.b3

    def rollout(self, inputs, state=None):
        """Return logits and the two trajectories needed by ``backward``.

        Pass ``(H1[-1], H2[-1])`` as the next chunk's initial state.
        The trajectories include the supplied or zero initial states at index 0.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError("inputs must have shape [time, input_size]")
        shape = (inputs.shape[0] + 1, self.hidden_size)
        H1 = np.zeros(shape, dtype=self.W1.dtype)
        H2 = np.zeros(shape, dtype=self.W2.dtype)
        if state is not None:
            h1, h2 = state
            if h1.shape != (self.hidden_size,) or h2.shape != (self.hidden_size,):
                raise ValueError("each initial state must have shape [hidden_size]")
            H1[0], H2[0] = h1, h2
        outputs = np.empty((inputs.shape[0], self.output_size), dtype=self.W3.dtype)
        for t in range(inputs.shape[0]):
            outputs[t] = self.step(inputs, H1, H2, t)
        return outputs, H1, H2

    def backward(self, X, H1, H2, dZ, state_mask=None):
        """Return parameter gradients over one cached sequence.

        Where ``state_mask[t]`` is False, the output gradient at t updates the
        readout only. State gradients from later ticks still cross that tick.
        The initial states are treated as fixed at the start of the sequence.
        """
        grads = {name: np.zeros_like(value) for name, value in zip(self.keys(), self.params())}
        dH1 = np.zeros(self.hidden_size)
        dH2 = np.zeros(self.hidden_size)
        for t in range(X.shape[0] - 1, -1, -1):
            grads['W3'] += np.outer(dZ[t], H2[t + 1])
            grads['b3'] += dZ[t]
            if state_mask is None or state_mask[t]:
                dH2 = dH2 + self.W3.T @ dZ[t]
            da2 = dH2 * (1.0 - H2[t + 1] ** 2)
            grads['W2'] += np.outer(da2, H1[t + 1])
            grads['U2'] += np.outer(da2, H2[t])
            grads['b2'] += da2
            dH1 = dH1 + self.W2.T @ da2
            dH2 = self.U2.T @ da2
            da1 = dH1 * (1.0 - H1[t + 1] ** 2)
            grads['W1'] += np.outer(da1, X[t])
            grads['U1'] += np.outer(da1, H1[t])
            grads['b1'] += da1
            dH1 = self.U1.T @ da1
        return grads

    def renorm(self, rho_target):
        """Rescale the two recurrent matrices together and return the previous maximum radius."""
        rho_before, _, _ = self.spectral_radius()
        if rho_before > 1e-12:
            scale = rho_target / rho_before
            self.U1 *= scale
            self.U2 *= scale
        return rho_before
