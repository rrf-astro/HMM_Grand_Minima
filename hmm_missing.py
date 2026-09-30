"""
hmm_missing.py
==============
1-D HMM on a regular annual grid, with missing years (NaN) treated as a
missing observation (emission likelihood = 1), and an optional
state-dependent AR(1) emission.

Model
-----
  s_t ~ Markov(pi, A)                      (step = 1 year, ALWAYS, including across gaps)
  ar=False: x_t | s_t            ~ N(mu_s, v_s)
  ar=True : x_t | x_{t-1}, s_t   ~ N(mu_s + phi_s (x_{t-1} - mu_s), v_s (1 - phi_s^2))
            (uses the marginal N(mu_s, v_s) variant when x_{t-1} is missing or t = 0)

  * mu_s, v_s = the state's MARGINAL mean and variance; v_s(1-phi_s^2) = conditional variance.
  * Pure maximum likelihood (no priors); variance floor `min_var`.
  * When ar=True the M-step uses only the (x_{t-1}, x_t) pairs that are both
    observed (generalized EM / conditional likelihood); the reported
    log-likelihood is that of the full model (including the marginal terms).

Minimal usage
-------------
    from hmm_missing import fit_restarts, bic, bootstrap_lrt
    x = annual_full["log1p_mean_groups"].values          # 401 values, NaN at the gaps
    res = fit_restarts(x, K=3, ar=False, n_restarts=50)
    m = res["best_model"]                                 # states ordered by increasing mu (0=C,1=B,2=A)
    path = m.viterbi(x); post = m.posterior(x)            # both on the FULL 401-year grid
"""
import numpy as np
from scipy.special import ndtr, ndtri

LOG2PI = np.log(2.0 * np.pi)


class MissingHMM:
    def __init__(self, n_states, ar=False, n_iter=200, tol=1e-4, min_var=1e-4):
        self.K, self.ar, self.n_iter, self.tol, self.min_var = n_states, ar, n_iter, tol, min_var

    # ------------------------------------------------------------------ emissions
    def _logB(self, x):
        T, K = len(x), self.K
        obs = ~np.isnan(x)
        prev = np.r_[np.nan, x[:-1]]
        paired = obs & ~np.isnan(prev)
        logB = np.zeros((T, K))
        with np.errstate(invalid="ignore"):
            for s in range(K):
                m, v = self.mu[s], self.var[s]
                lp = -0.5 * (LOG2PI + np.log(v) + (x - m) ** 2 / v)
                if self.ar:
                    ph = self.phi[s]
                    cv = v * (1 - ph ** 2)
                    cm = m + ph * (prev - m)
                    lpc = -0.5 * (LOG2PI + np.log(cv) + (x - cm) ** 2 / cv)
                    lp = np.where(paired, lpc, lp)
                logB[:, s] = np.where(obs, lp, 0.0)
        return logB

    # ------------------------------------------------------------- forward-backward (scaled)
    def _fb(self, logB):
        T, K = logB.shape
        mx = logB.max(axis=1, keepdims=True)
        B = np.exp(logB - mx)
        alpha = np.empty((T, K)); c = np.empty(T)
        a = self.pi * B[0]; c[0] = a.sum(); alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * B[t]
            c[t] = a.sum(); alpha[t] = a / c[t]
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        ll = np.log(c).sum() + mx.sum()
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        W = (B[1:] * beta[1:]) / c[1:, None]
        xi = self.A * (alpha[:-1].T @ W)
        return ll, gamma, xi

    # ------------------------------------------------------------------ M-step
    def _mstep(self, x, gamma, xi):
        K = self.K
        obs = ~np.isnan(x)
        prev = np.r_[np.nan, x[:-1]]
        paired = obs & ~np.isnan(prev)
        self.pi = gamma[0] / gamma[0].sum()
        self.A = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        for s in range(K):
            done = False
            if self.ar and gamma[paired, s].sum() > 5:
                w, y, z = gamma[paired, s], x[paired], prev[paired]
                wm = w.sum(); my = (w * y).sum() / wm; mz = (w * z).sum() / wm
                vz = (w * (z - mz) ** 2).sum() / wm
                cz = (w * (z - mz) * (y - my)).sum() / wm
                phi = float(np.clip(cz / vz if vz > 1e-12 else 0.0, -0.98, 0.98))
                cst = my - phi * mz
                e = max((w * (y - cst - phi * z) ** 2).sum() / wm, self.min_var)
                self.phi[s] = phi
                self.mu[s] = cst / (1 - phi)
                self.var[s] = e / (1 - phi ** 2)
                done = True
            if not done:
                w = gamma[obs, s]; y = x[obs]
                wm = max(w.sum(), 1e-12)
                self.mu[s] = (w * y).sum() / wm
                self.var[s] = max((w * (y - self.mu[s]) ** 2).sum() / wm, self.min_var)
                self.phi[s] = 0.0

    # --------------------------------------------------------------------- fit
    def fit(self, x, seed=0):
        rng = np.random.default_rng(seed)
        K = self.K
        xo = x[~np.isnan(x)]
        self.mu = np.sort(rng.choice(xo, K, replace=False)) + rng.normal(0, 0.01, K)
        self.var = np.full(K, xo.var() / K + self.min_var)
        self.phi = np.zeros(K)
        A = 0.6 * np.eye(K) + rng.dirichlet(np.ones(K), K) * 0.4
        self.A = A / A.sum(axis=1, keepdims=True)
        self.pi = rng.dirichlet(np.ones(K))
        prev_ll, self.converged, self.ll_path = -np.inf, False, []
        for it in range(self.n_iter):
            ll, gamma, xi = self._fb(self._logB(x))
            self.ll_path.append(ll)
            if abs(ll - prev_ll) < self.tol:
                self.converged = True
                break
            prev_ll = ll
            self._mstep(x, gamma, xi)
        self.loglik = self._fb(self._logB(x))[0]
        self.n_nonmonotone = int(np.sum(np.diff(self.ll_path) < -1e-6))
        self._reorder()
        return self

    def _reorder(self):
        o = np.argsort(self.mu)          # increasing: 0 = lowest (C), K-1 = highest (A)
        self.mu, self.var, self.phi = self.mu[o], self.var[o], self.phi[o]
        self.A, self.pi = self.A[np.ix_(o, o)], self.pi[o]

    # ----------------------------------------------------------- inference
    def posterior(self, x):
        return self._fb(self._logB(x))[1]

    def viterbi(self, x):
        logB = self._logB(x)
        T, K = logB.shape
        lA = np.log(np.maximum(self.A, 1e-300))
        d = np.empty((T, K)); psi = np.zeros((T, K), int)
        d[0] = np.log(np.maximum(self.pi, 1e-300)) + logB[0]
        for t in range(1, T):
            cand = d[t - 1][:, None] + lA
            psi[t] = cand.argmax(axis=0)
            d[t] = cand.max(axis=0) + logB[t]
        path = np.empty(T, int); path[-1] = d[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def sample(self, T, rng):
        """Simulate the FULL grid of T years (mask with NaN afterwards if needed)."""
        K = self.K
        s = np.empty(T, int); x = np.empty(T)
        s[0] = rng.choice(K, p=self.pi)
        x[0] = rng.normal(self.mu[s[0]], np.sqrt(self.var[s[0]]))
        for t in range(1, T):
            s[t] = rng.choice(K, p=self.A[s[t - 1]])
            m, v = self.mu[s[t]], self.var[s[t]]
            if self.ar:
                ph = self.phi[s[t]]
                x[t] = rng.normal(m + ph * (x[t - 1] - m), np.sqrt(v * (1 - ph ** 2)))
            else:
                x[t] = rng.normal(m, np.sqrt(v))
        return x, s

    def pseudo_residuals(self, x):
        """One-step-ahead normal forecast pseudo-residuals (Zucchini et al. 2016); ~N(0,1) if the model fits well."""
        logB = self._logB(x)
        T, K = logB.shape
        obs = ~np.isnan(x)
        z = np.full(T, np.nan)
        filt = None
        for t in range(T):
            pred = self.pi if t == 0 else filt @ self.A
            if obs[t]:
                m, sd = self.mu.copy(), np.sqrt(self.var)
                if self.ar and t > 0 and not np.isnan(x[t - 1]):
                    m = self.mu + self.phi * (x[t - 1] - self.mu)
                    sd = np.sqrt(self.var * (1 - self.phi ** 2))
                u = float(np.sum(pred * ndtr((x[t] - m) / sd)))
                z[t] = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
                w = pred * np.exp(logB[t] - logB[t].max())
                filt = w / w.sum()
            else:
                filt = pred
        return z

    def n_params(self):
        K = self.K
        return K * (K - 1) + (K - 1) + 2 * K + (K if self.ar else 0)


# ---------------------------------------------------------------------- utilities
def fit_restarts(x, K, ar=False, n_restarts=50, seed0=0, **kw):
    best, hist = None, []
    for sd in range(seed0, seed0 + n_restarts):
        try:
            m = MissingHMM(K, ar=ar, **kw).fit(x, seed=sd)
            hist.append((sd, m.loglik, m.converged, m.n_nonmonotone))
            if best is None or m.loglik > best.loglik:
                best = m
        except Exception:
            hist.append((sd, np.nan, False, 0))
    ll = np.array([h[1] for h in hist if not np.isnan(h[1])])
    return {"best_model": best, "best_logL": best.loglik, "best_seed": [h[0] for h in hist if h[1] == best.loglik][0],
            "history": hist, "n_failed": int(np.sum([np.isnan(h[1]) for h in hist])),
            "n_not_converged": int(np.sum([not h[2] for h in hist if not np.isnan(h[1])])),
            "n_nonmonotone_restarts": int(np.sum([h[3] > 0 for h in hist])),
            "frac_near_best": float(np.mean(ll > best.loglik - 0.01)),
            "logL_std": float(ll.std())}


def bic(model, x):
    n = int(np.sum(~np.isnan(x)))
    return -2 * model.loglik + model.n_params() * np.log(n)


def _one_rep(seed, m0, x_mask, K0, K1, ar, n_restarts):
    rng = np.random.default_rng(10_000 + seed)
    xs, _ = m0.sample(len(x_mask), rng)
    xs = np.where(x_mask, xs, np.nan)
    l0 = fit_restarts(xs, K0, ar, n_restarts)["best_logL"]
    l1 = fit_restarts(xs, K1, ar, n_restarts)["best_logL"]
    return 2 * (l1 - l0)


def bootstrap_lrt(x, K0=3, K1=4, ar=False, B=199, n_restarts=20, n_jobs=1):
    """Parametric bootstrap LRT (McLachlan 1987) with the SAME number of restarts for the
    observed value and for the replicates. The missing-data pattern of `x` is preserved in
    the replicates. p = (1 + #{LRT_b >= LRT_obs}) / (B + 1)."""
    r0 = fit_restarts(x, K0, ar, n_restarts); r1 = fit_restarts(x, K1, ar, n_restarts)
    obs = 2 * (r1["best_logL"] - r0["best_logL"])
    mask = ~np.isnan(x)
    args = [(b, r0["best_model"], mask, K0, K1, ar, n_restarts) for b in range(B)]
    if n_jobs == 1:
        lrt = np.array([_one_rep(*a) for a in args])
    else:
        from joblib import Parallel, delayed
        lrt = np.array(Parallel(n_jobs=n_jobs)(delayed(_one_rep)(*a) for a in args))
    p = (1 + np.sum(lrt >= obs)) / (B + 1)
    return {"lrt_obs": obs, "lrt_null": lrt, "p_value": float(p), "B": B, "n_restarts": n_restarts}


def contiguous_runs(idx):
    """[(start, end), ...] from a list of integers (years or positions)."""
    idx = sorted(idx)
    if not idx:
        return []
    out, a, b = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - b > 1:
            out.append((a, b)); a = v
        b = v
    out.append((a, b))
    return out
