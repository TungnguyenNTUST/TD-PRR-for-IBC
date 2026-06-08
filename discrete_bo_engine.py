from typing import Sequence, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


class DiscreteBO:
    """
    Bayesian Optimization over a finite candidate set (columns of `Input`).
    - Input: array with shape (8, N) where rows are features x1..x8, columns are candidates.
    - Kernel: Matern 5/2 with user-specified anisotropic length-scales (8-dim), signal variance,
             and a WhiteKernel for noise variance.
    - Acquisition: Expected Improvement (EI) with exploration parameter xi.
    """

    def __init__(
        self,
        Input: np.ndarray,                         # shape (D, N)
        length_scales: Sequence[float],            # len = D, one per input
        sigma_f2: float,                           # signal variance
        sigma_n2: float = 1e-6,                    # noise variance
        ei_acq: bool = True,
        xi: float = 0.01,                          # EI exploration parameter
        kappa: float = 1.96,
        normalize_y: bool = True,
        random_state: Optional[int] = None,
        allow_hyperparam_optimization: bool = False,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),  # used only if optimization=True
        sigma_f_bounds: Tuple[float, float] = (1e-5, 1e8),
        sigma_n_bounds: Tuple[float, float] = (1e-12, 1e3),
        hopt_every: int = 10,  # re-optimize hyperparameters every K observations
        hopt_warmup: int = 10,  # start optimizing only after at least this many observations

    ):
        assert Input.ndim == 2, "Input must be shape (D, N)"
        self.Input = Input
        self.Xc = Input.T.copy()
        self.N, self.D = self.Xc.shape
        self.ei_acq = bool(ei_acq)
        self.xi = float(xi)
        self.kappa = float(kappa)
        self.normalize_y = bool(normalize_y)
        self.random_state = random_state

        # store hyperparams
        self.length_scales = np.asarray(length_scales, dtype=float)
        if self.length_scales.shape != (self.D,):
            raise ValueError(f"length_scales must have shape ({self.D},)")
        self.sigma_f2 = float(sigma_f2)
        self.sigma_n2 = float(sigma_n2)
        self.allow_hopt = bool(allow_hyperparam_optimization)
        self.ls_bounds = length_scale_bounds
        self.sf_bounds = sigma_f_bounds
        self.sn_bounds = sigma_n_bounds
        self.hopt_every = int(hopt_every)
        self.hopt_warmup = int(hopt_warmup)

        # observation containers
        self._chosen_idx: List[int] = []          # indices (0-based) of evaluated candidates
        self._X_obs: Optional[np.ndarray] = None  # (m, 8)
        self._y_obs: Optional[np.ndarray] = None  # (m,)

        self._gp: Optional[GaussianProcessRegressor] = None

    def _make_kernel(self,do_opt: bool):
        # ConstantKernel * Matern(nu=2.5) + WhiteKernel
        if self.allow_hopt and do_opt:
            k = (
                ConstantKernel(constant_value=self.sigma_f2, constant_value_bounds=self.sf_bounds)
                * Matern(length_scale=self.length_scales, length_scale_bounds=self.ls_bounds, nu=2.5)
                + WhiteKernel(noise_level=self.sigma_n2, noise_level_bounds=self.sn_bounds)
            )
            optimizer = "fmin_l_bfgs_b"
        else:
            k = (
                ConstantKernel(constant_value=self.sigma_f2, constant_value_bounds="fixed")
                * Matern(length_scale=self.length_scales, length_scale_bounds="fixed", nu=2.5)
                + WhiteKernel(noise_level=self.sigma_n2, noise_level_bounds="fixed")
            )
            optimizer = None
        return k, optimizer

    def _fit_gp(self):
        if self._X_obs is None or self._y_obs is None or len(self._y_obs) == 0:
            raise RuntimeError("No observations to fit. Call initialize() or tell() first.")

        m = len(self._y_obs)
        # Decide whether to run hyperparameter optimization this time
        do_opt = False
        if self.allow_hopt:
            if m >= self.hopt_warmup and (m % self.hopt_every == 0):
                do_opt = True

        kernel, optimizer = self._make_kernel(do_opt=do_opt)

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
            n_restarts_optimizer=5 if optimizer is not None else 0,  # recommended
            alpha=1e-12,
        )
        self._gp.fit(self._X_obs, self._y_obs)
        # If we just optimized, store the learned hyperparameters back into your class
        # so the next "fixed" fits use the updated values.
        if optimizer is not None:
            k_opt = self._gp.kernel_

            # k_opt is (ConstantKernel * Matern) + WhiteKernel
            # Extract parameters safely:
            try:
                self.sigma_f2 = float(k_opt.k1.k1.constant_value)
                self.length_scales = np.asarray(k_opt.k1.k2.length_scale, dtype=float)
                self.sigma_n2 = float(k_opt.k2.noise_level)
            except Exception:
                # If kernel structure differs, skip updating
                pass

    def initialize(self, init_indices_1based: Sequence[int], y_values: Sequence[float]):
        """
        Seed the GP with initial observations.
        - init_indices_1based: 1-based column indices into Input (e.g., [init1, init2])
        - y_values: objective values measured at those candidates
        """
        idx0 = [int(i) - 1 for i in init_indices_1based]
        if len(idx0) != len(y_values):
            raise ValueError("init_indices_1based and y_values must have same length.")
        if len(set(idx0)) != len(idx0):
            raise ValueError("Duplicate initial indices are not allowed.")
        if any(i < 0 or i >= self.N for i in idx0):
            raise ValueError("An initial index is out of range.")

        self._chosen_idx = idx0.copy()
        self._X_obs = self.Xc[self._chosen_idx, :]
        self._y_obs = np.asarray(y_values, dtype=float)
        self._fit_gp()

    def tell(self, index_1based: int, y_value: float):
        """Add one new observation and refit the GP."""
        i0 = int(index_1based) - 1
        if i0 in self._chosen_idx:
            raise ValueError(f"Candidate {index_1based} already observed.")
        if i0 < 0 or i0 >= self.N:
            raise ValueError("Index out of range.")

        self._chosen_idx.append(i0)
        self._X_obs = self.Xc[self._chosen_idx, :]
        self._y_obs = np.append(self._y_obs, float(y_value))
        self._fit_gp()

    def _ei(self, mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
        """Expected Improvement (maximization)."""
        # Add small floor to sigma to avoid divide-by-zero
        sigma = np.maximum(sigma, 1e-12)
        imp = mu - y_best - self.xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        # If sigma was numerically ~0, EI should be ~0
        ei[sigma <= 1e-12] = 0.0
        return ei

    def _ucb(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound (maximization)."""
        sigma = np.maximum(sigma, 1e-12)
        return mu + self.kappa * sigma

    def _acq(self, mu: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.ei_acq:
            y_best = float(np.max(self._y_obs))
            return self._ei(mu, std, y_best), "EI"
        return self._ucb(mu, std), "UCB"

    def suggest_next(self, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self._gp is None:
            raise RuntimeError("Model not fitted. Call initialize() first.")

        mask = self._unobserved_mask()
        Xq = self.Xc[mask, :]
        mu, std = self._gp.predict(Xq, return_std=True)

        if self.ei_acq:
            y_best = float(np.max(self._y_obs))
            acq = self._ei(mu, std, y_best)
        else:
            acq = self._ucb(mu, std)

        unobs_idx = np.where(mask)[0]
        order = np.argsort(-acq)[:top_k]
        chosen_unobs = unobs_idx[order]
        return chosen_unobs + 1, acq[order]

    def rank_candidates(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Rank ALL unobserved candidates by acquisition value (EI or UCB, descending).

        Returns a DataFrame with columns:
        - index_1based
        - ACQ (EI or UCB value)
        - x1..xD (features)
        """
        if self._gp is None:
            raise RuntimeError("Model not fitted. Call initialize() first.")

        # 1) mask out observed candidates
        mask = np.ones(self.N, dtype=bool)
        mask[self._chosen_idx] = False

        # 2) predict GP on unobserved candidates
        Xq = self.Xc[mask, :]
        mu, std = self._gp.predict(Xq, return_std=True)

        # 3) compute acquisition
        if self.ei_acq:
            y_best = float(np.max(self._y_obs))
            acq = self._ei(mu, std, y_best)
            acq_name = "EI"
        else:
            acq = self._ucb(mu, std)
            acq_name = "UCB"

        # 4) rank by acquisition (descending)
        unobs_idx = np.where(mask)[0]
        order = np.argsort(-acq)
        if top_k is not None:
            order = order[:top_k]

        # 5) build output table
        idx1 = unobs_idx[order] + 1  # convert to 1-based indexing

        out = pd.DataFrame({
            "index_1based": idx1,
            acq_name: acq[order],
        })

        # 6) attach feature values x1..xD
        feats = self.Xc[unobs_idx[order], :]
        for d in range(self.D):
            out[f"x{d + 1}"] = feats[:, d]

        return out.reset_index(drop=True)

    def _unobserved_mask(self) -> np.ndarray:
        mask = np.ones(self.N, dtype=bool)
        mask[self._chosen_idx] = False
        return mask

    def candidates_meeting_target(
            self,
            eps: float,
            use_ucb_screen: bool = False,
            kappa_screen: float = 0.0,
    ) -> pd.DataFrame:
        """
        Return a table of unobserved candidates that satisfy the target condition.

        Target = y_best - eps

        If use_ucb_screen=True, screen by (mu + kappa_screen*std) >= Target (more conservative).
        Otherwise screen by mu >= Target (your spec).
        """
        if self._gp is None:
            raise RuntimeError("Model not fitted. Call initialize() first.")

        y_best = float(np.max(self._y_obs))
        target = y_best - float(eps)

        mask = self._unobserved_mask()
        Xq = self.Xc[mask, :]
        mu, std = self._gp.predict(Xq, return_std=True)

        if use_ucb_screen:
            score = mu + float(kappa_screen) * std
            keep = score >= target
        else:
            keep = mu >= target

        unobs_idx = np.where(mask)[0]
        kept_idx0 = unobs_idx[keep]  # 0-based indices in [0..N-1]
        kept_mu = mu[keep]
        kept_std = std[keep]

        # Build a ranked view (for debug)
        out = pd.DataFrame({
            "index_1based": kept_idx0 + 1,
            "mu": kept_mu,
            "std": kept_std,
            "target": target,
        }).sort_values("mu", ascending=False).reset_index(drop=True)

        return out

    def suggest_next_within_indices(
            self,
            candidate_indices_1based: np.ndarray,
            policy: str = "ucb",  # "mu", "std", "ucb"
            kappa: float = 1.96,
    ) -> int:
        """
        Choose one index_1based from a provided subset of candidates (must be unobserved).
        """
        if self._gp is None:
            raise RuntimeError("Model not fitted. Call initialize() first.")

        idx0 = np.asarray(candidate_indices_1based, dtype=int) - 1
        if idx0.ndim != 1:
            idx0 = idx0.ravel()

        # Remove any that are already observed (safety)
        chosen = set(self._chosen_idx)
        idx0 = np.array([i for i in idx0 if i not in chosen], dtype=int)

        # idx0 = np.array([i for i in idx0 if i not in set(self._chosen_idx)], dtype=int)
        if idx0.size == 0:
            raise RuntimeError("No unobserved candidates in the provided subset.")

        Xq = self.Xc[idx0, :]
        mu, std = self._gp.predict(Xq, return_std=True)

        if policy == "mu":
            best_local = int(idx0[np.argmax(mu)])
        elif policy == "std":
            best_local = int(idx0[np.argmax(std)])
        elif policy == "ucb":
            score = mu + float(kappa) * std
            best_local = int(idx0[np.argmax(score)])
        else:
            raise ValueError("policy must be one of: 'mu', 'std', 'ucb'")

        return best_local + 1

    def compute_Xp_candidates(self, eps, use_ucb=True, kappa=1.96,available_1based=None):
        if self._gp is None:
            raise RuntimeError("Model not fitted. Call initialize() first.")
        mu, std = self.predict_all()
        y_best = np.max(self._y_obs)
        target = y_best - eps

        mask = self._unobserved_mask()
        if available_1based is not None:
            avail_mask = np.array([(i + 1) in available_1based for i in range(self.N)], dtype=bool)
            mask = mask & avail_mask

        score = (mu + float(kappa) * std) if use_ucb else mu
        idx1 = np.where(mask & (score >= target))[0] + 1
        return idx1, mu, std


    def run(
        self,
        n_iters: int,
        evaluate_fn,               # callable: (index_1based:int, x:np.ndarray) -> float (objective)
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Sequential BO loop for n_iters steps.
        At each step, pick argmax EI, call evaluate_fn to get y, update model.
        Returns a history DataFrame.
        """
        history = []
        for it in range(1, n_iters + 1):
            idxs, acqs = self.suggest_next(top_k=1)
            idx = int(idxs[0])
            x = self.Xc[idx - 1, :]
            y = float(evaluate_fn(idx, x))
            self.tell(idx, y)
            acq_name = "EI" if self.ei_acq else "UCB"
            history.append({"iter": it, "index_1based": idx, "y": y, acq_name: float(acqs[0])})
            if verbose:
                print(f"[BO] iter {it:02d}: picked {idx}, ACQ={acqs[0]:.6g}, y={y:.6g}")
        return pd.DataFrame(history)

    def predict_all(self):
        """Posterior mean & std for all N candidates (order = Input columns)."""
        if self._gp is None:
            raise RuntimeError("GP not fitted yet. Call initialize()/tell() first.")
        mu, std = self._gp.predict(self.Xc, return_std=True)
        return mu, std  # shape (N,), (N,)

    def ei_all(self):
        """Expected Improvement for all N candidates (maximization)."""
        if self._gp is None:
            raise RuntimeError("GP not fitted yet. Call initialize()/tell() first.")
        mu, std = self._gp.predict(self.Xc, return_std=True)
        y_best = float(np.max(self._y_obs))
        ei = self._ei(mu, std, y_best)
        return ei  # shape (N,)

    def ucb_all(self):
        """Upper Confidence Bound for all N candidates (maximization)."""
        if self._gp is None:
            raise RuntimeError("GP not fitted yet. Call initialize()/tell() first.")
        mu, std = self._gp.predict(self.Xc, return_std=True)
        std = np.maximum(std, 1e-12)  # numerical safety
        ucb = mu + self.kappa * std
        return ucb  # shape (N,)
