# MS-GARCH(1,1) with Student-t innovations, fitted via MLE with JIT-compiled Hamilton filter

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from numba import njit
import math
import warnings
warnings.filterwarnings('ignore')

try:
    from src.utils import config
    TABLES_DIR = config.TABLES_DIR
except ImportError:
    TABLES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "output" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Numerical safety bounds
NU_LOWER_BOUND = 2.1
NU_WARM_START_UPPER_BOUND = 50.0
NU_OPTIMIZER_UPPER_BOUND = 50.0
INITIAL_VARIANCE_UPPER_BOUND = 1e5

ALPHA_LOWER = 0.01
ALPHA_UPPER = 0.50
BETA_LOWER = 0.30
BETA_UPPER = 0.995
P_LOWER = 0.50
P_UPPER = 0.995
PERSISTENCE_LOWER = 0.60
PERSISTENCE_UPPER = 0.995

MU_LOWER = -5.0
MU_UPPER =  5.0

OMEGA_FLOOR = 1e-12
OMEGA_CEIL  = 50.0    # In scaled space

WARM_START_PERTURBATION_SCALE = 0.08
FRESH_START_EVERY_N = 6

ADAPTIVE_WINDOW_LOW_VOL = 378
ADAPTIVE_WINDOW_MID_VOL = 252
ADAPTIVE_WINDOW_HIGH_VOL = 126
ADAPTIVE_VOL_LOOKBACK = 63

MSGARCH_CONFIDENCE_THRESHOLD = 0.72
MSGARCH_MAX_BLEND_WEIGHT = 0.85

MIN_ESTIMATION_DATE = pd.Timestamp("2017-01-01")

# Try to import arch for GARCH warm start
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("Warning: arch package not installed. GARCH warm start disabled.")


@njit(cache=True)
def _numba_gammaln(x):
    # Lanczos approximation
    g = 7
    c = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ])
    
    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _numba_gammaln(1 - x)
    
    x = x - 1
    a = c[0]
    for i in range(1, g + 2):
        a += c[i] / (x + i)
    
    t = x + g + 0.5
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


@njit(cache=True)
def _t_log_likelihood(x, nu, sigma2):
    # Log-likelihood of one observation under t(nu, 0, sigma2)
    if sigma2 <= 0 or nu <= 2.1:
        return -1e10
    const = _numba_gammaln((nu + 1) / 2) - _numba_gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi * sigma2)
    kernel = -((nu + 1) / 2) * np.log(1 + x**2 / ((nu - 2) * sigma2))
    return const + kernel


@njit(cache=True)
def hamilton_filter_jit(returns, omega_0, alpha_0, beta_0, omega_1, alpha_1, beta_1,
                        mu_0, mu_1, p00, p11, nu_0, nu_1):
    # Hamilton filter for MS-GARCH using Gray/Klaassen variance collapse to avoid 2^T path explosion
    T = len(returns)
    filtered_prob = np.zeros((T, 2))
    predicted_prob = np.zeros((T, 2))
    sigma2 = np.zeros((T, 2))
    log_likelihood = 0.0
    
    P = np.array([[p00, 1 - p00], 
                  [1 - p11, p11]])
    
    denom = (2 - p00 - p11)
    if abs(denom) < 1e-10:
        pi_stat = np.array([0.5, 0.5])
    else:
        pi_stat = np.array([(1 - p11) / denom, (1 - p00) / denom])
    
    sigma2_0_uncond = omega_0 / max(1 - alpha_0 - beta_0, 0.01)
    sigma2_1_uncond = omega_1 / max(1 - alpha_1 - beta_1, 0.01)
    sigma2_0_uncond = min(max(sigma2_0_uncond, 1e-8), INITIAL_VARIANCE_UPPER_BOUND)
    sigma2_1_uncond = min(max(sigma2_1_uncond, 1e-8), INITIAL_VARIANCE_UPPER_BOUND)
    
    # Collapsed variances initialised to unconditional variance
    prev_h_0 = sigma2_0_uncond
    prev_h_1 = sigma2_1_uncond
    prev_filtered = pi_stat.copy()
    
    for t in range(T):
        r = returns[t]
        
        if t == 0:
            curr_sigma2_0 = sigma2_0_uncond
            curr_sigma2_1 = sigma2_1_uncond
        else:
            eps_0_prev = returns[t-1] - mu_0
            eps_1_prev = returns[t-1] - mu_1
            curr_sigma2_0 = omega_0 + alpha_0 * eps_0_prev**2 + beta_0 * prev_h_0
            curr_sigma2_1 = omega_1 + alpha_1 * eps_1_prev**2 + beta_1 * prev_h_1
        
        curr_sigma2_0 = max(curr_sigma2_0, 1e-12)
        curr_sigma2_1 = max(curr_sigma2_1, 1e-12)
        sigma2[t, 0] = curr_sigma2_0
        sigma2[t, 1] = curr_sigma2_1
        
        eps_0 = r - mu_0
        eps_1 = r - mu_1
        ll_0 = _t_log_likelihood(eps_0, nu_0, curr_sigma2_0)
        ll_1 = _t_log_likelihood(eps_1, nu_1, curr_sigma2_1)
        
        max_ll = max(ll_0, ll_1)
        if max_ll < -500:
            lik_0 = 1e-200
            lik_1 = 1e-200
        else:
            lik_0 = np.exp(ll_0 - max_ll)
            lik_1 = np.exp(ll_1 - max_ll)
        
        pred_prob_t = P.T @ prev_filtered
        predicted_prob[t, :] = pred_prob_t
        
        joint_0 = lik_0 * pred_prob_t[0]
        joint_1 = lik_1 * pred_prob_t[1]
        marginal = joint_0 + joint_1
        if marginal < 1e-300:
            marginal = 1e-300
        
        filtered_prob[t, 0] = joint_0 / marginal
        filtered_prob[t, 1] = joint_1 / marginal
        log_likelihood += np.log(marginal) + max_ll
        
        filt_0 = filtered_prob[t, 0]
        filt_1 = filtered_prob[t, 1]
        
        pred_next_0 = P[0, 0] * filt_0 + P[1, 0] * filt_1
        pred_next_1 = P[0, 1] * filt_0 + P[1, 1] * filt_1
        
        # Collapse variance into each next-period regime
        if pred_next_0 > 1e-100:
            w_00 = P[0, 0] * filt_0 / pred_next_0
            w_10 = P[1, 0] * filt_1 / pred_next_0
        else:
            w_00 = 0.5
            w_10 = 0.5
        
        if pred_next_1 > 1e-100:
            w_01 = P[0, 1] * filt_0 / pred_next_1
            w_11 = P[1, 1] * filt_1 / pred_next_1
        else:
            w_01 = 0.5
            w_11 = 0.5
        
        prev_h_0 = (w_00 * (curr_sigma2_0 + (mu_0 - mu_0)**2) +
                     w_10 * (curr_sigma2_1 + (mu_1 - mu_0)**2))
        prev_h_1 = (w_01 * (curr_sigma2_0 + (mu_0 - mu_1)**2) +
                     w_11 * (curr_sigma2_1 + (mu_1 - mu_1)**2))
        prev_h_0 = max(prev_h_0, 1e-12)
        prev_h_1 = max(prev_h_1, 1e-12)
        prev_filtered = filtered_prob[t, :]
    
    return log_likelihood, filtered_prob, sigma2, predicted_prob


@njit(cache=True)
def kim_smoother_jit(filtered_prob, predicted_prob, p00, p11):
    # Kim smoother backward pass, gives P(S_t = i | all data), sharper than filtered probs
    T = filtered_prob.shape[0]
    smoothed_prob = np.zeros((T, 2))
    smoothed_prob[T-1, :] = filtered_prob[T-1, :]
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])
    for t in range(T - 2, -1, -1):
        for i in range(2):
            sum_val = 0.0
            for j in range(2):
                denom = predicted_prob[t+1, j]
                if denom < 1e-100:
                    denom = 1e-100
                sum_val += P[i, j] * smoothed_prob[t+1, j] / denom
            smoothed_prob[t, i] = filtered_prob[t, i] * sum_val
    return smoothed_prob


def get_garch_warm_start(returns):
    # Fit single-regime GARCH(1,1)-t for MS-GARCH starting values, falls back to heuristics if arch unavailable
    if not HAS_ARCH:
        var_ret = np.var(returns)
        return {
            'omega': var_ret * 0.05,
            'alpha': 0.08,
            'beta': 0.85,
            'nu': 8.0,
            'mu': np.mean(returns)
        }

    try:
        scale = np.std(returns) * 100
        returns_scaled = returns / scale * 100
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='t', mean='Constant')
        result = model.fit(disp='off', show_warning=False)

        omega = result.params['omega'] / (100**2) * (scale**2)
        alpha = result.params['alpha[1]']
        beta = result.params['beta[1]']
        mu = result.params['mu'] / 100 * scale
        nu = result.params.get('nu', 8.0)

        omega = max(omega, 1e-10)
        alpha = min(max(alpha, ALPHA_LOWER), ALPHA_UPPER)
        beta = min(max(beta, BETA_LOWER), BETA_UPPER)
        if alpha + beta < PERSISTENCE_LOWER:
            beta = max(PERSISTENCE_LOWER - alpha, BETA_LOWER)
        if alpha + beta >= PERSISTENCE_UPPER:
            beta = max(PERSISTENCE_UPPER - alpha, BETA_LOWER)
        nu = min(max(nu, NU_LOWER_BOUND), NU_WARM_START_UPPER_BOUND)

        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'nu': nu,
            'mu': mu
        }

    except Exception:
        var_ret = np.var(returns)
        return {
            'omega': var_ret * 0.05,
            'alpha': 0.08,
            'beta': 0.85,
            'nu': 8.0,
            'mu': np.mean(returns)
        }


def _build_two_regime_from_garch(garch_params):
    # Construct a two-regime parameter set from single-regime GARCH estimates
    omega_g = garch_params['omega']
    alpha_g = garch_params['alpha']
    beta_g = garch_params['beta']
    mu_g = garch_params['mu']
    nu_g = garch_params['nu']

    out = {
        'omega_0': omega_g * 0.5,
        'alpha_0': max(alpha_g * 0.6, ALPHA_LOWER),
        'beta_0': min(beta_g * 1.02, BETA_UPPER),
        'omega_1': omega_g * 2.0,
        'alpha_1': min(alpha_g * 1.5, ALPHA_UPPER),
        'beta_1': max(beta_g * 0.90, BETA_LOWER),
        'mu_0': mu_g,
        'mu_1': mu_g,
        'p00': 0.95,
        'p11': 0.90,
        'nu_0': min(nu_g * 1.5, NU_OPTIMIZER_UPPER_BOUND),
        'nu_1': max(nu_g * 0.6, NU_LOWER_BOUND),
    }

    for sfx in ('0', '1'):
        a = out[f'alpha_{sfx}']
        b = out[f'beta_{sfx}']
        if a + b >= PERSISTENCE_UPPER:
            out[f'beta_{sfx}'] = max(PERSISTENCE_UPPER - a, BETA_LOWER)
        if a + b < PERSISTENCE_LOWER:
            out[f'beta_{sfx}'] = max(PERSISTENCE_LOWER - a, BETA_LOWER)

    return out


def _compute_msgarch_diagnostics(params):
    # Compute diagnostics and a confidence score for regime quality
    eps = 1e-10
    persistence_0 = params['alpha_0'] + params['beta_0']
    persistence_1 = params['alpha_1'] + params['beta_1']
    uncond_var_0 = params['omega_0'] / max(1 - persistence_0, 1e-3)
    uncond_var_1 = params['omega_1'] / max(1 - persistence_1, 1e-3)
    vol_ratio = np.sqrt(max(uncond_var_1, eps) / max(uncond_var_0, eps))
    vol_ratio = max(vol_ratio, 1.0 / max(vol_ratio, eps))

    boundary_alpha = int(abs(params['alpha_0'] - ALPHA_LOWER) < 5e-4) + int(abs(params['alpha_1'] - ALPHA_LOWER) < 5e-4)
    boundary_nu = int(params['nu_0'] > NU_OPTIMIZER_UPPER_BOUND - 0.1) + int(params['nu_1'] > NU_OPTIMIZER_UPPER_BOUND - 0.1)
    near_transition_boundary = int(params['p00'] < 0.53 or params['p00'] > 0.99) + int(params['p11'] < 0.53 or params['p11'] > 0.99)

    score = 1.0
    if vol_ratio < 1.60:
        score -= min(0.55, 0.70 * (1.60 - vol_ratio))
    if vol_ratio < 1.35:
        score -= 0.12
    if abs(persistence_1 - persistence_0) < 0.05:
        score -= 0.18
    if persistence_1 < 0.75:
        score -= min(0.30, 0.45 * (0.75 - persistence_1))
    if params['p11'] < 0.70:
        score -= min(0.25, 0.60 * (0.70 - params['p11']))
    score -= 0.12 * boundary_alpha
    score -= 0.09 * boundary_nu
    score -= 0.10 * near_transition_boundary
    score = float(np.clip(score, 0.0, 1.0))

    return {
        'msgarch_confidence_score': score,
        'msgarch_vol_ratio': float(vol_ratio),
        'msgarch_persistence_0': float(persistence_0),
        'msgarch_persistence_1': float(persistence_1),
        'msgarch_alpha_boundary_hits': int(boundary_alpha),
        'msgarch_nu_boundary_hits': int(boundary_nu),
        'msgarch_transition_boundary_hits': int(near_transition_boundary),
    }


def _blend_params(primary, fallback, blend_weight):
    # Blend parameter dictionaries while preserving constraints
    w = float(np.clip(blend_weight, 0.0, 1.0))
    p = {}
    for key in ['omega_0', 'alpha_0', 'beta_0', 'omega_1', 'alpha_1', 'beta_1', 'mu_0', 'mu_1', 'p00', 'p11', 'nu_0', 'nu_1']:
        p[key] = (1.0 - w) * primary[key] + w * fallback[key]

    p['alpha_0'] = float(np.clip(p['alpha_0'], ALPHA_LOWER, ALPHA_UPPER))
    p['alpha_1'] = float(np.clip(p['alpha_1'], ALPHA_LOWER, ALPHA_UPPER))
    p['beta_0'] = float(np.clip(p['beta_0'], BETA_LOWER, BETA_UPPER))
    p['beta_1'] = float(np.clip(p['beta_1'], BETA_LOWER, BETA_UPPER))
    p['p00'] = float(np.clip(p['p00'], P_LOWER, P_UPPER))
    p['p11'] = float(np.clip(p['p11'], P_LOWER, P_UPPER))
    p['nu_0'] = float(np.clip(p['nu_0'], NU_LOWER_BOUND, NU_OPTIMIZER_UPPER_BOUND))
    p['nu_1'] = float(np.clip(p['nu_1'], NU_LOWER_BOUND, NU_OPTIMIZER_UPPER_BOUND))

    for sfx in ('0', '1'):
        a = p[f'alpha_{sfx}']
        b = p[f'beta_{sfx}']
        if a + b >= PERSISTENCE_UPPER:
            p[f'beta_{sfx}'] = max(PERSISTENCE_UPPER - a, BETA_LOWER)
        if a + b < PERSISTENCE_LOWER:
            p[f'beta_{sfx}'] = max(PERSISTENCE_LOWER - a, BETA_LOWER)

    return p


class MSGARCHOptimized:
    # Two-regime MS-GARCH(1,1)-t with JIT Hamilton filter, warm-start MLE, and regime-labeling convention (0=low-vol, 1=high-vol)
    
    def __init__(self, n_regimes=2):
        self.n_regimes = n_regimes
        self.params = None
        self.filtered_probs = None
        self.returns = None
        self.conditional_vol = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self._garch_warmstart = None
        self.window_regime_state = "Intermediate Vol"
    def fit(self, returns, verbose=True, init_params=None):
        # Fit the model; init_params optionally warm-starts from a previous window
        self.returns = np.asarray(returns).flatten()
        self.init_params = init_params
        valid_mask = np.isfinite(self.returns)
        if not np.all(valid_mask):
            self.returns = self.returns[valid_mask]
        if len(self.returns) < 50:
            raise ValueError("Insufficient data points for MS-GARCH estimation")
        return self._fit_mle_optimized(verbose)

    def _fit_mle_optimized(self, verbose=True):
        # Multi-start L-BFGS-B with rolling warm start, regime-differentiation penalties, and 3-phase refinement
        returns = self.returns
        T = len(returns)
        var_ret = np.var(returns)
        
        x0 = None
        x0_perturbed = None  # will be set if rolling warm start succeeds

        # Rolling warm start: reuse previous window's params (perturbed to avoid stickiness)
        if self.init_params is not None:
            try:
                p = self.init_params
                x0 = np.array([
                    self._omega_to_unconstrained(p['omega_0']),
                    self._alpha_to_unconstrained(p['alpha_0']),
                    self._beta_to_unconstrained(p['beta_0']),
                    self._omega_to_unconstrained(p['omega_1']),
                    self._alpha_to_unconstrained(p['alpha_1']),
                    self._beta_to_unconstrained(p['beta_1']),
                    self._mu_to_unconstrained(p['mu_0']),
                    self._mu_to_unconstrained(p['mu_1']),
                    self._p_to_unconstrained(p['p00']),
                    self._p_to_unconstrained(p['p11']),
                    self._nu_to_unconstrained(p['nu_0']),
                    self._nu_to_unconstrained(p['nu_1'])
                ])

                rng = np.random.RandomState(hash(str(returns[:5])) % (2**31))
                perturbation = rng.normal(0, WARM_START_PERTURBATION_SCALE, size=len(x0))
                perturbation[6] *= 0.3
                perturbation[7] *= 0.3
                x0_perturbed = x0 + perturbation
            except Exception:
                x0 = None
                x0_perturbed = None

        if x0 is None:
            garch_params = get_garch_warm_start(returns)
            self._garch_warmstart = garch_params
            omega_base = garch_params['omega']
            alpha_base = garch_params['alpha']
            beta_base = garch_params['beta']
            nu_base = garch_params['nu']
            mu_base = garch_params['mu']

            omega_0_init = omega_base * 0.4
            omega_1_init = omega_base * 2.5
            alpha_0_init = max(alpha_base * 0.5, 0.02)
            alpha_1_init = min(alpha_base * 1.8, 0.25)
            beta_0_init  = min(beta_base * 1.05, 0.97)
            beta_1_init  = max(beta_base * 0.85, 0.55)
            mu_0_init = mu_base + 0.0002
            mu_1_init = mu_base - 0.0003
            nu_0_init = max(min(nu_base * 2.0, NU_WARM_START_UPPER_BOUND), 3.0)
            nu_1_init = max(min(nu_base * 0.4, NU_WARM_START_UPPER_BOUND), NU_LOWER_BOUND)

            x0 = np.array([
                self._omega_to_unconstrained(omega_0_init),
                self._alpha_to_unconstrained(alpha_0_init),
                self._beta_to_unconstrained(beta_0_init),
                self._omega_to_unconstrained(omega_1_init),
                self._alpha_to_unconstrained(alpha_1_init),
                self._beta_to_unconstrained(beta_1_init),
                self._mu_to_unconstrained(mu_0_init),
                self._mu_to_unconstrained(mu_1_init),
                self._p_to_unconstrained(0.92),
                self._p_to_unconstrained(0.92),
                self._nu_to_unconstrained(nu_0_init),
                self._nu_to_unconstrained(nu_1_init)
            ])


        returns_array = np.ascontiguousarray(returns)
        window_state = getattr(self, "window_regime_state", "Intermediate Vol")
        if window_state == "High Vol":
            crisis_p11_floor = 0.80
            crisis_tail_weight = 45.0
        elif window_state == "Low Vol":
            crisis_p11_floor = 0.62
            crisis_tail_weight = 16.0
        else:
            crisis_p11_floor = 0.70
            crisis_tail_weight = 28.0
        
        def neg_log_likelihood(params):
            # NLL and soft economic penalties encouraging distinct, persistent regimes
            try:
                omega_0 = self._unconstrained_to_omega(params[0])
                alpha_0 = self._unconstrained_to_alpha(params[1])
                beta_0 = self._unconstrained_to_beta(params[2])
                
                omega_1 = self._unconstrained_to_omega(params[3])
                alpha_1 = self._unconstrained_to_alpha(params[4])
                beta_1  = self._unconstrained_to_beta(params[5])

                beta_0 = min(beta_0, max(PERSISTENCE_UPPER - alpha_0, BETA_LOWER))
                beta_1 = min(beta_1, max(PERSISTENCE_UPPER - alpha_1, BETA_LOWER))
                if alpha_0 + beta_0 < PERSISTENCE_LOWER:
                    beta_0 = max(beta_0, PERSISTENCE_LOWER - alpha_0)
                if alpha_1 + beta_1 < PERSISTENCE_LOWER:
                    beta_1 = max(beta_1, PERSISTENCE_LOWER - alpha_1)

                mu_0 = self._unconstrained_to_mu(params[6])
                mu_1 = self._unconstrained_to_mu(params[7])
                p00  = self._unconstrained_to_p(params[8])
                p11  = self._unconstrained_to_p(params[9])
                nu_0 = self._unconstrained_to_nu(params[10])
                nu_1 = self._unconstrained_to_nu(params[11])

                if alpha_0 + beta_0 >= PERSISTENCE_UPPER or alpha_1 + beta_1 >= PERSISTENCE_UPPER:
                    return 1e10

                ll, f_probs_tmp, sig2_tmp, _ = hamilton_filter_jit(
                    returns_array, omega_0, alpha_0, beta_0,
                    omega_1, alpha_1, beta_1,
                    mu_0, mu_1, p00, p11, nu_0, nu_1
                )
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10

                penalty = 0.0

                # Penalise insufficient unconditional-volatility separation
                uncond_vol_0 = np.sqrt(omega_0 / max(1 - alpha_0 - beta_0, 0.01))
                uncond_vol_1 = np.sqrt(omega_1 / max(1 - alpha_1 - beta_1, 0.01))
                vol_ratio = max(uncond_vol_1 / uncond_vol_0, uncond_vol_0 / uncond_vol_1)
                if vol_ratio < 1.5:
                    penalty += 50 * (1.5 - vol_ratio) ** 2

                uncond_var_0 = omega_0 / max(1 - alpha_0 - beta_0, 0.01)
                uncond_var_1 = omega_1 / max(1 - alpha_1 - beta_1, 0.01)
                var_gap = abs(uncond_var_1 - uncond_var_0)
                var_scale = max((uncond_var_1 + uncond_var_0) * 0.5, 1e-8)
                rel_var_gap = var_gap / var_scale
                if rel_var_gap < 0.40:
                    penalty += 60 * (0.40 - rel_var_gap) ** 2
                if rel_var_gap < 0.65:
                    penalty += 90 * (0.65 - rel_var_gap) ** 2

                # Penalise insufficient tail-behaviour separation
                nu_ratio = max(nu_0 / nu_1, nu_1 / nu_0)
                if nu_ratio < 1.3:
                    penalty += 10 * (1.3 - nu_ratio) ** 2
                if nu_ratio < 1.6:
                    penalty += 16 * (1.6 - nu_ratio) ** 2

                persistence_0 = alpha_0 + beta_0
                persistence_1 = alpha_1 + beta_1

                # Penalise low persistence
                for persist in [persistence_0, persistence_1]:
                    if persist < 0.70:
                        penalty += 300 * (0.70 - persist) ** 2
                    if persist < PERSISTENCE_LOWER:
                        penalty += 800 * (PERSISTENCE_LOWER - persist) ** 2

                if persistence_0 > 0.98:
                    penalty += 200 * (persistence_0 - 0.98) ** 2
                if persistence_0 > 0.99:
                    penalty += 1000 * (persistence_0 - 0.99) ** 2
                if persistence_1 > 0.98:
                    penalty += 150 * (persistence_1 - 0.98) ** 2
                if persistence_1 > 0.99:
                    penalty += 800 * (persistence_1 - 0.99) ** 2

                if abs(persistence_1 - persistence_0) < 0.08:
                    penalty += 20 * (0.08 - abs(persistence_1 - persistence_0)) ** 2

                alpha_ratio = max(alpha_1 / max(alpha_0, 0.001), alpha_0 / max(alpha_1, 0.001))
                if alpha_ratio < 1.5:
                    penalty += 15 * (1.5 - alpha_ratio) ** 2

                for a in (alpha_0, alpha_1):
                    if a < ALPHA_LOWER + 0.005:
                        penalty += 8 * (ALPHA_LOWER + 0.005 - a) ** 2
                for nu in (nu_0, nu_1):
                    if nu > NU_OPTIMIZER_UPPER_BOUND - 0.5:
                        penalty += 0.25 * (nu - (NU_OPTIMIZER_UPPER_BOUND - 0.5)) ** 2

                if p11 < crisis_p11_floor:
                    penalty += 30 * (crisis_p11_floor - p11) ** 2

                # Crisis-weighted fit, penalise underestimation in tail observations
                sigma_mix = np.sqrt(np.maximum(f_probs_tmp[:, 0] * sig2_tmp[:, 0] + f_probs_tmp[:, 1] * sig2_tmp[:, 1], 1e-10))
                mu_mix = f_probs_tmp[:, 0] * mu_0 + f_probs_tmp[:, 1] * mu_1
                abs_resid = np.abs(returns_array - mu_mix)
                q90 = np.quantile(abs_resid, 0.90)
                tail_mask = abs_resid >= q90
                if np.any(tail_mask):
                    tail_excess = np.maximum(abs_resid[tail_mask] - 2.25 * sigma_mix[tail_mask], 0.0)
                    tail_scale = np.maximum(sigma_mix[tail_mask], 1e-6)
                    penalty += crisis_tail_weight * np.mean((tail_excess / tail_scale) ** 2)

                return -ll + penalty

            except Exception:
                return 1e10

        
        # For warm starts use perturbed version
        if self.init_params is not None and x0_perturbed is not None:
            candidates = [('Perturbed', x0_perturbed)]
            try:
                garch_fresh = get_garch_warm_start(returns)
                omega_f = garch_fresh['omega']
                alpha_f = garch_fresh['alpha']
                beta_f = garch_fresh['beta']
                mu_f = garch_fresh['mu']
                nu_f = garch_fresh['nu']
                
                x0_fresh = np.array([
                    self._omega_to_unconstrained(omega_f * 0.3),
                    self._alpha_to_unconstrained(max(alpha_f * 0.5, 0.02)),
                    self._beta_to_unconstrained(min(beta_f * 1.05, 0.97)),
                    self._omega_to_unconstrained(omega_f * 3.0),
                    self._alpha_to_unconstrained(min(alpha_f * 2.0, 0.30)),
                    self._beta_to_unconstrained(max(beta_f * 0.80, 0.50)),
                    self._mu_to_unconstrained(mu_f + 0.0003),
                    self._mu_to_unconstrained(mu_f - 0.0005),
                    self._p_to_unconstrained(0.93),
                    self._p_to_unconstrained(0.90),
                    self._nu_to_unconstrained(max(min(nu_f * 2.0, 40.0), 3.0)),
                    self._nu_to_unconstrained(max(min(nu_f * 0.4, 40.0), NU_LOWER_BOUND))
                ])
                candidates.append(('FreshFromData', x0_fresh))
            except Exception:
                pass
        else:
            candidates = [('Base', x0)]

        if self.init_params is None:
            x0_extreme = np.array([
                self._omega_to_unconstrained(omega_base * 0.2),
                self._alpha_to_unconstrained(0.02),
                self._beta_to_unconstrained(0.96),
                self._omega_to_unconstrained(omega_base * 4.0),
                self._alpha_to_unconstrained(0.25),
                self._beta_to_unconstrained(0.60),
                self._mu_to_unconstrained(mu_base + 0.0008),
                self._mu_to_unconstrained(mu_base - 0.0010),
                self._p_to_unconstrained(0.95),
                self._p_to_unconstrained(0.95),
                self._nu_to_unconstrained(30.0),
                self._nu_to_unconstrained(2.8)
            ])
            candidates.append(('Extreme', x0_extreme))

            x0_moderate = np.array([
                self._omega_to_unconstrained(omega_base * 0.5),
                self._alpha_to_unconstrained(0.04),
                self._beta_to_unconstrained(0.94),
                self._omega_to_unconstrained(omega_base * 2.0),
                self._alpha_to_unconstrained(0.15),
                self._beta_to_unconstrained(0.75),
                self._mu_to_unconstrained(mu_base + 0.0003),
                self._mu_to_unconstrained(mu_base - 0.0005),
                self._p_to_unconstrained(0.88),
                self._p_to_unconstrained(0.88),
                self._nu_to_unconstrained(18.0),
                self._nu_to_unconstrained(5.0)
            ])
            candidates.append(('Moderate', x0_moderate))

            try:
                window_size = min(20, len(returns) // 5)
                rolling_var = np.array([np.var(returns[max(0,i-window_size):i+1]) for i in range(len(returns))])
                vol_median   = np.median(rolling_var)
                calm_mask    = rolling_var <= vol_median
                crisis_mask  = rolling_var > vol_median
                calm_var   = np.var(returns[calm_mask])   if np.sum(calm_mask)   > 10 else var_ret * 0.5
                crisis_var  = np.var(returns[crisis_mask]) if np.sum(crisis_mask)  > 10 else var_ret * 2.0
                calm_mean   = np.mean(returns[calm_mask])  if np.sum(calm_mask)   > 10 else mu_base
                crisis_mean = np.mean(returns[crisis_mask]) if np.sum(crisis_mask) > 10 else mu_base
                x0_crisis = np.array([
                    self._omega_to_unconstrained(calm_var * 0.05),
                    self._alpha_to_unconstrained(0.03),
                    self._beta_to_unconstrained(0.95),
                    self._omega_to_unconstrained(crisis_var * 0.08),
                    self._alpha_to_unconstrained(0.20),
                    self._beta_to_unconstrained(0.70),
                    self._mu_to_unconstrained(calm_mean),
                    self._mu_to_unconstrained(crisis_mean),
                    self._p_to_unconstrained(0.94),
                    self._p_to_unconstrained(0.92),
                    self._nu_to_unconstrained(25.0),
                    self._nu_to_unconstrained(3.5)
                ])
                candidates.append(('DataDriven', x0_crisis))
            except Exception:
                pass


        best_result   = None
        best_nll      = 1e12
        best_init_name = None

        for init_name, x0_candidate in candidates:
            is_warm = (self.init_params is not None)
            try:
                if not np.all(np.isfinite(x0_candidate)):
                    continue
                result = minimize(
                    neg_log_likelihood,
                    x0_candidate,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 1500,
                        'ftol': 1e-8 if is_warm else 1e-9,
                        'gtol': 1e-5 if is_warm else 1e-7,
                        'maxfun': 3000,
                        'maxls': 50,
                        'disp': False
                    }
                )
                if np.isfinite(result.fun) and result.fun < best_nll:
                    best_result = result
                    best_nll = result.fun
                    best_init_name = init_name
            except Exception as e:
                continue

        # If every candidate failed, fall back to single-regime GARCH params
        if best_result is None:
            
            garch_fb = get_garch_warm_start(returns_array)
            self.params = _build_two_regime_from_garch(garch_fb)
            p = self.params
            self.log_likelihood, self.filtered_probs, sigma2, pred_prob = hamilton_filter_jit(
                returns_array,
                p['omega_0'], p['alpha_0'], p['beta_0'],
                p['omega_1'], p['alpha_1'], p['beta_1'],
                p['mu_0'], p['mu_1'], p['p00'], p['p11'],
                p['nu_0'], p['nu_1']
            )
            self.filtered_probs = kim_smoother_jit(self.filtered_probs, pred_prob, p['p00'], p['p11'])

            self.conditional_vol = np.sqrt(
                self.filtered_probs[:, 0] * sigma2[:, 0] +
                self.filtered_probs[:, 1] * sigma2[:, 1]
            )
            n_params = 12
            self.aic = 2 * n_params - 2 * self.log_likelihood
            self.bic = np.log(T) * n_params - 2 * self.log_likelihood
            return self.params

        # Refinement with tighter tolerances
        try:
            result_refined = minimize(
                neg_log_likelihood,
                best_result.x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 4000, 'maxls': 100, 'disp': False}
            )
            if np.isfinite(result_refined.fun) and result_refined.fun < best_nll:
                best_result = result_refined
                best_nll = result_refined.fun
        except Exception as e:
            if verbose:
                print(f"      Phase 2 refinement failed: {e}")

        # Alternative optimizer if still not converged
        if not getattr(best_result, 'success', False) or best_nll > 1e8:
            try:
                result_tnc = minimize(
                    neg_log_likelihood,
                    best_result.x,
                    method='TNC',
                    options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False}
                )
                if np.isfinite(result_tnc.fun) and result_tnc.fun < best_nll:
                    best_result = result_tnc
                    best_nll = result_tnc.fun
            except Exception as e:
                if verbose:
                    print(f"TNC failed: {e}")

        result = best_result
        params_opt = result.x
        omega_0 = self._unconstrained_to_omega(params_opt[0])
        alpha_0 = self._unconstrained_to_alpha(params_opt[1])
        beta_0 = self._unconstrained_to_beta(params_opt[2])
        omega_1 = self._unconstrained_to_omega(params_opt[3])
        alpha_1 = self._unconstrained_to_alpha(params_opt[4])
        beta_1  = self._unconstrained_to_beta(params_opt[5])

        beta_0 = min(beta_0, max(PERSISTENCE_UPPER - alpha_0, BETA_LOWER))
        beta_1 = min(beta_1, max(PERSISTENCE_UPPER - alpha_1, BETA_LOWER))
        if alpha_0 + beta_0 < PERSISTENCE_LOWER:
            beta_0 = max(beta_0, PERSISTENCE_LOWER - alpha_0)
        if alpha_1 + beta_1 < PERSISTENCE_LOWER:
            beta_1 = max(beta_1, PERSISTENCE_LOWER - alpha_1)

        mu_0 = self._unconstrained_to_mu(params_opt[6])
        mu_1 = self._unconstrained_to_mu(params_opt[7])
        p00  = self._unconstrained_to_p(params_opt[8])
        p11  = self._unconstrained_to_p(params_opt[9])
        nu_0 = self._unconstrained_to_nu(params_opt[10])
        nu_1 = self._unconstrained_to_nu(params_opt[11])

        uncond_vol_0  = np.sqrt(omega_0 / max(1 - alpha_0 - beta_0, 0.01))
        uncond_vol_1  = np.sqrt(omega_1 / max(1 - alpha_1 - beta_1, 0.01))
        persistence_0 = alpha_0 + beta_0
        persistence_1 = alpha_1 + beta_1

        # Label-switching correction: composite calm-score (vol share + nu share); if >1 regime 0 is actually high-vol → swap
        calm_score_0 = uncond_vol_0 / (uncond_vol_0 + uncond_vol_1) + nu_1 / (nu_0 + nu_1)
        if calm_score_0 > 1.0:
            omega_0, omega_1 = omega_1, omega_0
            alpha_0, alpha_1 = alpha_1, alpha_0
            beta_0, beta_1 = beta_1, beta_0
            mu_0, mu_1 = mu_1, mu_0
            p00, p11 = p11, p00
            nu_0, nu_1 = nu_1, nu_0
            uncond_vol_0, uncond_vol_1 = uncond_vol_1, uncond_vol_0
            persistence_0, persistence_1 = persistence_1, persistence_0

        self.params = {
            'omega_0': omega_0, 'alpha_0': alpha_0, 'beta_0': beta_0,
            'omega_1': omega_1, 'alpha_1': alpha_1, 'beta_1': beta_1,
            'mu_0': mu_0, 'mu_1': mu_1, 'p00': p00, 'p11': p11,
            'nu_0': nu_0, 'nu_1': nu_1
        }

        self.log_likelihood, self.filtered_probs, sigma2, pred_prob = hamilton_filter_jit(
            returns_array, omega_0, alpha_0, beta_0, omega_1, alpha_1, beta_1,
            mu_0, mu_1, p00, p11, nu_0, nu_1
        )
        self.filtered_probs  = kim_smoother_jit(self.filtered_probs, pred_prob, p00, p11)
        self.conditional_vol = np.sqrt(
            self.filtered_probs[:, 0] * sigma2[:, 0] + 
            self.filtered_probs[:, 1] * sigma2[:, 1]
        )
        n_params = 12
        self.aic = 2 * n_params - 2 * self.log_likelihood
        self.bic = np.log(T) * n_params - 2 * self.log_likelihood

        if verbose:
            vol_ratio_disp = uncond_vol_1 / uncond_vol_0
            sep_warn = " (Warning: poorly separated)" if vol_ratio_disp < 1.3 else ""
            print(f"  MS-GARCH: LL={self.log_likelihood:.1f}  AIC={self.aic:.1f}"
                  f"  vol0={uncond_vol_0*100:.2f}%  vol1={uncond_vol_1*100:.2f}%"
                  f"  ratio={vol_ratio_disp:.2f}x  p00={p00:.3f}  p11={p11:.3f}{sep_warn}")


        return self.params

    # Bounded-sigmoid parameter transforms

    def _alpha_to_unconstrained(self, alpha):
        alpha = min(max(alpha, ALPHA_LOWER + 1e-6), ALPHA_UPPER - 1e-6)
        t = (alpha - ALPHA_LOWER) / (ALPHA_UPPER - ALPHA_LOWER)
        return np.log(t / (1 - t))

    def _unconstrained_to_alpha(self, x):
        return ALPHA_LOWER + (ALPHA_UPPER - ALPHA_LOWER) * expit(x)

    def _beta_to_unconstrained(self, beta):
        beta = min(max(beta, BETA_LOWER + 1e-6), BETA_UPPER - 1e-6)
        t = (beta - BETA_LOWER) / (BETA_UPPER - BETA_LOWER)
        return np.log(t / (1 - t))

    def _unconstrained_to_beta(self, x):
        return BETA_LOWER + (BETA_UPPER - BETA_LOWER) * expit(x)

    def _p_to_unconstrained(self, p):
        p = min(max(p, P_LOWER + 1e-6), P_UPPER - 1e-6)
        t = (p - P_LOWER) / (P_UPPER - P_LOWER)
        return np.log(t / (1 - t))

    def _unconstrained_to_p(self, x):
        return P_LOWER + (P_UPPER - P_LOWER) * expit(x)

    def _nu_to_unconstrained(self, nu):
        # log(nu-2) maps nu to unconstrained space
        nu = min(max(nu, NU_LOWER_BOUND), NU_OPTIMIZER_UPPER_BOUND)
        return np.log(nu - 2)

    def _unconstrained_to_nu(self, x):
        return min(2 + np.exp(x), NU_OPTIMIZER_UPPER_BOUND)

    def _mu_to_unconstrained(self, mu):
        mu = min(max(mu, MU_LOWER + 1e-6), MU_UPPER - 1e-6)
        t  = (mu - MU_LOWER) / (MU_UPPER - MU_LOWER)
        return np.log(t / (1 - t))

    def _unconstrained_to_mu(self, x):
        return MU_LOWER + (MU_UPPER - MU_LOWER) * expit(x)

    def _omega_to_unconstrained(self, omega):
        omega = min(max(omega, OMEGA_FLOOR), OMEGA_CEIL)
        return np.log(omega)

    def _unconstrained_to_omega(self, x):
        return min(max(np.exp(x), OMEGA_FLOOR), OMEGA_CEIL)

    def get_volatility_series(self):
        return self.conditional_vol

    def get_regime_probabilities(self):
        return self.filtered_probs



def run_ms_garch_estimation_optimized(data_df,
                                      gvkey_selected=None,
                                      return_column='asset_return_daily',
                                      gvkey_column='gvkey',
                                      output_file=None,
                                      verbose=True):
    # Rolling MS-GARCH estimation for multiple firms; saves month-end params and forward-fills to daily
    output_path = TABLES_DIR / 'ms_garch_parameters.csv' if output_file is None else Path(output_file)
    if not output_path.is_absolute() and output_file is not None:
        output_path = TABLES_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    firms = data_df[gvkey_column].unique() if gvkey_selected is None else gvkey_selected
    firms = [f for f in firms if not pd.isna(f)]

    if verbose:
        print(f"Processing {len(firms)} firms...\n")

    all_params    = []
    data_with_vol = data_df.copy()
    data_with_vol['ms_garch_volatility'] = np.nan
    data_with_vol['ms_garch_regime_prob'] = np.nan

    for i, gvkey in enumerate(firms):
        firm_data = data_df[data_df[gvkey_column] == gvkey].copy()

        if 'date' not in firm_data.columns:
            if isinstance(firm_data.index, pd.DatetimeIndex):
                firm_data['date'] = firm_data.index

        firm_data['date'] = pd.to_datetime(firm_data['date'])
        firm_data = firm_data.sort_values('date')

        start_date = firm_data['date'].min()
        end_date = firm_data['date'].max()
        try:
            estimation_start = max(start_date + pd.DateOffset(months=12), MIN_ESTIMATION_DATE)
            if estimation_start >= end_date:
                continue
            month_ends = pd.date_range(start=estimation_start, end=end_date, freq='ME')
        except Exception as e:
            print(f"Error defining date range for {gvkey}: {e}")
            continue

        last_params_firm = None
        window_counter   = 0

        for date_point in month_ends:
            try:
                data_up_to_point = firm_data[firm_data['date'] <= date_point]

                if len(data_up_to_point) < ADAPTIVE_WINDOW_HIGH_VOL:
                    continue

                if "asset_return_daily_scaled" in data_up_to_point.columns:
                    hist_returns_raw = data_up_to_point["asset_return_daily_scaled"].dropna().values
                else:
                    hist_returns_raw = data_up_to_point[return_column].dropna().values
                    if len(hist_returns_raw) > 0 and np.std(hist_returns_raw) < 0.1:
                        hist_returns_raw = hist_returns_raw * 100.0

                window_days = ADAPTIVE_WINDOW_MID_VOL
                window_regime_state = "Intermediate Vol"
                if len(hist_returns_raw) >= max(ADAPTIVE_VOL_LOOKBACK + 40, 90):
                    hist_series = pd.Series(hist_returns_raw)
                    hist_roll_vol = hist_series.rolling(
                        ADAPTIVE_VOL_LOOKBACK,
                        min_periods=max(20, ADAPTIVE_VOL_LOOKBACK // 2)
                    ).std().dropna()
                    if len(hist_roll_vol) >= 40:
                        recent_vol = hist_roll_vol.iloc[-1]
                        q_low = hist_roll_vol.quantile(0.33)
                        q_high = hist_roll_vol.quantile(0.67)
                        if recent_vol >= q_high:
                            window_days = ADAPTIVE_WINDOW_HIGH_VOL
                            window_regime_state = "High Vol"
                        elif recent_vol <= q_low:
                            window_days = ADAPTIVE_WINDOW_LOW_VOL
                            window_regime_state = "Low Vol"

                window_days = int(min(window_days, len(data_up_to_point)))
                window_df   = data_up_to_point.iloc[-window_days:].copy()

                valid_count = (window_df['asset_return_daily_scaled'].notna().sum()
                               if 'asset_return_daily_scaled' in window_df.columns
                               else window_df['asset_return_daily'].notna().sum())
                if valid_count < max(100, int(0.75 * window_days)):
                    continue

                scale_factor = 1.0
                if "asset_return_daily_scaled" in window_df.columns:
                    returns = window_df["asset_return_daily_scaled"].dropna().values
                    scale_factor = 100.0
                else:
                    returns = window_df[return_column].dropna().values
                    if np.std(returns) < 0.1:
                        returns = returns * 100.0
                        scale_factor = 100.0

                if len(returns) < 100:
                    continue

                model = MSGARCHOptimized()
                model.window_regime_state = window_regime_state

                init_p = None
                window_counter += 1
                if last_params_firm is not None and window_counter % FRESH_START_EVERY_N != 0:
                    init_p = last_params_firm.copy()

                params = model.fit(returns, verbose=False, init_params=init_p)

                if params is None:
                    if last_params_firm is not None:
                        params = last_params_firm.copy()
                    else:
                        garch_fb = get_garch_warm_start(returns)
                        params   = _build_two_regime_from_garch(garch_fb)

                import copy
                last_params_firm = copy.deepcopy(params)  # deep copy protects warm-start from later unscaling

                try:
                    vol_series = model.get_volatility_series()
                    regime_probs = model.get_regime_probabilities()
                except Exception:
                    p = params
                    _, f_probs, sig2, f_pred = hamilton_filter_jit(
                        returns,
                        p['omega_0'], p['alpha_0'], p['beta_0'],
                        p['omega_1'], p['alpha_1'], p['beta_1'],
                        p['mu_0'], p['mu_1'], p['p00'], p['p11'],
                        p['nu_0'], p['nu_1']
                    )
                    f_probs      = kim_smoother_jit(f_probs, f_pred, p['p00'], p['p11'])
                    vol_series   = np.sqrt(f_probs[:, 0] * sig2[:, 0] + f_probs[:, 1] * sig2[:, 1])
                    regime_probs = f_probs

                diag = _compute_msgarch_diagnostics(params)
                blend_weight = 0.0
                if diag['msgarch_confidence_score'] < MSGARCH_CONFIDENCE_THRESHOLD:
                    garch_fb = get_garch_warm_start(returns)
                    fallback_params = _build_two_regime_from_garch(garch_fb)

                    shortfall = MSGARCH_CONFIDENCE_THRESHOLD - diag['msgarch_confidence_score']
                    blend_weight = np.clip(
                        MSGARCH_MAX_BLEND_WEIGHT * (shortfall / MSGARCH_CONFIDENCE_THRESHOLD),
                        0.0, MSGARCH_MAX_BLEND_WEIGHT
                    )
                    params = _blend_params(params, fallback_params, blend_weight)

                    ll_blend, f_probs, sig2, f_pred = hamilton_filter_jit(
                        returns,
                        params['omega_0'], params['alpha_0'], params['beta_0'],
                        params['omega_1'], params['alpha_1'], params['beta_1'],
                        params['mu_0'], params['mu_1'], params['p00'], params['p11'],
                        params['nu_0'], params['nu_1']
                    )
                    f_probs = kim_smoother_jit(f_probs, f_pred, params['p00'], params['p11'])
                    vol_series = np.sqrt(f_probs[:, 0] * sig2[:, 0] + f_probs[:, 1] * sig2[:, 1])
                    regime_probs = f_probs
                    model.log_likelihood = ll_blend
                    diag = _compute_msgarch_diagnostics(params)

                params['msgarch_confidence_score'] = diag['msgarch_confidence_score']
                params['msgarch_vol_ratio'] = diag['msgarch_vol_ratio']
                params['msgarch_persistence_0'] = diag['msgarch_persistence_0']
                params['msgarch_persistence_1'] = diag['msgarch_persistence_1']
                params['msgarch_alpha_boundary_hits'] = diag['msgarch_alpha_boundary_hits']
                params['msgarch_nu_boundary_hits'] = diag['msgarch_nu_boundary_hits']
                params['msgarch_transition_boundary_hits'] = diag['msgarch_transition_boundary_hits']
                params['msgarch_low_confidence'] = bool(diag['msgarch_confidence_score'] < MSGARCH_CONFIDENCE_THRESHOLD)
                params['msgarch_blend_weight'] = float(blend_weight)
                params['adaptive_window_days'] = int(window_days)
                params['window_regime_state'] = window_regime_state

                if scale_factor != 1.0:
                    params['omega_0'] /= (scale_factor ** 2)
                    params['omega_1'] /= (scale_factor ** 2)
                    params['mu_0'] /= scale_factor
                    params['mu_1'] /= scale_factor
                    vol_series /= scale_factor

                uncond_vol_0_unscaled = np.sqrt(params['omega_0'] / max(1 - params['alpha_0'] - params['beta_0'], 0.01))
                uncond_vol_1_unscaled = np.sqrt(params['omega_1'] / max(1 - params['alpha_1'] - params['beta_1'], 0.01))
                if not (0.0001 < uncond_vol_0_unscaled < 0.15):
                    print(f"    ! Warning: Regime 0 uncond vol {uncond_vol_0_unscaled:.6f} outside reasonable range")
                if not (0.0001 < uncond_vol_1_unscaled < 0.15):
                    print(f"    ! Warning: Regime 1 uncond vol {uncond_vol_1_unscaled:.6f} outside reasonable range")

                last_trading_date = window_df['date'].max()
                prev_month_end = date_point - pd.DateOffset(months=1)
                
                
                if "asset_return_daily_scaled" in window_df.columns:
                    valid_mask = window_df["asset_return_daily_scaled"].notna()
                else:
                    valid_mask = window_df[return_column].notna()

                valid_indices = window_df.index[valid_mask]
                valid_dates = window_df.loc[valid_mask, 'date']
                new_month_mask = valid_dates > prev_month_end

                if new_month_mask.any():
                    update_indices = valid_indices[new_month_mask]
                    update_vol = vol_series[new_month_mask.values]
                    update_probs = regime_probs[new_month_mask.values]
                    
                    data_with_vol.loc[update_indices, 'ms_garch_volatility'] = update_vol
                    data_with_vol.loc[update_indices, 'ms_garch_regime_prob'] = update_probs[:, 1] # Prob of regime 1

                params['gvkey'] = gvkey
                params['date'] = last_trading_date
                params['log_likelihood'] = model.log_likelihood - len(returns) * np.log(scale_factor) if scale_factor != 1.0 else model.log_likelihood
                params['aic'] = 2 * 12 - 2 * params['log_likelihood']
                params['bic'] = np.log(len(returns)) * 12 - 2 * params['log_likelihood']
                params['n_obs'] = len(returns)
                all_params.append(params)

            except Exception as e:
                continue

    if len(all_params) > 0:
        params_df = pd.DataFrame(all_params)
        cols = ['gvkey', 'date', 'omega_0', 'alpha_0', 'beta_0', 'omega_1', 'alpha_1', 'beta_1',
                'mu_0', 'mu_1', 'p00', 'p11', 'nu_0', 'nu_1',
                'msgarch_confidence_score', 'msgarch_vol_ratio',
                'msgarch_persistence_0', 'msgarch_persistence_1',
                'msgarch_alpha_boundary_hits', 'msgarch_nu_boundary_hits',
                'msgarch_transition_boundary_hits', 'msgarch_low_confidence',
                'msgarch_blend_weight', 'adaptive_window_days', 'window_regime_state',
                'log_likelihood', 'aic', 'bic', 'n_obs']
        params_df = params_df[[c for c in cols if c in params_df.columns]]
        params_df.to_csv(output_path, index=False)

        if verbose:
            print(f"MS-GARCH parameters saved: {len(params_df)} firm-months → {output_path}")

        data_with_vol['date'] = pd.to_datetime(data_with_vol['date'])
        merge_cols = [c for c in params_df.columns if c not in ['gvkey', 'date', 'log_likelihood', 'aic', 'bic', 'n_obs']]
        data_with_vol = data_with_vol.drop(columns=[c for c in merge_cols if c in data_with_vol.columns])
        merge_df = params_df[['gvkey', 'date'] + merge_cols]
        data_with_vol = pd.merge(data_with_vol, merge_df, on=['gvkey', 'date'], how='left')
        data_with_vol = data_with_vol.sort_values(['gvkey', 'date'])
        data_with_vol[merge_cols] = data_with_vol.groupby('gvkey')[merge_cols].ffill()

    return data_with_vol

# Aliases for backward compatibility
MSGARCH = MSGARCHOptimized
run_ms_garch_estimation = run_ms_garch_estimation_optimized
