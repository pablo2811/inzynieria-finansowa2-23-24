import numpy as np
import numpy.typing as npt

from scipy.stats import norm

from typing import Sequence, List, Any


def cc_imp_curves(
    FX_spot: float,
    tenor: npt.NDArray[Any] | Any,
    yf: npt.NDArray[np.float_] | float,
    rate_ccy: npt.NDArray[np.float_] | float,
    swapp: npt.NDArray[np.float_],
):
    rt = yf * rate_ccy
    df_ccy = 1 / (1 + rt)
    FX_fwd = FX_spot + swapp / 1_000
    df_pln = FX_fwd / FX_spot * df_ccy
    rate_pln = (1 / df_pln - 1) / yf

    return [
        tenor,
        yf,
        df_ccy,
        rate_ccy,
        df_pln,
        rate_pln,
    ]


def cc_imp_curves2(
    FX_spot: float,
    tenor: npt.NDArray[Any] | Any,
    days: npt.NDArray[np.float_] | float,
    rate_ccy: npt.NDArray[np.float_] | float,
    ccy_basis: int,
    pln_basis: int,
    swapp: npt.NDArray[np.float_],
):
    yf_ccy = days / ccy_basis
    yf_pln = days / pln_basis

    rt = yf_ccy * rate_ccy
    df_ccy = 1 / (1 + rt)
    FX_fwd = FX_spot + swapp / 10_000
    df_pln = FX_spot / FX_fwd * df_ccy
    rate_pln = (1 / df_pln - 1) / yf_pln

    return [
        tenor,
        yf_ccy,
        df_ccy,
        rate_ccy,
        yf_pln,
        df_pln,
        rate_pln,
    ]


def int_df(t, yf, df):
    # DF(t) = DF(t_i) ^ (1 - tau) * DF(t_i+1) ^ tau
    # tau = (t - t_i) / (t_i+1 - t_i)
    # if less DF(t) = DF(t_1) ^ (t / t_1) = e ^ (-r * t)
    i_max = len(yf) - 1
    if t < yf[0]:
        dft = df[0] ** (t / yf[0])
    elif t > yf[i_max]:
        dft = df[i_max] ** (t / yf[i_max])
    else:
        i = 0
        yfi = yf[i]
        while t > yfi:
            i += 1
            yfi = yf[i]
        tau = (t - yf[i - 1]) / (yf[i] - yf[i - 1])
        dft = df[i - 1] ** (1 - tau) * df[i] ** tau

    return dft


## ----------------------------


def BS_value(
    fx_spot: npt.NDArray[np.float_] | float,
    df_ccy_t: float,
    df_pln_t: float,
    tau: float,
    sigma: float,
    omega: int,
    tenor: float,
    strike: float,
    nominal: float,
):

    fx_fwd = fx_spot * df_ccy_t / df_pln_t
    d1 = (np.log(fx_fwd / strike) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    value = (
        nominal
        * omega
        * (
            df_ccy_t * fx_spot * norm.cdf(omega * d1)
            - df_pln_t * strike * norm.cdf(omega * d2)
        )
    )
    return value


def BS_payout(
    omega: int,
    strike: float,
    nominal: float,
    fx_spot: npt.NDArray[np.float_],
):
    return nominal * np.where(
        omega * (fx_spot - strike) > 0, omega * (fx_spot - strike), 0
    )


def BS_delta(
    fx_spot: npt.NDArray[np.float_] | float,
    df_ccy_t: float,
    df_pln_t: float,
    tau: float,
    sigma: float,
    omega: int,
    tenor: float,
    strike: float,
    nominal: float,
    is_spot: bool,
    with_premium: bool,
):

    fx_fwd = fx_spot * df_ccy_t / df_pln_t
    d1 = (np.log(fx_fwd / strike) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if is_spot:
        df_ccy = df_ccy_t
    else:
        df_ccy = 1
    if with_premium:
        delta = nominal * omega * (strike / fx_fwd) * df_ccy * norm.cdf(omega * d2)
    else:
        delta = nominal * omega * df_ccy * norm.cdf(omega * d1)
    return delta


def BS_strike_from_delta(
    delta: npt.NDArray[np.float_] | float,
    fx_spot: npt.NDArray[np.float_] | float,
    df_ccy_t: float,
    df_pln_t: float,
    tau: float,
    sigma: float,
    omega: int,
    tenor: float,
    is_spot: bool,
    with_premium: bool,
):

    fx_fwd = fx_spot * df_ccy_t / df_pln_t
    if is_spot:
        df_ccy = df_ccy_t
    else:
        df_ccy = 1
    if with_premium:
        from functools import partial

        optimize_fn = partial(
            BS_delta,
            fx_spot=fx_spot,
            df_ccy_t=df_ccy_t,
            df_pln_t=df_pln_t,
            tau=tau,
            sigma=sigma,
            omega=omega,
            tenor=tenor,
            nominal=1,
            is_spot=is_spot,
            with_premium=with_premium,
        )

        epsilon = 0.1
        K_prev = fx_spot - 2 * epsilon
        K_current = fx_spot
        n_iter = 0
        dk = 1e-2
        while abs(K_prev - K_current) > epsilon and n_iter < 10:
            f_current = optimize_fn(strike=K_current) - delta  # type: ignore
            f_current1 = optimize_fn(strike=K_current + dk) - delta  # type: ignore
            f_current2 = optimize_fn(strike=K_current - dk) - delta  # type: ignore
            df_dstrike = (f_current1 - f_current2) / (2 * dk)
            K_prev = K_current
            K_current = K_current - f_current / df_dstrike
            n_iter += 1
        strike = K_current
    else:
        d1 = norm.ppf(delta / (omega * df_ccy))
        strike = fx_fwd / np.exp(d1 * sigma * np.sqrt(tau) - 0.5 * sigma**2 * tau)

    return strike


strategy = [
    {
        "position": "BUY",
        "type": "CALL",  # CALL, PUT
        "nominal": 100_000,
        "strike": 4.5,
    },
    {
        "position": "SELL",
        "strike": 4.5,
        "type": "CALL",  # CALL, PUT
        "nominal": 100_000,
    },
]


def _evaluate_option(
    position: str,
    strike: float,
    nominal: float,
    type: str,
    fx_spot: float,
) -> float:
    if type == 'CALL':
        omega = 1
    else:
        omega = -1

    bs_value = BS_payout(
        omega,
        strike=strike,
        nominal=nominal,
        fx_spot=fx_spot, # type: ignore
    )

    if position == 'BUY':
        return bs_value  # type: ignore
    else:
        return -bs_value  # type: ignore


def evaluate_strategy(
    strategy: List[dict[str, Any]],
    fx_spot: float,
):
    total = 0
    for option in strategy:
        total += _evaluate_option(
            option["position"],
            option["strike"],
            option["nominal"],
            option["type"],
            fx_spot,
        )

    return total
