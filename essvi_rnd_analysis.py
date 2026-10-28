#!/usr/bin/env python3
"""Calibrate an arbitrage-controlled eSSVI surface and extract RNDs.

The script consumes the converged Black--Scholes IVs produced by
``part1_analysis.py``.  For each market and maturity it uses out-of-the-money
quotes (puts below the forward and calls at/above the forward), fits an eSSVI
total-variance slice, and imposes:

* the standard sufficient eSSVI butterfly constraints; and
* non-decreasing total variance over a common log-moneyness grid.

Risk-neutral densities are then obtained analytically from the fitted total
variance using the Breeden--Litzenberger relation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

from part1_analysis import MARKETS, MarketConfig


@dataclass(frozen=True)
class ESSVIParameters:
    theta: float
    rho: float
    psi: float


CALENDAR_K_GRID = np.linspace(-0.30, 0.20, 101)
RND_MONEYNESS_GRID = np.linspace(0.35, 1.80, 1451)


def essvi_total_variance(
    k: np.ndarray | float, theta: float, rho: float, psi: float
) -> np.ndarray:
    """eSSVI total variance in the natural (theta, rho, psi) coordinates."""
    k_array = np.asarray(k, dtype=float)
    radicand = (
        (psi * k_array + theta * rho) ** 2
        + theta**2 * (1.0 - rho**2)
    )
    return 0.5 * (
        theta + rho * psi * k_array + np.sqrt(np.maximum(radicand, 0.0))
    )


def essvi_derivatives(
    k: np.ndarray, theta: float, rho: float, psi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return w, dw/dk, and d2w/dk2 for one eSSVI slice."""
    a = psi * k + theta * rho
    d = np.sqrt(a * a + theta * theta * (1.0 - rho * rho))
    d = np.maximum(d, 1e-14)
    w = 0.5 * (theta + rho * psi * k + d)
    w_prime = 0.5 * (rho * psi + psi * a / d)
    w_second = (
        0.5
        * psi**2
        * theta**2
        * (1.0 - rho**2)
        / d**3
    )
    return w, w_prime, w_second


def butterfly_g(
    k: np.ndarray, theta: float, rho: float, psi: float
) -> np.ndarray:
    """Gatheral's density factor; non-negativity excludes butterfly arbitrage."""
    w, w_prime, w_second = essvi_derivatives(k, theta, rho, psi)
    w = np.maximum(w, 1e-14)
    return (
        (1.0 - k * w_prime / (2.0 * w)) ** 2
        - 0.25 * w_prime**2 * (1.0 / w + 0.25)
        + 0.5 * w_second
    )


def prepare_otm_quotes(data: pd.DataFrame, config: MarketConfig) -> pd.DataFrame:
    """Build one price-efficient OTM IV observation per strike and maturity."""
    usable = data[
        data["Converged"].astype(bool) & data["ImpliedVolatility"].notna()
    ].copy()
    usable["Forward"] = config.spot * np.exp(
        (config.risk_free_rate - config.dividend_yield)
        * usable["TimeToMaturityYears"]
    )
    use_put = (usable["OptionType"] == "Put") & (
        usable["Strike"] < usable["Forward"]
    )
    use_call = (usable["OptionType"] == "Call") & (
        usable["Strike"] >= usable["Forward"]
    )
    usable = usable[use_put | use_call].copy()
    usable["LogMoneyness"] = np.log(usable["Strike"] / usable["Forward"])
    usable["ObservedTotalVariance"] = (
        usable["ImpliedVolatility"] ** 2 * usable["TimeToMaturityYears"]
    )
    return usable.sort_values(["Maturity", "Strike"]).reset_index(drop=True)


def initial_parameters(
    k: np.ndarray, observed_w: np.ndarray
) -> ESSVIParameters:
    nearest = int(np.argmin(np.abs(k)))
    theta = max(float(observed_w[nearest]), 1e-6)
    if len(k) >= 3 and np.ptp(k) > 1e-8:
        slope = float(np.polyfit(k, observed_w, 1)[0])
    else:
        slope = -0.1 * math.sqrt(theta)
    rho = -0.70 if slope <= 0 else 0.40
    proposed_psi = abs(slope / rho) if abs(rho) > 1e-8 else math.sqrt(theta)
    feasible_psi = min(
        3.95 / (1.0 + abs(rho)),
        math.sqrt(3.95 * theta / (1.0 + abs(rho))),
    )
    psi = min(max(proposed_psi, 1e-5), 0.80 * feasible_psi)
    return ESSVIParameters(theta=theta, rho=rho, psi=psi)


def fit_slice(
    group: pd.DataFrame,
    previous: ESSVIParameters | None,
) -> tuple[ESSVIParameters, dict[str, object]]:
    """Fit one maturity, enforcing butterfly and sequential calendar bounds."""
    k = group["LogMoneyness"].to_numpy(dtype=float)
    observed_iv = group["ImpliedVolatility"].to_numpy(dtype=float)
    maturity = float(group["TimeToMaturityYears"].iloc[0])
    observed_w = observed_iv**2 * maturity
    initial = initial_parameters(k, observed_w)
    scale = max(float(np.median(observed_iv)), 0.01)

    def objective(x: np.ndarray) -> float:
        theta, rho, psi = x
        fitted_w = essvi_total_variance(k, theta, rho, psi)
        fitted_iv = np.sqrt(np.maximum(fitted_w, 1e-14) / maturity)
        residual = (fitted_iv - observed_iv) / scale
        regularizer = 2e-5 * (rho - initial.rho) ** 2
        return float(np.mean(residual**2) + regularizer)

    constraints: list[dict[str, object]] = [
        {
            "type": "ineq",
            "fun": lambda x: 4.0 - x[2] * (1.0 + abs(x[1])),
        },
        {
            "type": "ineq",
            "fun": lambda x: (
                4.0 * x[0] - x[2] ** 2 * (1.0 + abs(x[1]))
            ),
        },
    ]
    if previous is not None:
        previous_w = essvi_total_variance(
            CALENDAR_K_GRID, previous.theta, previous.rho, previous.psi
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, prior=previous_w: (
                    essvi_total_variance(
                        CALENDAR_K_GRID, x[0], x[1], x[2]
                    )
                    - prior
                    - 1e-10
                ),
            }
        )

    theta_upper = max(0.50, 5.0 * float(np.max(observed_w)))
    starts = []
    for rho_start in (initial.rho, -0.90, -0.50, -0.15, 0.20):
        theta_start = initial.theta
        if previous is not None:
            theta_start = max(theta_start, previous.theta * 1.01)
        feasible_psi = min(
            3.90 / (1.0 + abs(rho_start)),
            math.sqrt(3.90 * theta_start / (1.0 + abs(rho_start))),
        )
        psi_start = min(initial.psi, 0.75 * feasible_psi)
        starts.append(np.array([theta_start, rho_start, max(psi_start, 1e-6)]))

    candidates = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=((1e-8, theta_upper), (-0.999, 0.999), (1e-8, 4.0)),
            constraints=constraints,
            options={"maxiter": 2500, "ftol": 1e-12, "disp": False},
        )
        butterfly_1 = 4.0 - result.x[2] * (1.0 + abs(result.x[1]))
        butterfly_2 = (
            4.0 * result.x[0]
            - result.x[2] ** 2 * (1.0 + abs(result.x[1]))
        )
        calendar_min = math.inf
        if previous is not None:
            calendar_min = float(
                np.min(
                    essvi_total_variance(
                        CALENDAR_K_GRID, result.x[0], result.x[1], result.x[2]
                    )
                    - essvi_total_variance(
                        CALENDAR_K_GRID,
                        previous.theta,
                        previous.rho,
                        previous.psi,
                    )
                )
            )
        feasible = (
            butterfly_1 >= -1e-7
            and butterfly_2 >= -1e-7
            and calendar_min >= -1e-7
        )
        if feasible:
            candidates.append(result)

    if not candidates:
        raise RuntimeError(
            f"No feasible eSSVI fit for maturity "
            f"{pd.Timestamp(group['Maturity'].iloc[0]).date()}"
        )
    successful = [item for item in candidates if item.success]
    result = min(successful or candidates, key=lambda item: float(item.fun))
    params = ESSVIParameters(*map(float, result.x))
    fitted_w = essvi_total_variance(k, params.theta, params.rho, params.psi)
    fitted_iv = np.sqrt(np.maximum(fitted_w, 0.0) / maturity)
    residual = fitted_iv - observed_iv
    sse = float(np.sum(residual**2))
    centered = observed_iv - np.mean(observed_iv)
    sst = float(np.sum(centered**2))
    dense_k = np.linspace(-1.0, 1.0, 2001)
    diagnostics = {
        "Success": bool(result.success),
        "OptimizerMessage": str(result.message),
        "Objective": float(result.fun),
        "RMSE_IV": float(np.sqrt(np.mean(residual**2))),
        "MAE_IV": float(np.mean(np.abs(residual))),
        "R2_IV": 1.0 - sse / sst if sst > 0 else math.nan,
        "ButterflyConstraint1Slack": (
            4.0 - params.psi * (1.0 + abs(params.rho))
        ),
        "ButterflyConstraint2Slack": (
            4.0 * params.theta
            - params.psi**2 * (1.0 + abs(params.rho))
        ),
        "MinDensityFactorG": float(
            np.min(butterfly_g(dense_k, params.theta, params.rho, params.psi))
        ),
    }
    if previous is None:
        diagnostics["MinCalendarVarianceIncrement"] = math.nan
    else:
        diagnostics["MinCalendarVarianceIncrement"] = float(
            np.min(
                essvi_total_variance(
                    CALENDAR_K_GRID, params.theta, params.rho, params.psi
                )
                - essvi_total_variance(
                    CALENDAR_K_GRID,
                    previous.theta,
                    previous.rho,
                    previous.psi,
                )
            )
        )
    return params, diagnostics


def calibrate_market(
    data: pd.DataFrame, config: MarketConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quotes = prepare_otm_quotes(data, config)
    parameter_rows: list[dict[str, object]] = []
    fitted_groups: list[pd.DataFrame] = []
    previous: ESSVIParameters | None = None

    for maturity, group in quotes.groupby("Maturity", sort=True):
        if len(group) < 5:
            continue
        params, diagnostics = fit_slice(group, previous)
        group = group.copy()
        maturity_years = float(group["TimeToMaturityYears"].iloc[0])
        group["FittedTotalVariance"] = essvi_total_variance(
            group["LogMoneyness"].to_numpy(dtype=float),
            params.theta,
            params.rho,
            params.psi,
        )
        group["FittedIV"] = np.sqrt(
            group["FittedTotalVariance"] / maturity_years
        )
        group["IVResidual"] = group["ImpliedVolatility"] - group["FittedIV"]
        fitted_groups.append(group)
        parameter_rows.append(
            {
                "Market": config.ticker,
                "Maturity": pd.Timestamp(maturity),
                "TimeToMaturityYears": maturity_years,
                "Forward": float(group["Forward"].iloc[0]),
                "Observations": len(group),
                "Theta": params.theta,
                "Rho": params.rho,
                "Psi": params.psi,
                "ATMVolatility": math.sqrt(params.theta / maturity_years),
                **diagnostics,
            }
        )
        previous = params

    if not parameter_rows:
        raise ValueError(f"No eSSVI slices could be fitted for {config.ticker}")
    return pd.DataFrame(parameter_rows), pd.concat(fitted_groups, ignore_index=True)


def extract_rnd(
    parameter_table: pd.DataFrame, config: MarketConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    density_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for row in parameter_table.itertuples(index=False):
        theta, rho, psi = float(row.Theta), float(row.Rho), float(row.Psi)
        forward = float(row.Forward)
        normalized_strike = RND_MONEYNESS_GRID
        strike = forward * normalized_strike
        k = np.log(normalized_strike)
        w, _, _ = essvi_derivatives(k, theta, rho, psi)
        g = butterfly_g(k, theta, rho, psi)
        d_minus = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
        standard_normal_pdf = np.exp(-0.5 * d_minus**2) / math.sqrt(
            2.0 * math.pi
        )
        density = standard_normal_pdf * g / (strike * np.sqrt(w))
        mass = float(np.trapezoid(density, strike))
        normalized_density = density / mass
        mean = float(np.trapezoid(strike * normalized_density, strike))
        variance = float(
            np.trapezoid((strike - mean) ** 2 * normalized_density, strike)
        )
        std = math.sqrt(max(variance, 0.0))
        skewness = float(
            np.trapezoid(
                ((strike - mean) / std) ** 3 * normalized_density, strike
            )
        )
        excess_kurtosis = float(
            np.trapezoid(
                ((strike - mean) / std) ** 4 * normalized_density, strike
            )
            - 3.0
        )
        downside_80 = float(
            np.trapezoid(
                normalized_density[normalized_strike <= 0.80],
                strike[normalized_strike <= 0.80],
            )
        )
        upside_120 = float(
            np.trapezoid(
                normalized_density[normalized_strike >= 1.20],
                strike[normalized_strike >= 1.20],
            )
        )
        summaries.append(
            {
                "Market": config.ticker,
                "Maturity": pd.Timestamp(row.Maturity),
                "TimeToMaturityYears": float(row.TimeToMaturityYears),
                "Forward": forward,
                "MassOnGrid": mass,
                "Mean": mean,
                "MeanOverForward": mean / forward,
                "StandardDeviation": std,
                "CoefficientOfVariation": std / mean,
                "Skewness": skewness,
                "ExcessKurtosis": excess_kurtosis,
                "ProbabilityBelow80PctForward": downside_80,
                "ProbabilityAbove120PctForward": upside_120,
                "MinimumRawDensity": float(np.min(density)),
                "MinimumDensityFactorG": float(np.min(g)),
            }
        )
        density_frames.append(
            pd.DataFrame(
                {
                    "Market": config.ticker,
                    "Maturity": pd.Timestamp(row.Maturity),
                    "TimeToMaturityYears": float(row.TimeToMaturityYears),
                    "Forward": forward,
                    "Strike": strike,
                    "StrikeOverForward": normalized_strike,
                    "LogMoneyness": k,
                    "TotalVariance": w,
                    "ImpliedVolatility": np.sqrt(
                        w / float(row.TimeToMaturityYears)
                    ),
                    "DensityFactorG": g,
                    "RiskNeutralDensity": density,
                    "NormalizedRiskNeutralDensity": normalized_density,
                    "StandardizedDensity": forward * normalized_density,
                }
            )
        )

    return pd.concat(density_frames, ignore_index=True), pd.DataFrame(summaries)


def selected_maturities(values: Iterable[pd.Timestamp], count: int = 4) -> list:
    ordered = list(pd.to_datetime(list(values)))
    indices = np.unique(np.linspace(0, len(ordered) - 1, count).round().astype(int))
    return [ordered[index] for index in indices]


def plot_fits(
    fitted: pd.DataFrame,
    parameters: pd.DataFrame,
    config: MarketConfig,
    path: Path,
) -> None:
    chosen = selected_maturities(parameters["Maturity"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, maturity in zip(axes.flat, chosen):
        points = fitted[fitted["Maturity"] == maturity].sort_values(
            "LogMoneyness"
        )
        row = parameters[parameters["Maturity"] == maturity].iloc[0]
        k_grid = np.linspace(
            points["LogMoneyness"].min() - 0.02,
            points["LogMoneyness"].max() + 0.02,
            400,
        )
        fit_iv = np.sqrt(
            essvi_total_variance(
                k_grid, float(row.Theta), float(row.Rho), float(row.Psi)
            )
            / float(row.TimeToMaturityYears)
        )
        for option_type, marker in (("Put", "v"), ("Call", "^")):
            subset = points[points["OptionType"] == option_type]
            ax.scatter(
                subset["LogMoneyness"],
                100.0 * subset["ImpliedVolatility"],
                s=24,
                marker=marker,
                alpha=0.75,
                label=f"Observed {option_type}",
            )
        ax.plot(k_grid, 100.0 * fit_iv, color="black", lw=2, label="eSSVI")
        ax.set_title(pd.Timestamp(maturity).strftime("%Y-%m-%d"))
        ax.set_xlabel(r"Log-forward moneyness $k=\log(K/F)$")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Implied volatility (%)")
    axes[1, 0].set_ylabel("Implied volatility (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"{config.ticker}: observed OTM IV and constrained eSSVI fits")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rnd(
    density: pd.DataFrame,
    parameters: pd.DataFrame,
    config: MarketConfig,
    path: Path,
) -> None:
    chosen = selected_maturities(parameters["Maturity"])
    fig, ax = plt.subplots(figsize=(11, 7))
    for maturity in chosen:
        subset = density[density["Maturity"] == maturity]
        ax.plot(
            subset["StrikeOverForward"],
            subset["StandardizedDensity"],
            lw=1.8,
            label=pd.Timestamp(maturity).strftime("%Y-%m-%d"),
        )
    ax.axvline(1.0, color="black", ls="--", lw=1, label="Forward")
    ax.set(
        title=f"{config.ticker}: eSSVI-implied risk-neutral densities",
        xlabel="Terminal index level / forward",
        ylabel="Density of normalized terminal level",
        xlim=(0.60, 1.40),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_market_rnd(
    densities: dict[str, pd.DataFrame], path: Path
) -> None:
    common = sorted(
        set(pd.to_datetime(densities["SPX"]["Maturity"]).unique())
        & set(pd.to_datetime(densities["NDX"]["Maturity"]).unique())
    )
    if not common:
        return
    target = min(common, key=lambda value: abs(pd.Timestamp(value) - pd.Timestamp("2024-12-20")))
    fig, ax = plt.subplots(figsize=(11, 7))
    for ticker, density in densities.items():
        subset = density[density["Maturity"] == target]
        ax.plot(
            subset["StrikeOverForward"],
            subset["StandardizedDensity"],
            lw=2,
            label=ticker,
        )
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.set(
        title=(
            "SPX versus NDX risk-neutral density "
            f"({pd.Timestamp(target).strftime('%Y-%m-%d')})"
        ),
        xlabel="Terminal index level / forward",
        ylabel="Density of normalized terminal level",
        xlim=(0.65, 1.35),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    parameter_tables: dict[str, pd.DataFrame],
    rnd_summaries: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    lines = [
        "# eSSVI and Risk-Neutral Density Results",
        "",
        "The calibration uses OTM puts below the forward and OTM calls at or above "
        "the forward. Each maturity is fitted in total-variance space with the "
        "eSSVI natural parameters `(theta, rho, psi)`. Sufficient butterfly "
        "constraints and non-decreasing total variance on a common "
        "log-moneyness grid are imposed during estimation.",
        "",
    ]
    for ticker in ("SPX", "NDX"):
        params = parameter_tables[ticker]
        rnd = rnd_summaries[ticker]
        lines.extend(
            [
                f"## {ticker}",
                "",
                (
                    f"- Fitted {len(params)} maturity slices with "
                    f"{int(params['Observations'].sum())} OTM observations."
                ),
                (
                    f"- Median IV RMSE: "
                    f"{100*params['RMSE_IV'].median():.3f} volatility points; "
                    f"maximum IV RMSE: {100*params['RMSE_IV'].max():.3f} points."
                ),
                (
                    f"- Median fitted rho: {params['Rho'].median():.3f}; "
                    f"range [{params['Rho'].min():.3f}, {params['Rho'].max():.3f}]."
                ),
                (
                    f"- Minimum butterfly density factor across fitted slices: "
                    f"{params['MinDensityFactorG'].min():.6f}; "
                    f"minimum calendar increment: "
                    f"{params['MinCalendarVarianceIncrement'].dropna().min():.8f}."
                ),
                (
                    f"- Median RND skewness: {rnd['Skewness'].median():.3f}; "
                    f"median excess kurtosis: {rnd['ExcessKurtosis'].median():.3f}."
                ),
                (
                    f"- Median model-implied probability below 80% of the forward: "
                    f"{100*rnd['ProbabilityBelow80PctForward'].median():.2f}%."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation caution",
            "",
            "The source strike ranges are narrow for several short maturities. "
            "Central-density comparisons are supported by observed quotes, but "
            "far-tail probabilities depend materially on eSSVI extrapolation and "
            "should be reported as model-implied rather than directly observed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    parameter_tables: dict[str, pd.DataFrame] = {}
    rnd_summaries: dict[str, pd.DataFrame] = {}
    densities: dict[str, pd.DataFrame] = {}

    for config in MARKETS:
        source = input_root / config.ticker / "csv" / "detailed_results.csv"
        data = pd.read_csv(
            source,
            parse_dates=["ValuationDate", "Maturity"],
        )
        parameters, fitted = calibrate_market(data, config)
        density, rnd_summary = extract_rnd(parameters, config)
        market_dir = output_root / config.ticker
        csv_dir = market_dir / "csv"
        chart_dir = market_dir / "charts"
        csv_dir.mkdir(parents=True, exist_ok=True)
        chart_dir.mkdir(parents=True, exist_ok=True)
        parameters.to_csv(
            csv_dir / "essvi_parameters.csv", index=False, date_format="%Y-%m-%d"
        )
        fitted.to_csv(
            csv_dir / "essvi_fitted_points.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        density.to_csv(
            csv_dir / "risk_neutral_density.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        rnd_summary.to_csv(
            csv_dir / "rnd_summary.csv", index=False, date_format="%Y-%m-%d"
        )
        plot_fits(
            fitted,
            parameters,
            config,
            chart_dir / "essvi_selected_smiles.png",
        )
        plot_rnd(
            density,
            parameters,
            config,
            chart_dir / "rnd_selected_maturities.png",
        )
        parameter_tables[config.ticker] = parameters
        rnd_summaries[config.ticker] = rnd_summary
        densities[config.ticker] = density

    plot_cross_market_rnd(
        densities, output_root / "spx_ndx_rnd_comparison.png"
    )
    write_summary(
        parameter_tables, rnd_summaries, output_root / "analysis_summary.md"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs"),
        help="Root containing SPX/csv and NDX/csv detailed IV results.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/essvi_rnd"),
    )
    args = parser.parse_args()
    run(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
