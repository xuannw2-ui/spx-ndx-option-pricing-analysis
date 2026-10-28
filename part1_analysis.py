#!/usr/bin/env python3
"""Final Project Part 1: implied-volatility analysis for SPX and NDX."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.optimize import brentq


VALUATION_DATE = pd.Timestamp("2024-06-15")


@dataclass(frozen=True)
class MarketConfig:
    ticker: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    filename: str


MARKETS = (
    MarketConfig(
        ticker="SPX",
        spot=5431.60,
        risk_free_rate=0.0553,
        dividend_yield=0.0134,
        filename="Final Project--Part1--SPX--Data.xlsx",
    ),
    MarketConfig(
        ticker="NDX",
        spot=19659.80,
        risk_free_rate=0.0553,
        dividend_yield=0.0080,
        filename="Final Project--Part1--NDX--Data.xlsx",
    ),
)

DETAIL_COLUMNS = [
    "Market",
    "ValuationDate",
    "Maturity",
    "DaysToMaturity",
    "TimeToMaturityYears",
    "OptionType",
    "Strike",
    "Ticker",
    "Bid",
    "Ask",
    "MidPrice",
    "Last",
    "Volume",
    "ImpliedVolatility",
    "BSRepricedValue",
    "PriceError",
    "SolverIterations",
    "SolverMethod",
    "Converged",
    "Status",
]


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    option_type: str,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_maturity: float,
    volatility: float,
) -> float:
    """Price a European call or put with a continuous dividend yield."""
    if min(spot, strike, time_to_maturity, volatility) <= 0:
        raise ValueError("spot, strike, time and volatility must be positive")

    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility)
        * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot * math.exp(-dividend_yield * time_to_maturity)
    discounted_strike = strike * math.exp(-rate * time_to_maturity)

    if option_type == "Call":
        return discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    if option_type == "Put":
        return discounted_strike * norm_cdf(-d2) - discounted_spot * norm_cdf(-d1)
    raise ValueError("option_type must be 'Call' or 'Put'")


def black_scholes_vega(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_maturity: float,
    volatility: float,
) -> float:
    """Derivative of the Black-Scholes price with respect to volatility."""
    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility)
        * time_to_maturity
    ) / (volatility * sqrt_t)
    return (
        spot
        * math.exp(-dividend_yield * time_to_maturity)
        * norm_pdf(d1)
        * sqrt_t
    )


def no_arbitrage_bounds(
    option_type: str,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_maturity: float,
) -> tuple[float, float]:
    discounted_spot = spot * math.exp(-dividend_yield * time_to_maturity)
    discounted_strike = strike * math.exp(-rate * time_to_maturity)
    if option_type == "Call":
        return max(0.0, discounted_spot - discounted_strike), discounted_spot
    return max(0.0, discounted_strike - discounted_spot), discounted_strike


def implied_volatility_newton(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_maturity: float,
    initial_volatility: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    min_volatility: float = 1e-6,
    max_volatility: float = 5.0,
    vega_tolerance: float = 1e-12,
) -> Dict[str, Union[float, int, bool, str]]:
    """
    Invert Black-Scholes using Newton's method.

    The update is sigma_(n+1) = sigma_n - (BS(sigma_n)-market_price)/Vega.
    A step is halved when it would leave the allowed positive-volatility
    interval. Brent's bracketed method is used when Vega is too small or all
    Newton starts fail.
    """
    if not all(
        math.isfinite(x)
        for x in (market_price, spot, strike, rate, dividend_yield, time_to_maturity)
    ):
        return _iv_result(status="non_finite_input")
    if market_price <= 0 or spot <= 0 or strike <= 0 or time_to_maturity <= 0:
        return _iv_result(status="invalid_input")
    if (
        not math.isfinite(min_volatility)
        or not math.isfinite(max_volatility)
        or min_volatility <= 0
        or min_volatility >= max_volatility
    ):
        return _iv_result(status="invalid_volatility_bounds")

    lower, upper = no_arbitrage_bounds(
        option_type, spot, strike, rate, dividend_yield, time_to_maturity
    )
    bound_tolerance = max(tolerance, 1e-10 * max(1.0, market_price))
    if market_price < lower - bound_tolerance or market_price > upper + bound_tolerance:
        return _iv_result(status="outside_no_arbitrage_bounds")

    price_tolerance = max(tolerance, 1e-10 * market_price)
    lower_bound_price = black_scholes_price(
        option_type,
        spot,
        strike,
        rate,
        dividend_yield,
        time_to_maturity,
        min_volatility,
    )
    upper_bound_price = black_scholes_price(
        option_type,
        spot,
        strike,
        rate,
        dividend_yield,
        time_to_maturity,
        max_volatility,
    )
    if (
        market_price < lower_bound_price - price_tolerance
        or market_price > upper_bound_price + price_tolerance
    ):
        return _iv_result(status="outside_volatility_bounds")

    starting_guesses = tuple(
        dict.fromkeys((initial_volatility, 0.10, 0.30, 0.50, 1.00))
    )
    best: Optional[Dict[str, Union[float, int, bool, str]]] = None

    for starting_guess in starting_guesses:
        sigma = min(max(starting_guess, min_volatility), max_volatility)
        for iteration in range(1, max_iterations + 1):
            model_price = black_scholes_price(
                option_type,
                spot,
                strike,
                rate,
                dividend_yield,
                time_to_maturity,
                sigma,
            )
            error = model_price - market_price
            candidate = {
                "iv": sigma,
                "repriced": model_price,
                "error": error,
                "iterations": iteration,
                "method": "Newton",
                "converged": abs(error) <= price_tolerance,
                "status": "converged" if abs(error) <= price_tolerance else "iterating",
            }
            if best is None or abs(error) < abs(float(best["error"])):
                best = candidate
            if abs(error) <= price_tolerance:
                return candidate

            vega = black_scholes_vega(
                spot, strike, rate, dividend_yield, time_to_maturity, sigma
            )
            if not math.isfinite(vega) or abs(vega) < vega_tolerance:
                return _implied_volatility_brent(
                    option_type=option_type,
                    market_price=market_price,
                    spot=spot,
                    strike=strike,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    time_to_maturity=time_to_maturity,
                    min_volatility=min_volatility,
                    max_volatility=max_volatility,
                    price_tolerance=price_tolerance,
                    max_iterations=max_iterations,
                )

            step = error / vega
            next_sigma = sigma - step
            halvings = 0
            while (
                not math.isfinite(next_sigma)
                or next_sigma <= min_volatility
                or next_sigma >= max_volatility
            ) and halvings < 60:
                step *= 0.5
                next_sigma = sigma - step
                halvings += 1
            if (
                not math.isfinite(next_sigma)
                or next_sigma <= min_volatility
                or next_sigma >= max_volatility
            ):
                break
            sigma = next_sigma

    brent_result = _implied_volatility_brent(
        option_type=option_type,
        market_price=market_price,
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        time_to_maturity=time_to_maturity,
        min_volatility=min_volatility,
        max_volatility=max_volatility,
        price_tolerance=price_tolerance,
        max_iterations=max_iterations,
    )
    if bool(brent_result["converged"]):
        return brent_result
    if best is None:
        return brent_result
    best["converged"] = False
    best["status"] = "newton_and_brent_not_converged"
    return best


def _implied_volatility_brent(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_maturity: float,
    min_volatility: float,
    max_volatility: float,
    price_tolerance: float,
    max_iterations: int,
) -> Dict[str, Union[float, int, bool, str]]:
    """Solve for IV with Brent's bracketed root-finding method."""

    def objective(volatility: float) -> float:
        return (
            black_scholes_price(
                option_type,
                spot,
                strike,
                rate,
                dividend_yield,
                time_to_maturity,
                volatility,
            )
            - market_price
        )

    lower_error = objective(min_volatility)
    upper_error = objective(max_volatility)
    for volatility, error in (
        (min_volatility, lower_error),
        (max_volatility, upper_error),
    ):
        if abs(error) <= price_tolerance:
            return {
                "iv": volatility,
                "repriced": market_price + error,
                "error": error,
                "iterations": 0,
                "method": "Brent",
                "converged": True,
                "status": "converged",
            }

    if lower_error * upper_error > 0:
        return _iv_result(status="brent_root_not_bracketed", solver_method="Brent")

    try:
        volatility, root_info = brentq(
            objective,
            min_volatility,
            max_volatility,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=max_iterations,
            full_output=True,
            disp=False,
        )
    except (RuntimeError, ValueError, OverflowError):
        return _iv_result(status="brent_failed", solver_method="Brent")

    repriced = black_scholes_price(
        option_type,
        spot,
        strike,
        rate,
        dividend_yield,
        time_to_maturity,
        volatility,
    )
    error = repriced - market_price
    converged = bool(root_info.converged) and abs(error) <= price_tolerance
    return {
        "iv": volatility,
        "repriced": repriced,
        "error": error,
        "iterations": int(root_info.iterations),
        "method": "Brent",
        "converged": converged,
        "status": "converged" if converged else "brent_not_converged",
    }


def _iv_result(
    status: str, solver_method: str = ""
) -> Dict[str, Union[float, int, bool, str]]:
    return {
        "iv": math.nan,
        "repriced": math.nan,
        "error": math.nan,
        "iterations": 0,
        "method": solver_method,
        "converged": False,
        "status": status,
    }


def parse_maturity(sheet_name: str) -> Optional[pd.Timestamp]:
    try:
        return pd.Timestamp(datetime.strptime(sheet_name, "%m-%d-%y"))
    except ValueError:
        return None


def ticker_root(ticker_text: object) -> str:
    if not isinstance(ticker_text, str) or not ticker_text.strip():
        return ""
    return ticker_text.strip().split()[0]


def valid_quote(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def read_market_workbook(path: Path, config: MarketConfig) -> pd.DataFrame:
    """
    Read monthly options only.

    The source files also contain SPXW/NDXP weekly or end-of-month contracts,
    sometimes in the same sheet. The assignment explicitly identifies SPX and
    NDX as the monthly option tickers, so rows whose ticker root differs from
    the market ticker are excluded.
    """
    workbook = load_workbook(path, read_only=True, data_only=True)
    records: list[dict[str, object]] = []

    for worksheet in workbook.worksheets:
        maturity = parse_maturity(worksheet.title)
        if maturity is None:
            continue
        days = int((maturity - VALUATION_DATE).days)
        if days <= 0:
            continue
        time_to_maturity = days / 365.0

        for row in worksheet.iter_rows(min_row=3, values_only=True):
            for option_type, offset in (("Call", 0), ("Put", 6)):
                strike, ticker, bid, ask, last, volume = row[offset : offset + 6]
                if not isinstance(strike, (int, float)):
                    continue
                if ticker_root(ticker) != config.ticker:
                    continue

                record: dict[str, object] = {
                    "Market": config.ticker,
                    "ValuationDate": VALUATION_DATE,
                    "Maturity": maturity,
                    "DaysToMaturity": days,
                    "TimeToMaturityYears": time_to_maturity,
                    "OptionType": option_type,
                    "Strike": float(strike),
                    "Ticker": ticker,
                    "Bid": float(bid) if valid_quote(bid) else math.nan,
                    "Ask": float(ask) if valid_quote(ask) else math.nan,
                    "MidPrice": math.nan,
                    "Last": float(last) if valid_quote(last) else math.nan,
                    "Volume": float(volume)
                    if isinstance(volume, (int, float)) and math.isfinite(float(volume))
                    else math.nan,
                    "ImpliedVolatility": math.nan,
                    "BSRepricedValue": math.nan,
                    "PriceError": math.nan,
                    "SolverIterations": 0,
                    "SolverMethod": "",
                    "Converged": False,
                    "Status": "missing_or_zero_bid_ask",
                }

                if valid_quote(bid) and valid_quote(ask):
                    mid_price = (float(bid) + float(ask)) / 2.0
                    result = implied_volatility_newton(
                        option_type=option_type,
                        market_price=mid_price,
                        spot=config.spot,
                        strike=float(strike),
                        rate=config.risk_free_rate,
                        dividend_yield=config.dividend_yield,
                        time_to_maturity=time_to_maturity,
                    )
                    record.update(
                        {
                            "MidPrice": mid_price,
                            "ImpliedVolatility": result["iv"],
                            "BSRepricedValue": result["repriced"],
                            "PriceError": result["error"],
                            "SolverIterations": result["iterations"],
                            "SolverMethod": result["method"],
                            "Converged": result["converged"],
                            "Status": result["status"],
                        }
                    )
                records.append(record)

    data = pd.DataFrame.from_records(records, columns=DETAIL_COLUMNS)
    if data.empty:
        raise ValueError(f"No monthly {config.ticker} option rows found in {path}")

    duplicate_keys = ["Maturity", "OptionType", "Strike"]
    duplicates = data.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        examples = data.loc[duplicates, duplicate_keys + ["Ticker"]].head(10)
        raise ValueError(f"Duplicate monthly option rows found:\n{examples}")

    return data.sort_values(duplicate_keys).reset_index(drop=True)


def active_quote_universe(data: pd.DataFrame) -> tuple[list[float], list[pd.Timestamp]]:
    """
    Return the common strike/maturity axes used by both Call and Put matrices.

    A maturity whose monthly options have no usable Bid/Ask quotes at all is
    retained in the detailed audit table but excluded from the IV matrix. This
    matches the assignment's SPX example of 80 strikes across 13 maturities.
    Within an active maturity, every listed monthly strike enters the union,
    even if one individual option has a missing quote.
    """
    active_maturities = sorted(
        pd.Timestamp(value)
        for value in data.loc[data["MidPrice"].notna(), "Maturity"].unique()
    )
    active_rows = data[data["Maturity"].isin(active_maturities)]
    strikes = sorted(float(value) for value in active_rows["Strike"].unique())
    return strikes, active_maturities


def build_iv_matrix(
    data: pd.DataFrame,
    option_type: str,
    strikes: Iterable[float],
    maturities: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    usable = data[
        (data["OptionType"] == option_type)
        & data["Converged"]
        & data["ImpliedVolatility"].notna()
    ]
    matrix = usable.pivot(
        index="Strike", columns="Maturity", values="ImpliedVolatility"
    )
    return matrix.reindex(index=list(strikes), columns=list(maturities))


def plot_smiles(
    data: pd.DataFrame, config: MarketConfig, option_type: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    usable = data[
        (data["OptionType"] == option_type)
        & data["Converged"]
        & data["ImpliedVolatility"].notna()
    ]
    for maturity, group in usable.groupby("Maturity", sort=True):
        group = group.sort_values("Strike")
        if len(group) < 2:
            continue
        ax.plot(
            group["Strike"],
            100.0 * group["ImpliedVolatility"],
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            label=pd.Timestamp(maturity).strftime("%Y-%m-%d"),
        )

    ax.axvline(config.spot, color="black", linestyle="--", linewidth=1, label="Spot")
    ax.set(
        title=f"{config.ticker} {option_type} Implied-Volatility Smiles / Skews",
        xlabel="Strike",
        ylabel="Implied volatility (%)",
    )
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_term_structures(
    matrix: pd.DataFrame,
    config: MarketConfig,
    option_type: str,
    output_path: Path,
    strikes: Optional[Iterable[float]] = None,
    with_legend: bool = False,
) -> None:
    selected = matrix if strikes is None else matrix.loc[matrix.index.intersection(strikes)]
    fig, ax = plt.subplots(figsize=(13, 8))
    norm = colors.Normalize(
        vmin=float(selected.index.min()), vmax=float(selected.index.max())
    )
    cmap = plt.get_cmap("viridis")

    for strike, row in selected.iterrows():
        valid = row.dropna()
        if len(valid) < 2:
            continue
        ax.plot(
            valid.index,
            100.0 * valid.values,
            color=cmap(norm(float(strike))),
            marker="o",
            markersize=2.3,
            linewidth=1.0,
            alpha=0.8,
            label=f"{strike:g}",
        )

    ax.set(
        title=f"{config.ticker} {option_type} IV Term Structures by Fixed Strike",
        xlabel="Maturity",
        ylabel="Implied volatility (%)",
    )
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.grid(alpha=0.25)
    if with_legend:
        ax.legend(
            title="Strike",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
        )
    else:
        colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        colorbar.set_label("Strike")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_surface(
    data: pd.DataFrame, config: MarketConfig, option_type: str, output_path: Path
) -> None:
    usable = data[
        (data["OptionType"] == option_type)
        & data["Converged"]
        & data["ImpliedVolatility"].notna()
    ].copy()
    if len(usable) < 3:
        return

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    x = usable["Strike"].to_numpy(dtype=float)
    y = usable["TimeToMaturityYears"].to_numpy(dtype=float)
    z = 100.0 * usable["ImpliedVolatility"].to_numpy(dtype=float)
    try:
        surface = ax.plot_trisurf(
            x,
            y,
            z,
            cmap="viridis",
            linewidth=0.15,
            antialiased=True,
            alpha=0.90,
        )
        fig.colorbar(surface, ax=ax, shrink=0.60, pad=0.10, label="IV (%)")
    except (RuntimeError, ValueError):
        scatter = ax.scatter(x, y, z, c=z, cmap="viridis", s=14)
        fig.colorbar(scatter, ax=ax, shrink=0.60, pad=0.10, label="IV (%)")

    ax.scatter(x, y, z, c="black", s=4, alpha=0.30)
    ax.set(
        title=f"{config.ticker} {option_type} Implied-Volatility Surface",
        xlabel="Strike",
        ylabel="Time to maturity (years)",
        zlabel="Implied volatility (%)",
    )
    ax.view_init(elev=27, azim=-130)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def nearby_atm_strikes(matrix: pd.DataFrame, spot: float, count: int = 21) -> list[float]:
    strikes = np.asarray(matrix.index, dtype=float)
    order = np.argsort(np.abs(strikes - spot))
    return sorted(strikes[order[: min(count, len(strikes))]].tolist())


def write_csv_outputs(
    data: pd.DataFrame,
    call_matrix: pd.DataFrame,
    put_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_dir / "detailed_results.csv", index=False, date_format="%Y-%m-%d")
    call_matrix.to_csv(csv_dir / "call_iv_matrix.csv", date_format="%Y-%m-%d")
    put_matrix.to_csv(csv_dir / "put_iv_matrix.csv", date_format="%Y-%m-%d")


def summary_rows(data: pd.DataFrame, config: MarketConfig) -> list[tuple[str, object]]:
    maturity_count = int(data["Maturity"].nunique())
    valid_maturities = int(data.loc[data["Converged"], "Maturity"].nunique())
    active_strikes, active_maturity_axis = active_quote_universe(data)
    converged = int(data["Converged"].sum())
    return [
        ("Market", config.ticker),
        ("Valuation date", VALUATION_DATE.date()),
        ("Spot", config.spot),
        ("Risk-free rate", config.risk_free_rate),
        ("Dividend yield", config.dividend_yield),
        ("Day-count convention", "Actual days / 365"),
        ("Monthly option ticker filter", config.ticker),
        ("Maturity sheets containing monthly rows", maturity_count),
        ("Maturities with at least one usable quote", len(active_maturity_axis)),
        ("Maturities with at least one converged IV", valid_maturities),
        ("Strike union across quoted maturities", len(active_strikes)),
        ("Monthly call/put rows read", len(data)),
        ("Converged IV calculations", converged),
        ("Converged with Newton", int((data["SolverMethod"] == "Newton").sum())),
        ("Converged with Brent", int((data["SolverMethod"] == "Brent").sum())),
        ("Rows skipped or failed", len(data) - converged),
        ("Newton initial volatility", 0.20),
        ("Newton price tolerance", "max(1e-8, 1e-10 × market price)"),
        ("Newton maximum iterations per start", 100),
    ]


def write_results_workbook(
    data: pd.DataFrame,
    call_matrix: pd.DataFrame,
    put_matrix: pd.DataFrame,
    config: MarketConfig,
    output_path: Path,
) -> None:
    status_counts = (
        data.groupby(["OptionType", "Status"], dropna=False)
        .size()
        .rename("Count")
        .reset_index()
    )
    summary = pd.DataFrame(summary_rows(data, config), columns=["Item", "Value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
        status_counts.to_excel(
            writer, sheet_name="Summary", index=False, startrow=len(summary) + 3
        )
        data.to_excel(writer, sheet_name="Detailed Results", index=False)
        call_matrix.to_excel(writer, sheet_name="Call IV Matrix")
        put_matrix.to_excel(writer, sheet_name="Put IV Matrix")

    workbook = load_workbook(output_path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        for column_cells in worksheet.columns:
            width = min(
                38,
                max(
                    10,
                    max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells
                    )
                    + 2,
                ),
            )
            worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width

    details = workbook["Detailed Results"]
    headers = {cell.value: cell.column for cell in details[1]}
    for row in range(2, details.max_row + 1):
        for name in ("ValuationDate", "Maturity"):
            details.cell(row, headers[name]).number_format = "yyyy-mm-dd"
        details.cell(row, headers["TimeToMaturityYears"]).number_format = "0.000000"
        details.cell(row, headers["ImpliedVolatility"]).number_format = "0.0000%"
        details.cell(row, headers["PriceError"]).number_format = "0.0000000000"

    for sheet_name in ("Call IV Matrix", "Put IV Matrix"):
        worksheet = workbook[sheet_name]
        for cell in worksheet[1][1:]:
            cell.number_format = "yyyy-mm-dd"
        for row in worksheet.iter_rows(min_row=2, min_col=2):
            for cell in row:
                cell.number_format = "0.0000%"

    summary_sheet = workbook["Summary"]
    for row in range(2, summary_sheet.max_row + 1):
        label = summary_sheet.cell(row, 1).value
        if label in {"Risk-free rate", "Dividend yield"}:
            summary_sheet.cell(row, 2).number_format = "0.00%"

    workbook.save(output_path)


def market_chart_metrics(
    data: pd.DataFrame, config: MarketConfig
) -> dict[str, object]:
    """Summarize features visible in the smile, term-structure, and surface charts."""
    valid = data[data["Converged"]].copy()
    if valid.empty:
        return {
            "row_count": len(data),
            "converged_count": 0,
            "maturity_count": 0,
            "strike_count": 0,
            "iv_min": math.nan,
            "iv_max": math.nan,
            "options": {},
        }

    active_strikes, _ = active_quote_universe(data)
    option_metrics: dict[str, dict[str, object]] = {}
    for option_type in ("Call", "Put"):
        subset = valid[valid["OptionType"] == option_type]
        if subset.empty:
            continue

        nearest = (
            subset.assign(Distance=(subset["Strike"] - config.spot).abs())
            .sort_values(["Maturity", "Distance"])
            .groupby("Maturity", as_index=False)
            .first()
            .sort_values("Maturity")
        )

        negative_slopes = 0
        fitted_maturities = 0
        for _, group in subset.groupby("Maturity"):
            if len(group) < 3 or group["Strike"].nunique() < 3:
                continue
            slope = np.polyfit(
                group["Strike"].to_numpy(dtype=float) / config.spot,
                group["ImpliedVolatility"].to_numpy(dtype=float),
                1,
            )[0]
            fitted_maturities += 1
            negative_slopes += int(slope < 0)

        option_metrics[option_type] = {
            "median_atm_iv": float(nearest["ImpliedVolatility"].median()),
            "first_atm_iv": float(nearest.iloc[0]["ImpliedVolatility"]),
            "first_maturity": pd.Timestamp(nearest.iloc[0]["Maturity"]),
            "last_atm_iv": float(nearest.iloc[-1]["ImpliedVolatility"]),
            "last_maturity": pd.Timestamp(nearest.iloc[-1]["Maturity"]),
            "negative_slopes": negative_slopes,
            "fitted_maturities": fitted_maturities,
        }

    return {
        "row_count": len(data),
        "converged_count": len(valid),
        "maturity_count": int(valid["Maturity"].nunique()),
        "strike_count": len(active_strikes),
        "iv_min": float(valid["ImpliedVolatility"].min()),
        "iv_max": float(valid["ImpliedVolatility"].max()),
        "options": option_metrics,
    }


def option_feature_line(
    ticker: str, option_type: str, metrics: dict[str, object]
) -> str:
    first_date = pd.Timestamp(metrics["first_maturity"]).strftime("%Y-%m-%d")
    last_date = pd.Timestamp(metrics["last_maturity"]).strftime("%Y-%m-%d")
    return (
        f"- **{ticker} {option_type}:** "
        f"{metrics['negative_slopes']} of {metrics['fitted_maturities']} "
        "sufficiently populated maturity smiles have a negative fitted strike slope. "
        f"The nearest-to-spot IV moves from {100*float(metrics['first_atm_iv']):.2f}% "
        f"on {first_date} to {100*float(metrics['last_atm_iv']):.2f}% on {last_date}; "
        f"its maturity median is {100*float(metrics['median_atm_iv']):.2f}%."
    )


def write_analysis_summary(
    all_data: dict[str, pd.DataFrame],
    configs: Iterable[MarketConfig],
    output_path: Path,
) -> None:
    config_list = list(configs)
    metrics = {
        config.ticker: market_chart_metrics(all_data[config.ticker], config)
        for config in config_list
    }
    spx = metrics["SPX"]
    ndx = metrics["NDX"]
    spx_options = spx["options"]
    ndx_options = ndx["options"]

    lines = [
        "# Part 1 Results and Discussion",
        "",
        "## Part I — Features in the SPX and NDX charts",
        "",
        "### Data coverage and overall IV levels",
        "",
        (
            f"- **SPX:** {spx['converged_count']:,} of {spx['row_count']:,} rows "
            f"produced a converged IV across {spx['maturity_count']} maturities and "
            f"{spx['strike_count']} union strikes. Valid IVs range from "
            f"{100*float(spx['iv_min']):.2f}% to {100*float(spx['iv_max']):.2f}%."
        ),
        (
            f"- **NDX:** {ndx['converged_count']:,} of {ndx['row_count']:,} rows "
            f"produced a converged IV across {ndx['maturity_count']} maturities and "
            f"{ndx['strike_count']} union strikes. Valid IVs range from "
            f"{100*float(ndx['iv_min']):.2f}% to {100*float(ndx['iv_max']):.2f}%."
        ),
        "",
        "### Smile/skew and term-structure features",
        "",
    ]
    for config in config_list:
        option_metrics = metrics[config.ticker]["options"]
        for option_type in ("Call", "Put"):
            lines.append(
                option_feature_line(
                    config.ticker, option_type, option_metrics[option_type]
                )
            )

    lines.extend(
        [
            "",
            "### Cross-market and Call/Put comparison",
            "",
            (
                "- **NDX is the higher-volatility market near spot:** its median "
                f"nearest-to-spot IV is "
                f"{100*float(ndx_options['Call']['median_atm_iv']):.2f}% for Calls and "
                f"{100*float(ndx_options['Put']['median_atm_iv']):.2f}% for Puts, versus "
                f"{100*float(spx_options['Call']['median_atm_iv']):.2f}% and "
                f"{100*float(spx_options['Put']['median_atm_iv']):.2f}% for SPX."
            ),
            (
                "- **The surfaces are neither flat nor parallel to the strike-maturity "
                "plane.** Both markets show lower IV at higher strikes and material changes "
                "across maturity. NDX also has the wider observed IV range and the higher "
                "long-horizon near-spot level."
            ),
            (
                "- **Call and Put IVs are close in shape but not identical in level.** "
                "For both indices, the median nearest-to-spot Call IV is about one "
                "percentage point above the corresponding Put IV. This gap should be treated "
                "as a quote/input effect to investigate, not automatically as a separate "
                "economic volatility forecast."
            ),
            "",
            "## Part II — Answers to the two Part 1 questions",
            "",
            "### Question 1: What would the charts look like if the Black–Scholes assumptions held?",
            "",
            (
                "Under the strict Black–Scholes assumption, each underlying has one constant "
                "volatility. IV would therefore be independent of both strike and maturity: "
                "every fixed-maturity smile would be horizontal, every fixed-strike term "
                "structure would be horizontal, and each 3D surface would be a flat plane. "
                "Call and Put IVs with the same strike and maturity would also agree when "
                "prices and model inputs satisfy put–call parity."
            ),
            "",
            (
                "The actual charts reject that constant-volatility description. Nearly every "
                "fitted SPX and NDX smile has a negative strike slope, near-spot IV changes "
                "substantially with maturity, NDX sits above SPX, and Call/Put IVs are not "
                "exactly equal. These are systematic features rather than one isolated point "
                "on a chart."
            ),
            "",
            "### Question 2: Why do the actual charts differ from the Black–Scholes prediction?",
            "",
            (
                "- **Downside crash risk and protection demand explain the left skew.** "
                "The observed negative slopes mean low-strike options carry higher IV. Equity "
                "returns have fat left tails and possible jumps, while investors pay for "
                "index downside protection. The leverage effect reinforces this pattern "
                "because falling equity prices are often associated with rising financial "
                "leverage and volatility."
            ),
            (
                "- **Future uncertainty and the variance risk premium help explain the term "
                "structure.** The near-spot IVs generally rise from the first to the last "
                "available maturity in these data. A very short contract covers only the "
                "immediate window, whereas longer contracts span more earnings, monetary-policy, "
                "election, and macroeconomic uncertainty. Investors may also pay a premium to "
                "lock in protection over those longer horizons. This is a market expectation "
                "and risk-premium effect, not a deterministic forecast that realized volatility "
                "must rise."
            ),
            (
                "- **NDX's higher IV is economically plausible.** NDX is more concentrated in "
                "growth and technology companies and is more sensitive to interest rates, "
                "earnings surprises, and changes in long-duration valuations. SPX is broader "
                "and more sector-diversified, so the higher NDX near-spot IV and wider range "
                "are consistent with its greater concentration and factor exposure."
            ),
            (
                "- **Call/Put differences need a market-microstructure explanation.** In an "
                "ideal synchronized market, put–call parity implies matching IVs. Here, prices "
                "are separate Bid/Ask midpoints; spreads, stale or asynchronous quotes, "
                "liquidity differences, and approximations in spot, rates, and dividend yield "
                "can create the observed gap. Low Vega for deep in- or out-of-the-money options "
                "can further magnify a small price discrepancy into a larger IV discrepancy."
            ),
            (
                "- **Not every ridge or isolated surface point is economic.** SPX has "
                f"{int(spx['row_count'])-int(spx['converged_count'])} rows without a converged "
                "IV, and the strike grids vary by maturity. Sparse coverage, wide spreads, and "
                "3D triangulation can make local irregularities look important. Conclusions "
                "should therefore emphasize the repeated skew, term, and cross-market patterns "
                "and check unusual points against the detailed quotes."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_market(
    config: MarketConfig, data_dir: Path, output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_path = data_dir / config.filename
    if not source_path.exists():
        raise FileNotFoundError(f"Missing input file: {source_path}")

    market_output = output_root / config.ticker
    chart_dir = market_output / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    data = read_market_workbook(source_path, config)
    strikes, maturities = active_quote_universe(data)
    call_matrix = build_iv_matrix(data, "Call", strikes, maturities)
    put_matrix = build_iv_matrix(data, "Put", strikes, maturities)

    write_csv_outputs(data, call_matrix, put_matrix, market_output)
    write_results_workbook(
        data,
        call_matrix,
        put_matrix,
        config,
        market_output / f"{config.ticker}_Part1_results.xlsx",
    )

    for option_type, matrix in (("Call", call_matrix), ("Put", put_matrix)):
        name = option_type.lower()
        plot_smiles(
            data,
            config,
            option_type,
            chart_dir / f"{name}_smiles_all_maturities.png",
        )
        plot_term_structures(
            matrix,
            config,
            option_type,
            chart_dir / f"{name}_term_structures_all_strikes.png",
        )
        atm_strikes = nearby_atm_strikes(matrix, config.spot, 21)
        plot_term_structures(
            matrix,
            config,
            option_type,
            chart_dir / f"{name}_term_structures_21_near_atm.png",
            strikes=atm_strikes,
            with_legend=True,
        )
        plot_surface(
            data,
            config,
            option_type,
            chart_dir / f"{name}_volatility_surface_3d.png",
        )

    return data, call_matrix, put_matrix


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calculate SPX and NDX implied-volatility smiles, terms and surfaces."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="Directory containing the two source workbooks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs",
        help="Directory in which generated tables and charts are written.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_data: dict[str, pd.DataFrame] = {}

    for config in MARKETS:
        print(f"Analyzing {config.ticker}...")
        data, call_matrix, put_matrix = analyze_market(
            config, args.data_dir, args.output_dir
        )
        all_data[config.ticker] = data
        print(
            f"  {len(data):,} rows, {int(data['Converged'].sum()):,} converged; "
            f"call matrix {call_matrix.shape}, put matrix {put_matrix.shape}"
        )

    write_analysis_summary(
        all_data, MARKETS, args.output_dir / "analysis_summary.md"
    )
    print(f"Done. Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
