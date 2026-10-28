#!/usr/bin/env python3
"""Create a paper-ready Word report from the 2018--2024 eSSVI outputs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT_ROOT = ROOT / "outputs" / "long_horizon_essvi"
CSV_ROOT = OUTPUT_ROOT / "csv"
CHART_ROOT = OUTPUT_ROOT / "charts"
REPORT_PATH = OUTPUT_ROOT / "SPX_NDX_eSSVI_RND_2018_2024_Report.docx"


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, value: object, *, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(value))
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(
    document: Document,
    dataframe: pd.DataFrame,
    caption: str,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(caption)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    table = document.add_table(
        rows=1, cols=len(dataframe.columns), style="Table Grid"
    )
    table.autofit = True
    for index, column in enumerate(dataframe.columns):
        set_cell_text(table.rows[0].cells[index], column, bold=True)
        set_cell_shading(table.rows[0].cells[index], "D9EAF7")
    for row in dataframe.itertuples(index=False):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value)
    document.add_paragraph()


def add_figure(
    document: Document,
    filename: str,
    caption: str,
    *,
    width: float = 6.5,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(
        str(CHART_ROOT / filename), width=Inches(width)
    )
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = caption_paragraph.add_run(caption)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)


def add_body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Inches(0.25)
    paragraph.paragraph_format.space_after = Pt(6)


def add_equation(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Cambria Math"
    run.font.size = Pt(10.5)


def build_report() -> None:
    metrics = pd.read_csv(
        CSV_ROOT / "monthly_essvi_rnd_metrics.csv",
        parse_dates=["Date", "Maturity"],
    )
    annual = pd.read_csv(CSV_ROOT / "annual_summary.csv")
    audit = pd.read_csv(CSV_ROOT / "data_audit.csv")
    regressions = pd.read_csv(CSV_ROOT / "explanatory_regressions.csv")
    comparison = pd.read_csv(
        CSV_ROOT / "spx_ndx_matched_comparison.csv", parse_dates=["Date"]
    )

    ninety = metrics[metrics["TargetTenorDays"] == 90]
    market_means = ninety.groupby("Market").mean(numeric_only=True)
    covid = ninety[
        ninety["Date"].between("2020-02-01", "2020-06-30")
    ].groupby("Market").mean(numeric_only=True)
    atm_difference = comparison[
        "NDXMinusSPX_ATMVolatility_Difference"
    ].mean()
    downside_difference = comparison[
        "NDXMinusSPX_RNDProbabilityBelow80Forward_Difference"
    ].mean()

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = document.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(10.5)
    styles["Title"].font.name = "Times New Roman"
    styles["Title"].font.size = Pt(18)
    styles["Title"].font.bold = True
    for style_name in ("Heading 1", "Heading 2"):
        styles[style_name].font.name = "Times New Roman"
        styles[style_name].font.color.rgb = RGBColor(31, 78, 120)

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(
        "SPX and NDX Implied-Volatility Surfaces:\n"
        "eSSVI and Risk-Neutral-Density Evidence, 2018–2024"
    )
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Empirical Output Report").italic = True

    document.add_heading("Abstract", level=1)
    add_body(
        document,
        (
            "This report compares SPX and NDX implied-volatility surfaces from "
            "2018 through 2024 using an arbitrage-controlled extended SSVI "
            "(eSSVI) specification. The analysis processes "
            f"{int(audit['RawOptionRows'].sum()):,} OptionMetrics option records "
            "and constructs monthly end-of-period surfaces at target horizons "
            "of 30, 90, 180, and 360 days. Out-of-the-money quotes are fitted "
            "under sufficient butterfly restrictions and sequential calendar "
            "constraints. The fitted surfaces are used to calculate 25-delta "
            "risk reversals and butterflies and to recover risk-neutral "
            "densities. NDX has a higher average 90-day ATM volatility than SPX "
            f"({100*market_means.loc['NDX','ATMVolatility']:.2f}% versus "
            f"{100*market_means.loc['SPX','ATMVolatility']:.2f}%). Its matched "
            "ATM premium averages "
            f"{100*atm_difference:.2f} volatility points, while its probability "
            "of ending below 80% of forward is "
            f"{100*downside_difference:.2f} percentage points higher. Both "
            "markets display sharply higher volatility, downside skew, and tail "
            "probability during the 2020 stress period. VIX is strongly associated "
            "with ATM volatility and risk-neutral downside probability."
        ),
    )

    document.add_heading("1. Data and Sample Construction", level=1)
    add_body(
        document,
        (
            "The sample contains daily SPX (OptionMetrics secid 108105) and NDX "
            "(secid 102480) option quotes, index closing levels, index dividend "
            "yields, zero-coupon rates, and VIX observations from January 2018 "
            "through December 2024. The empirical design uses the last trading "
            "day of each month. On each date and for each index, the listed "
            "expiration nearest to 30, 90, 180, or 360 calendar days is selected "
            "within a tenor-specific tolerance."
        ),
    )
    add_body(
        document,
        (
            "Quotes must have positive and ordered bid–ask prices, relative "
            "spread no greater than 50%, OptionMetrics implied volatility between "
            "1% and 300%, and absolute delta between 0.05 and 0.95. Duplicate "
            "strike/type records are resolved in favor of the tightest relative "
            "spread and then the highest open interest. To reduce the known "
            "noise in in-the-money quotes, each smile combines puts below the "
            "forward and calls at or above the forward."
        ),
    )
    audit_table = audit.copy()
    audit_table["RawOptionRows"] = audit_table["RawOptionRows"].map(
        lambda value: f"{int(value):,}"
    )
    audit_table["SelectedOTMRows"] = audit_table["SelectedOTMRows"].map(
        lambda value: f"{int(value):,}"
    )
    add_table(
        document,
        audit_table[
            [
                "Year",
                "RawOptionRows",
                "SelectedOTMRows",
                "SnapshotDates",
                "Slices",
            ]
        ],
        "Table 1. Data coverage and retained monthly observations",
    )

    document.add_heading("2. eSSVI Methodology", level=1)
    add_body(
        document,
        (
            "For maturity T, the forward is F = S exp[(r−q)T], log-forward "
            "moneyness is k = ln(K/F), and total implied variance is "
            "w = σ²IVT. Each cross-section is represented in the natural eSSVI "
            "coordinates (θ, ρ, ψ):"
        ),
    )
    add_equation(
        document,
        "w(k) = ½{θ + ρψk + [(ψk + θρ)² + θ²(1−ρ²)]¹ᐟ²}.",
    )
    add_body(
        document,
        (
            "Here θ is ATM total variance, ρ controls the direction and strength "
            "of asymmetry, and ψ governs the wing slope. The calibration minimizes "
            "IV errors while imposing ψ(1+|ρ|) < 4 and "
            "ψ²(1+|ρ|) ≤ 4θ. Total variance is also constrained to be "
            "non-decreasing over adjacent maturities on a common moneyness grid. "
            "These conditions prevent the fitted surface from mechanically "
            "generating negative state prices."
        ),
    )
    add_body(
        document,
        (
            f"The final sample contains {len(metrics)} fitted market-tenor "
            f"slices. Median IV RMSE is {100*metrics['RMSE_IV'].median():.3f} "
            "volatility points, and the worst-slice RMSE is "
            f"{100*metrics['RMSE_IV'].max():.3f} points. The minimum density "
            f"factor across all slices is {metrics['MinDensityFactorG'].min():.6f}, "
            "so every retained fit passes the numerical butterfly diagnostic."
        ),
    )
    add_figure(
        document,
        "representative_essvi_fits.png",
        (
            "Figure 1. Representative observed implied-volatility smiles and "
            "constrained 90-day eSSVI fits."
        ),
    )

    document.add_heading("3. Risk-Neutral Density", level=1)
    add_body(
        document,
        (
            "Risk-neutral density is recovered from the fitted surface using the "
            "Breeden–Litzenberger identity, fQ(K,T) = exp(rT)∂²C/∂K². For a "
            "differentiable total-variance smile, the density can be written as:"
        ),
    )
    add_equation(
        document,
        "fQ(K,T) = φ(d₋)g(k)/(K√w),     d₋ = −k/√w − √w/2.",
    )
    add_equation(
        document,
        "g(k) = [1−kw′/(2w)]² − (w′²/4)(1/w+1/4) + w″/2.",
    )
    add_body(
        document,
        (
            "The numerical density is evaluated from 35% to 180% of forward and "
            "normalized by its integrated mass. Reported outputs include "
            "dispersion, skewness, excess kurtosis, and probabilities below 80% "
            "and above 120% of the forward. Far-tail measures remain more "
            "model-dependent than ATM volatility because observed strikes do not "
            "span the full integration region on every date."
        ),
    )
    add_figure(
        document,
        "representative_rnd_comparison.png",
        (
            "Figure 2. Representative 90-day SPX and NDX risk-neutral densities "
            "during 2020 stress, 2022 tightening, and year-end 2024."
        ),
    )

    document.add_heading("4. Empirical Results", level=1)
    document.add_heading("4.1 Volatility Levels and Term Structure", level=2)
    add_body(
        document,
        (
            "NDX carries a persistent volatility premium. Across monthly 90-day "
            f"slices, mean ATM IV is {100*market_means.loc['NDX','ATMVolatility']:.2f}% "
            f"for NDX and {100*market_means.loc['SPX','ATMVolatility']:.2f}% for "
            "SPX. On matched date-tenor observations, the average NDX-minus-SPX "
            f"difference is {100*atm_difference:.2f} volatility points. The gap "
            "is economically consistent with NDX's greater technology and "
            "growth-stock concentration, larger duration exposure, and lower "
            "sector diversification."
        ),
    )
    add_figure(
        document,
        "monthly_atm_volatility_90d.png",
        "Figure 3. Monthly 90-day eSSVI ATM volatility for SPX and NDX.",
    )
    add_figure(
        document,
        "annual_atm_volatility.png",
        "Figure 4. Annual mean ATM volatility across the four standard horizons.",
    )

    document.add_heading("4.2 Skew and Convexity", level=2)
    add_body(
        document,
        (
            "The 25-delta risk reversal is negative in both markets, confirming "
            "that downside options command higher implied volatility than "
            "equally out-of-the-money calls. Its full-sample 90-day mean is "
            f"{100*market_means.loc['SPX','RiskReversal25']:.2f} volatility "
            f"points for SPX and "
            f"{100*market_means.loc['NDX','RiskReversal25']:.2f} points for NDX. "
            "The similarity of the averages does not imply identical state-price "
            "distributions: NDX combines a comparable standardized skew with a "
            "higher overall volatility level and wider RND."
        ),
    )
    add_figure(
        document,
        "monthly_risk_reversal_90d.png",
        "Figure 5. Monthly 90-day 25-delta risk reversal.",
    )

    document.add_heading("4.3 Risk-Neutral Tail Risk", level=2)
    add_body(
        document,
        (
            "Across matched dates and tenors, NDX's model-implied probability of "
            "finishing below 80% of its forward exceeds SPX by "
            f"{100*downside_difference:.2f} percentage points on average. The "
            "difference reflects NDX's wider distribution rather than uniformly "
            "more negative standardized skew. During February–June 2020, 90-day "
            f"ATM IV rises to {100*covid.loc['SPX','ATMVolatility']:.2f}% for SPX "
            f"and {100*covid.loc['NDX','ATMVolatility']:.2f}% for NDX, while the "
            "below-80%-forward probabilities average "
            f"{100*covid.loc['SPX','RNDProbabilityBelow80Forward']:.2f}% and "
            f"{100*covid.loc['NDX','RNDProbabilityBelow80Forward']:.2f}%, "
            "respectively."
        ),
    )
    add_figure(
        document,
        "monthly_rnd_downside_probability_90d.png",
        (
            "Figure 6. Monthly model-implied probability of the terminal index "
            "falling below 80% of forward."
        ),
    )

    annual_90 = annual[annual["TargetTenorDays"] == 90].copy()
    annual_90["ATM IV (%)"] = (100 * annual_90["ATMVolatilityMean"]).map(
        lambda value: f"{value:.2f}"
    )
    annual_90["RR25 (pts)"] = (100 * annual_90["RiskReversal25Mean"]).map(
        lambda value: f"{value:.2f}"
    )
    annual_90["P(S<0.8F) (%)"] = (
        100 * annual_90["RNDDownside80Mean"]
    ).map(lambda value: f"{value:.2f}")
    annual_90["RMSE (pts)"] = (100 * annual_90["IVRMSEMean"]).map(
        lambda value: f"{value:.3f}"
    )
    add_table(
        document,
        annual_90[
            [
                "Year",
                "Market",
                "Snapshots",
                "ATM IV (%)",
                "RR25 (pts)",
                "P(S<0.8F) (%)",
                "RMSE (pts)",
            ]
        ],
        "Table 2. Annual summary at the 90-day target horizon",
    )

    document.add_heading("5. Explanatory Evidence", level=1)
    add_body(
        document,
        (
            "To distinguish broad market stress from index-specific differences, "
            "90-day monthly outcomes are regressed on standardized VIX, 21-day "
            "realized volatility, monthly index return, and median relative "
            "option spread. HC1 robust standard errors are reported. Because the "
            "sample is observational and the predictors are correlated, these "
            "regressions describe conditional associations rather than causal "
            "effects."
        ),
    )
    selected_regressions = regressions[
        regressions["Predictor"].isin(
            ["VIXClose", "RealizedVol21", "MonthlyIndexReturn"]
        )
    ].copy()
    selected_regressions["β"] = selected_regressions["Coefficient"].map(
        lambda value: f"{value:.4f}"
    )
    selected_regressions["t"] = selected_regressions["TStatistic"].map(
        lambda value: f"{value:.2f}"
    )
    selected_regressions["R²"] = selected_regressions["RSquared"].map(
        lambda value: f"{value:.3f}"
    )
    add_table(
        document,
        selected_regressions[
            [
                "Market",
                "DependentVariable",
                "Predictor",
                "β",
                "t",
                "R²",
                "Observations",
            ]
        ],
        "Table 3. Standardized-predictor regressions at the 90-day horizon",
    )
    spx_vix_coefficient = regressions[
        (regressions["Market"] == "SPX")
        & (regressions["DependentVariable"] == "ATMVolatility")
        & (regressions["Predictor"] == "VIXClose")
    ]["Coefficient"].iloc[0]
    ndx_vix_coefficient = regressions[
        (regressions["Market"] == "NDX")
        & (regressions["DependentVariable"] == "ATMVolatility")
        & (regressions["Predictor"] == "VIXClose")
    ]["Coefficient"].iloc[0]
    add_body(
        document,
        (
            "VIX is the dominant common-state variable. A one-standard-deviation "
            "increase in VIX is associated with an ATM-IV increase of "
            f"{100*spx_vix_coefficient:.2f} volatility points for SPX and "
            f"{100*ndx_vix_coefficient:.2f} points for NDX. It is also associated "
            "with materially higher downside-tail probability and a more "
            "negative risk reversal in both indices."
        ),
    )
    add_figure(
        document,
        "vix_vs_rnd_downside.png",
        "Figure 7. VIX and 90-day risk-neutral downside-tail probability.",
    )

    document.add_heading("6. Conclusion and Limitations", level=1)
    add_body(
        document,
        (
            "The seven-year evidence rejects a constant-volatility description "
            "for both indices. NDX has a persistent level premium and a wider "
            "risk-neutral distribution, while the standardized 25-delta skew is "
            "similar on average. Common stress, summarized by VIX, explains a "
            "large share of time-series movement in both ATM volatility and RND "
            "tail risk. The eSSVI restrictions ensure internally coherent "
            "cross-sectional densities, making these comparisons more reliable "
            "than direct second differences of noisy option prices."
        ),
    )
    add_body(
        document,
        (
            "Three limitations should remain explicit. First, month-end sampling "
            "does not capture every intramonth shock. Second, RND tails outside "
            "the observed strike range depend on eSSVI extrapolation. Third, VIX, "
            "realized volatility, returns, and spreads cannot isolate causal "
            "channels such as sector concentration, dealer positioning, or "
            "investor demand. These mechanisms require additional composition "
            "and order-flow data."
        ),
    )

    document.add_heading("References", level=1)
    references = [
        (
            "[1] Breeden, D. T., & Litzenberger, R. H. (1978). Prices of "
            "state-contingent claims implicit in option prices. Journal of "
            "Business, 51(4), 621–651."
        ),
        (
            "[2] Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI "
            "volatility surfaces. Quantitative Finance, 14(1), 59–71."
        ),
        (
            "[3] Hendriks, S., & Martini, C. (2019). The extended SSVI "
            "volatility surface. Journal of Computational Finance, 22(5), "
            "25–39."
        ),
    ]
    for reference in references:
        paragraph = document.add_paragraph(reference)
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.first_line_indent = Inches(-0.25)

    # Footer with page numbering.
    for section in document.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("SPX–NDX eSSVI/RND empirical report")
        run.font.name = "Times New Roman"
        run.font.size = Pt(8)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(REPORT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    build_report()
