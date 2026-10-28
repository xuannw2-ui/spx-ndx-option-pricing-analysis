import math
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from part1_analysis import (  # noqa: E402
    black_scholes_price,
    implied_volatility_newton,
)


class BlackScholesTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "spot": 100.0,
            "strike": 105.0,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "time_to_maturity": 0.75,
            "volatility": 0.27,
        }

    def test_put_call_parity(self):
        call = black_scholes_price("Call", **self.inputs)
        put = black_scholes_price("Put", **self.inputs)
        rhs = (
            self.inputs["spot"]
            * math.exp(
                -self.inputs["dividend_yield"] * self.inputs["time_to_maturity"]
            )
            - self.inputs["strike"]
            * math.exp(-self.inputs["rate"] * self.inputs["time_to_maturity"])
        )
        self.assertAlmostEqual(call - put, rhs, places=10)

    def test_newton_recovers_call_volatility(self):
        market_price = black_scholes_price("Call", **self.inputs)
        result = implied_volatility_newton(
            option_type="Call",
            market_price=market_price,
            spot=self.inputs["spot"],
            strike=self.inputs["strike"],
            rate=self.inputs["rate"],
            dividend_yield=self.inputs["dividend_yield"],
            time_to_maturity=self.inputs["time_to_maturity"],
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["iv"], self.inputs["volatility"], places=9)

    def test_newton_recovers_put_volatility(self):
        market_price = black_scholes_price("Put", **self.inputs)
        result = implied_volatility_newton(
            option_type="Put",
            market_price=market_price,
            spot=self.inputs["spot"],
            strike=self.inputs["strike"],
            rate=self.inputs["rate"],
            dividend_yield=self.inputs["dividend_yield"],
            time_to_maturity=self.inputs["time_to_maturity"],
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["iv"], self.inputs["volatility"], places=9)

    def test_rejects_price_outside_no_arbitrage_bounds(self):
        result = implied_volatility_newton(
            option_type="Call",
            market_price=200.0,
            spot=100.0,
            strike=105.0,
            rate=0.05,
            dividend_yield=0.02,
            time_to_maturity=0.75,
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "outside_no_arbitrage_bounds")

    def test_rejects_price_outside_configured_volatility_bounds(self):
        market_price = black_scholes_price("Call", **self.inputs)
        result = implied_volatility_newton(
            option_type="Call",
            market_price=market_price,
            spot=self.inputs["spot"],
            strike=self.inputs["strike"],
            rate=self.inputs["rate"],
            dividend_yield=self.inputs["dividend_yield"],
            time_to_maturity=self.inputs["time_to_maturity"],
            max_volatility=0.20,
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "outside_volatility_bounds")

    def test_uses_brent_when_vega_is_too_small(self):
        market_price = black_scholes_price("Call", **self.inputs)
        result = implied_volatility_newton(
            option_type="Call",
            market_price=market_price,
            spot=self.inputs["spot"],
            strike=self.inputs["strike"],
            rate=self.inputs["rate"],
            dividend_yield=self.inputs["dividend_yield"],
            time_to_maturity=self.inputs["time_to_maturity"],
            vega_tolerance=1e9,
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["method"], "Brent")
        self.assertAlmostEqual(result["iv"], self.inputs["volatility"], places=9)


if __name__ == "__main__":
    unittest.main()
