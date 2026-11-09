"""Monthly demand forecasting (optional Module E)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import FORECAST_HORIZON

HW_NAME = "Holt-Winters (damped, seasonal)"
SN_NAME = "Seasonal naive (baseline)"
MA_NAME = "Moving average (3 months)"


def monthly_demand(sales):
    """Units sold per month by product category (cancelled orders excluded), plus a Total column."""
    active = sales[sales["is_cancelled"] == 0]
    month = active["Sale_Date"].dt.to_period("M").dt.to_timestamp().rename("year_month")
    table = active.pivot_table(index=month, columns="Product_Category", values="Quantity", aggfunc="sum").fillna(0)
    table["Total"] = table.sum(axis=1)
    return table


def _holt_winters(y, horizon, period=12):
    """Damped additive trend with additive seasonality.

    Uses statsmodels when it is installed (Anaconda includes it); otherwise a built-in version with
    the same equations, whose parameters are fitted by least squares.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(np.asarray(y, float), trend="add", damped_trend=True, seasonal="add",
                                   seasonal_periods=period).fit()
        return np.asarray(fit.forecast(horizon))
    except ImportError:
        pass
    y = np.asarray(y, float)
    level0 = y[:period].mean()
    trend0 = (y[period:2 * period].mean() - y[:period].mean()) / period
    season0 = y[:period] - level0

    def run(params, steps=0):
        alpha, beta, gamma, phi = params
        level, trend, season = level0, trend0, list(season0)
        fitted = []
        for t, obs in enumerate(y):
            s = season[t % period]
            fitted.append(level + phi * trend + s)
            new_level = alpha * (obs - s) + (1 - alpha) * (level + phi * trend)
            trend = beta * (new_level - level) + (1 - beta) * phi * trend
            season[t % period] = gamma * (obs - new_level) + (1 - gamma) * s
            level = new_level
        forecast, damp = [], 0.0
        for h in range(1, steps + 1):
            damp += phi ** h
            forecast.append(level + damp * trend + season[(len(y) + h - 1) % period])
        return np.array(fitted), np.array(forecast)

    def sse(params):
        fitted, _ = run(params)
        return np.sum((y - fitted) ** 2)

    bounds = [(0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.80, 0.995)]
    starts = [(0.3, 0.05, 0.1, 0.9), (0.1, 0.01, 0.3, 0.95), (0.5, 0.1, 0.05, 0.85), (0.2, 0.0001, 0.0001, 0.98)]
    best = min((minimize(sse, s, bounds=bounds, method="L-BFGS-B") for s in starts), key=lambda r: r.fun)
    return run(best.x, horizon)[1]


def evaluate_series(series, horizon=FORECAST_HORIZON):
    """Hold out the last `horizon` months and compare three methods. Returns (accuracy, forecasts)."""
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    forecasts = {
        SN_NAME: train.iloc[-12:].to_numpy()[:horizon],
        MA_NAME: np.repeat(train.iloc[-3:].mean(), horizon),
        HW_NAME: _holt_winters(train.to_numpy(), horizon),
    }
    actual = test.to_numpy(float)
    rows = {}
    for name, f in forecasts.items():
        err = f - actual
        rows[name] = {"WAPE": np.abs(err).sum() / actual.sum(), "RMSE": np.sqrt(np.mean(err ** 2)),
                      "Bias": err.sum() / actual.sum()}
    accuracy = pd.DataFrame(rows).T.rename_axis("method")
    accuracy["WAPE_improvement_vs_baseline"] = 1 - accuracy["WAPE"] / accuracy.loc[SN_NAME, "WAPE"]
    forecast_table = pd.DataFrame({"month": test.index, "actual": actual, **forecasts})
    return accuracy, forecast_table
