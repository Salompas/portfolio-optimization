import os

from alpha_vantage.timeseries import TimeSeries
import numpy as np

# Load API key from config file
with open(os.path.join(".", "data", "config.txt")) as f:
    api_key = f.readline().rstrip("\n")

# Initialize API wrapper class
ts = TimeSeries(key=api_key, output_format="pandas")


class Stock:
    def __init__(self, symbol: str, intraday=None, daily=None):
        self.symbol = symbol
        if intraday is None or daily is None:
            self.intraday, _ = ts.get_intraday(
                symbol, interval="5min", outputsize="full"
            )
            self.daily, _ = ts.get_daily(symbol)
            # Select variables
            drop = ["1. open", "2. high", "3. low", "5. volume"]
            rename = {"4. close": "price"}
            self.intraday = (
                self.intraday.drop(columns=drop).rename(columns=rename).sort_index()
            )
            self.daily = (
                self.daily.drop(columns=drop).rename(columns=rename).sort_index()
            )
        else:
            self.intraday = intraday
            self.daily = daily

    def __repr__(self):
        return f"Stock({self.symbol})"

    @property
    def intraday_ret(self):
        return (
            self.intraday.groupby(by=self.intraday.index.date)
            .apply(lambda df: np.log(df).diff())
            .rename(columns={"price": "return"})
        )

    @property
    def rv(self):
        return (
            self.intraday_ret.groupby(by=self.intraday_ret.index.date)
            .apply(lambda df: (df ** 2).sum())
            .rename(columns={"return": "rv"})
        )

    def return_between(self, start_date, end_date):
        return float(self.daily.loc[end_date] / self.daily.loc[start_date] - 1)
