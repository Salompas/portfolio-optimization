from typing import List

import pandas as pd
import numpy as np
from stock import Stock


class Portfolio:
    def __init__(self, *stocks: List[Stock]):
        self.stocks = {s.symbol: s for s in stocks}
        self.weights = {s.symbol: 1 / len(self.stocks) for s in self.stocks.values()}

    def __repr__(self):
        return f"Portfolio({', '.join(repr(s) for s in self.stocks.values())})"

    @property
    def intraday(self):
        intraday = pd.concat([s.intraday for s in self.stocks.values()], axis=1)
        intraday.columns = self.stocks.keys()
        return intraday

    @property
    def daily(self):
        daily = pd.concat([s.daily for s in self.stocks.values()], axis=1)
        daily.columns = self.stocks.keys()
        return daily

    @property
    def rv(self):
        rv = pd.concat([s.rv for s in self.stocks.values()], axis=1)
        rv.columns = self.stocks.keys()
        return rv.set_index(pd.DatetimeIndex(rv.index))

    @property
    def intraday_price(self):
        return self.intraday.dot(list(self.weights.values()))

    @property
    def daily_price(self):
        return self.daily.dot(list(self.weights.values()))

    @property
    def intraday_ret(self):
        return self.intraday_price.groupby(by=self.intraday_price.index.date).apply(
            lambda df: np.log(df).diff()
        )

    @property
    def portfolio_rv(self):
        rv = self.intraday_ret.groupby(self.intraday_ret.index.date).apply(
            lambda df: (df ** 2).sum()
        )
        return rv.reindex(pd.DatetimeIndex(rv.index))

    def risk(self, start_date=None, end_date=None):
        if start_date is None and end_date is None:
            risk = self.portfolio_rv.sum()
        else:
            risk = self.portfolio_rv.loc[start_date:end_date].sum()
        return risk

    def return_between(self, start_date, end_date):
        ret = 0
        for s in self.stocks:
            ret += self.weights[s] * self.stocks[s].return_between(start_date, end_date)
        return ret

    def optimize(self, portfolio=None, date=None):
        if portfolio is None:
            optimized = self
        else:
            optimized = Portfolio(*self.stocks.values(), *portfolio.stocks.values())
        if date is None:
            inverse = 1 / optimized.rv.iloc[-1, :]
        else:
            inverse = 1 / optimized.rv.loc[date, :]
        optimized.weights = (inverse / inverse.sum()).to_dict()
        return optimized

    def backtest(self, start_date, end_date):
        optimized = self.optimize(self, start_date)
        total_return = optimized.return_between(start_date, end_date)
        risk = optimized.risk(start_date, end_date)
        return total_return, risk
