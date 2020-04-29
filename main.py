import os
import pickle

from stock import Stock
from portfolio import Portfolio

if __name__ == "__main__":
    # Load stored stocks
    stocks = {}
    for s in os.listdir(os.path.join(".", "data", "stock")):
        with open(os.path.join(".", "data", "stock", s), "rb") as f:
            stocks[s] = pickle.load(f)


    print(stocks)
    print(Portfolio(*stocks.values()))
