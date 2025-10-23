import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the Momentum Strategy class
class MomentumStrategy:
    def __init__(self, data, short_window=20, long_window=50, risk_factor=0.01):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.risk_factor = risk_factor
        self.signals = pd.DataFrame(index=self.data.index)
        self.positions = pd.DataFrame(index=self.data.index)

    def generate_signals(self):
        self.signals['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        self.signals['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        self.signals['signal'] = 0
        self.signals['signal'][self.short_window:] = np.where(self.signals['short_mavg'][self.short_window:] > self.signals['long_mavg'][self.short_window:], 1, 0)
        self.signals['positions'] = self.signals['signal'].diff()

    def position_sizing(self):
        self.positions['size'] = (self.data['Close'] * self.signals['positions']).fillna(0) * self.risk_factor

    def backtest(self):
        self.data['strategy_returns'] = self.data['Close'].pct_change() * self.signals['signal'].shift(1)
        self.data['cumulative_strategy_returns'] = (1 + self.data['strategy_returns']).cumprod()
        self.data['cumulative_market_returns'] = (1 + self.data['Close'].pct_change()).cumprod()

    def performance_metrics(self):
        sharpe_ratio = np.mean(self.data['strategy_returns']) / np.std(self.data['strategy_returns'])
        max_drawdown = (self.data['cumulative_strategy_returns'].max() - self.data['cumulative_strategy_returns']).max()
        return sharpe_ratio, max_drawdown

# Sample data generation
def generate_sample_data():
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
    prices = np.random.normal(loc=100, scale=10, size=len(dates)).cumsum()
    return pd.DataFrame(data={'Close': prices}, index=dates)

# Main execution block
if __name__ == "__main__":
    try:
        sample_data = generate_sample_data()
        strategy = MomentumStrategy(data=sample_data)
        strategy.generate_signals()
        strategy.position_sizing()
        strategy.backtest()
        sharpe_ratio, max_drawdown = strategy.performance_metrics()
        
        print("Sharpe Ratio: %f" % sharpe_ratio)
        print("Max Drawdown: %f" % max_drawdown)
        
        plt.figure(figsize=(12, 6))
        plt.plot(strategy.data['cumulative_strategy_returns'], label='Strategy Returns')
        plt.plot(strategy.data['cumulative_market_returns'], label='Market Returns')
        plt.title('Momentum Strategy vs Market Returns')
        plt.legend()
        plt.show()
        
    except Exception as e:
        logging.error("An error occurred: %s" % str(e))