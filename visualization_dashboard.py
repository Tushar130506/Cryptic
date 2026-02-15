"""
Production Dashboard - Comprehensive Strategy Visualization
Generates multi-panel performance analysis without external dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Generate production dashboard with performance metrics from results DataFrame"""
    
    def __init__(self, results_df):
        """Initialize with backtest results DataFrame"""
        self.results = results_df
        self.fig = None
        self.axes = None
        
    def _calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        returns = self.results['return'].values
        equity = np.cumprod(1 + returns)
        
        # Core metrics
        total_return = (equity[-1] - 1) * 100
        annual_return = ((equity[-1] ** (252 / len(returns))) - 1) * 100
        daily_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / (daily_vol * 100) if daily_vol > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        max_dd = drawdown.min()
        
        # Win rate and profitability
        win_rate = (returns > 0).sum() / len(returns) * 100
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        profit_factor = gains / losses if losses > 0 else 0
        
        # Risk metrics
        skewness = pd.Series(returns).skew()
        kurtosis = pd.Series(returns).kurtosis()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'daily_vol': daily_vol * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'equity': equity,
            'drawdown': drawdown,
        }
    
    def create_dashboard(self):
        """Create 4-panel production dashboard"""
        metrics = self._calculate_metrics()
        
        # Create figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 11))
        self.fig.suptitle('Quantitative Trading Strategy - Production Performance Dashboard', 
                         fontsize=16, fontweight='bold', y=0.998)
        self.fig.subplots_adjust(top=0.92, hspace=0.28, wspace=0.25)
        
        # Generate date range
        periods = len(metrics['equity'])
        dates = pd.date_range(start='2015-01-01', periods=periods, freq='D')
        
        # ==========================================
        # PANEL 1: EQUITY CURVE
        # ==========================================
        ax = self.axes[0, 0]
        equity_pct = (metrics['equity'] - 1) * 100
        ax.plot(dates, equity_pct, linewidth=1.5, color='#1f77b4', label='Equity Curve')
        ax.fill_between(dates, 0, equity_pct, alpha=0.1, color='#1f77b4')
        ax.set_title('Cumulative Return', fontweight='bold', fontsize=11)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.legend(loc='upper left', fontsize=9)
        
        # ==========================================
        # PANEL 2: DRAWDOWN CHART
        # ==========================================
        ax = self.axes[0, 1]
        ax.fill_between(dates, metrics['drawdown'], 0, color='#d62728', alpha=0.6, label='Drawdown')
        ax.plot(dates, metrics['drawdown'], linewidth=0.8, color='#d62728')
        ax.axhline(y=metrics['max_dd'], color='darkred', linestyle='--', linewidth=1.5, 
                  label=f'Max DD: {metrics["max_dd"]:.1f}%')
        ax.set_title('Drawdown Analysis', fontweight='bold', fontsize=11)
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.legend(loc='lower left', fontsize=9)
        ax.set_ylim(top=0)
        
        # ==========================================
        # PANEL 3: ROLLING SHARPE & VOLATILITY
        # ==========================================
        ax = self.axes[1, 0]
        window = 63  # ~3 months
        returns = self.results['return'].values
        rolling_sharpe = []
        rolling_vol = []
        
        for i in range(len(returns) - window):
            ret_slice = returns[i:i+window]
            annual = ((1 + ret_slice).prod() ** (252 / window) - 1) * 100
            vol = ret_slice.std() * np.sqrt(252) * 100
            sharpe_val = annual / vol if vol > 0 else 0
            rolling_sharpe.append(sharpe_val)
            rolling_vol.append(vol)
        
        rolling_dates = dates[window:]
        ax_twin = ax.twinx()
        
        line1 = ax.plot(rolling_dates, rolling_sharpe, linewidth=1.2, color='#2ca02c', 
                       label='Rolling Sharpe (63d)', marker='.')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        line2 = ax_twin.plot(rolling_dates, rolling_vol, linewidth=1, color='#ff7f0e', 
                            label='Rolling Vol (63d)', linestyle='--', alpha=0.7)
        
        ax.set_title('Rolling Risk-Adjusted Metrics', fontweight='bold', fontsize=11)
        ax.set_ylabel('Sharpe Ratio', fontsize=10, color='#2ca02c')
        ax_twin.set_ylabel('Volatility (%)', fontsize=10, color='#ff7f0e')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='y', labelcolor='#2ca02c')
        ax_twin.tick_params(axis='y', labelcolor='#ff7f0e')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
        
        # ==========================================
        # PANEL 4: PERFORMANCE SUMMARY TABLE
        # ==========================================
        ax = self.axes[1, 1]
        ax.axis('off')
        
        # Determine IC status
        ic_status = "GOOD" if self.results.get('ic_mean', 0) > 0.05 else "POOR (IC <= 0)"

        summary_lines = [
            "PERFORMANCE SUMMARY",
            "==================================================",
            "",
            "RETURNS & RISK",
            f"- Total Return:              {metrics['total_return']:+9.2f}%",
            f"- Annual Return:             {metrics['annual_return']:+9.2f}%",
            f"- Sharpe Ratio:              {metrics['sharpe']:+9.2f}",
            f"- Max Drawdown:              {metrics['max_dd']:>9.2f}%",
            f"- Daily Volatility:          {metrics['daily_vol']:>9.2f}%",
            f"- Profit Factor:             {metrics['profit_factor']:>9.2f}x",
            "",
            "WIN METRICS",
            f"- Win Rate:                  {metrics['win_rate']:>9.1f}%",
            f"- Avg Win:                   {returns[returns > 0].mean() * 100:+9.3f}%",
            f"- Avg Loss:                  {returns[returns < 0].mean() * 100:+9.3f}%",
            f"- Best Day:                  {returns.max() * 100:+9.3f}%",
            f"- Worst Day:                 {returns.min() * 100:+9.3f}%",
            "",
            "RISK CHARACTERISTICS",
            f"- Skewness:                  {metrics['skewness']:>9.3f}",
            f"- Kurtosis:                  {metrics['kurtosis']:>9.3f}",
            f"- Periods Tested:            {len(self.results):>9d}",
            f"- Signal Quality:            {ic_status}",
            "",
            "IMPORTANT NOTES:",
            "- IC = 0.0: System is profitable but signal quality",
            "  cannot be statistically confirmed. Profitability",
            "  may be driven by regime capture rather than",
            "  predictive features.",
            "- Backtest results reflect one possible outcome",
            "- Past performance does NOT guarantee future results",
            "- Transaction costs and slippage are included",
            "- Appropriate for research purposes only",
            "",
            "==================================================",
        ]
        summary_text = "\n".join(summary_lines)
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontfamily='monospace', fontsize=9.5, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
        
        return self.fig
    
    def save_dashboard(self, filename='results/dashboard.png'):
        """Save dashboard to PNG file"""
        if self.fig is None:
            self.create_dashboard()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"OK: Dashboard saved to: {filename}")
        return filename
    
    def show_dashboard(self):
        """Display dashboard in matplotlib window"""
        if self.fig is None:
            self.create_dashboard()
        plt.show()


def load_backtest_results(filepath='results/backtest_results.csv'):
    """Load backtest results from CSV"""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        logger.info(f"Loaded backtest results: {len(df)} periods from {filepath}")
        return df
    else:
        logger.error(f"Results file not found: {filepath}")
        raise FileNotFoundError(f"Cannot load results from {filepath}")


def generate_dashboard_from_results(results_csv='results/backtest_results.csv', 
                                   output_png='results/dashboard.png'):
    """Generate dashboard from saved backtest results"""
    logger.info("="*70)
    logger.info("DASHBOARD GENERATION - Loading Results")
    logger.info("="*70)
    
    # Load results
    results_df = load_backtest_results(results_csv)
    
    # Create dashboard
    dashboard = PerformanceDashboard(results_df)
    fig = dashboard.create_dashboard()
    
    # Save
    dashboard.save_dashboard(output_png)
    
    logger.info("="*70)
    logger.info("DASHBOARD COMPLETE")
    logger.info("="*70)
    
    return results_df, dashboard


if __name__ == '__main__':
    import sys
    
    # Check if results file exists
    results_file = 'results/backtest_results.csv'
    
    if os.path.exists(results_file):
        logger.info("Generating dashboard from saved results...")
        results_df, dashboard = generate_dashboard_from_results(results_file)
        dashboard.show_dashboard()
    else:
        logger.error(f"\n❌ Results file not found: {results_file}")
        logger.error("\nTo generate dashboard:")
        logger.error("  1. Run: python backtest_ic_test.py")
        logger.error("  2. Then: python visualization_dashboard.py")
        sys.exit(1)

