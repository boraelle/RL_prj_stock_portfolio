"""
Utility Functions for RL Portfolio Management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os


def calculate_metrics(capital_history: List[float], initial_capital: float) -> Dict:
    """
    포트폴리오 성과 지표 계산
    
    Args:
        capital_history: 일별 자산 가치 리스트
        initial_capital: 초기 자본
    
    Returns:
        dict: 성과 지표 (수익률, 샤프비율, MDD 등)
    """
    capital_history = np.array(capital_history)
    
    # 일간 수익률
    daily_returns = np.diff(capital_history) / capital_history[:-1]
    
    # 총 수익률
    total_return = (capital_history[-1] / initial_capital) - 1
    
    # 연환산 수익률 (252 거래일 기준)
    n_days = len(capital_history)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    # 샤프 비율 (무위험 수익률 0 가정)
    if len(daily_returns) > 1:
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # 최대 낙폭 (MDD)
    cummax = np.maximum.accumulate(capital_history)
    drawdown = (capital_history - cummax) / cummax
    mdd = np.min(drawdown)
    
    # 변동성 (연환산)
    volatility = np.std(daily_returns) * np.sqrt(252)
    
    return {
        'total_return': total_return * 100,  # %
        'annualized_return': annualized_return * 100,  # %
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': mdd * 100,  # %
        'volatility': volatility * 100,  # %
        'final_capital': capital_history[-1]
    }


def plot_portfolio_performance(
    capital_history: List[float],
    initial_capital: float,
    save_path: str = None
):
    """
    포트폴리오 성과 시각화
    
    Args:
        capital_history: 일별 자산 가치
        initial_capital: 초기 자본
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 자산 가치 추이
    ax1 = axes[0, 0]
    ax1.plot(capital_history, linewidth=2, color='#2E86AB')
    ax1.axhline(initial_capital, color='red', linestyle='--', alpha=0.5, label='초기자본')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Capital (KRW)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # 2. 누적 수익률
    ax2 = axes[0, 1]
    cumulative_returns = [(c / initial_capital - 1) * 100 for c in capital_history]
    ax2.plot(cumulative_returns, linewidth=2, color='#A23B72')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Return (%)')
    ax2.grid(alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[1, 0]
    cummax = np.maximum.accumulate(capital_history)
    drawdown = ((np.array(capital_history) - cummax) / cummax) * 100
    ax3.fill_between(range(len(drawdown)), drawdown, 0, color='#F18F01', alpha=0.6)
    ax3.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(alpha=0.3)
    
    # 4. 성과 지표 텍스트
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics = calculate_metrics(capital_history, initial_capital)
    
    metrics_text = f"""
    📊 Performance Metrics
    
    Total Return: {metrics['total_return']:.2f}%
    Annualized Return: {metrics['annualized_return']:.2f}%
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Max Drawdown: {metrics['max_drawdown']:.2f}%
    Volatility: {metrics['volatility']:.2f}%
    
    Final Capital: {metrics['final_capital']:,.0f} KRW
    Initial Capital: {initial_capital:,.0f} KRW
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 차트 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_strategies(
    results: Dict[str, List[float]],
    initial_capital: float,
    save_path: str = None
):
    """
    여러 전략 비교 시각화
    
    Args:
        results: {'전략명': capital_history} 딕셔너리
        initial_capital: 초기 자본
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 누적 수익률 비교
    ax1 = axes[0]
    for name, capital_history in results.items():
        cumulative_returns = [(c / initial_capital - 1) * 100 for c in capital_history]
        ax1.plot(cumulative_returns, linewidth=2, label=name, alpha=0.8)
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('Cumulative Return Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. 성과 지표 비교 (바 차트)
    ax2 = axes[1]
    metrics_df = pd.DataFrame({
        name: calculate_metrics(capital_history, initial_capital)
        for name, capital_history in results.items()
    }).T
    
    metrics_df[['total_return', 'sharpe_ratio']].plot(
        kind='bar', ax=ax2, color=['#2E86AB', '#F18F01'], alpha=0.7
    )
    ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value')
    ax2.set_xlabel('Strategy')
    ax2.legend(['Total Return (%)', 'Sharpe Ratio'])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 비교 차트 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_results(
    metrics: Dict,
    capital_history: List[float],
    output_dir: str = 'results'
):
    """
    학습 결과 저장
    
    Args:
        metrics: 성과 지표
        capital_history: 자산 가치 이력
        output_dir: 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 메트릭 CSV 저장
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # 2. 자본 이력 CSV 저장
    capital_df = pd.DataFrame({
        'day': range(len(capital_history)),
        'capital': capital_history
    })
    capital_path = os.path.join(output_dir, 'capital_history.csv')
    capital_df.to_csv(capital_path, index=False)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   - {metrics_path}")
    print(f"   - {capital_path}")


if __name__ == "__main__":
    # 테스트
    import numpy as np
    
    # Dummy 데이터
    initial = 100_000_000
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, 250)
    capital = [initial]
    for r in daily_returns:
        capital.append(capital[-1] * (1 + r))
    
    # 메트릭 계산
    metrics = calculate_metrics(capital, initial)
    print("📊 성과 지표:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.2f}")
    
    # 시각화
    plot_portfolio_performance(capital, initial, save_path='/tmp/test_performance.png')
