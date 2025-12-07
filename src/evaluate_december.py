"""
2025년 12월 데이터 평가 스크립트 (제출 후 사용)
학습된 모델을 미래 데이터(12월)에서 검증
"""

import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import SectorPortfolioEnv
from utils import calculate_metrics, plot_portfolio_performance, compare_strategies


def load_december_data(data_path: str = 'data/etf_data_december.csv'):
    """
    12월 데이터 로드
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ 12월 데이터가 없습니다: {data_path}\n\n"
            f"💡 해결 방법:\n"
            f"   1. 12월이 끝난 후 데이터를 수집하세요:\n"
            f"      python src/collect_december.py\n\n"
            f"   이 평가는 프로젝트 제출 후 추가로 수행하는 것입니다."
        )
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ 12월 데이터 로드 완료: {len(df):,}개 레코드")
    print(f"   기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print(f"   거래일 수: {df['Date'].nunique()}일")
    print(f"   종목 수: {df['Ticker'].nunique()}개")
    
    # 최소 거래일 체크
    if df['Date'].nunique() < 20:
        print(f"\n⚠️ 경고: 거래일이 {df['Date'].nunique()}일로 부족합니다.")
        print(f"   환경의 n_history=20이므로 최소 20일 필요합니다.")
        print(f"   평가를 계속 진행하지만 결과가 부정확할 수 있습니다.")
    
    return df


def load_trained_model(model_path: str = 'results/ppo_portfolio_final.zip'):
    """
    학습된 모델 로드
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ 학습된 모델이 없습니다: {model_path}\n"
            f"먼저 학습을 실행하세요: python src/train.py"
        )
    
    model = PPO.load(model_path)
    print(f"✅ 모델 로드 완료: {model_path}")
    
    return model


def create_test_env(df: pd.DataFrame, **kwargs):
    """
    Test 환경 생성
    """
    def _init():
        return SectorPortfolioEnv(df, **kwargs)
    
    return DummyVecEnv([_init])


def evaluate_on_december(model, env):
    """
    12월 데이터에서 모델 평가
    """
    print(f"\n📊 12월 데이터 평가 중...")
    
    obs = env.reset()
    done = False
    capital_history = []
    actions_history = []
    
    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if isinstance(info, list):
            info = info[0]
        
        capital_history.append(info['capital'])
        actions_history.append(info['portfolio'])
        
        step += 1
        if done:
            break
    
    # 메트릭 계산
    initial_capital = capital_history[0] / (1 + info['total_return'])
    metrics = calculate_metrics(capital_history, initial_capital)
    
    print(f"\n  ✅ 평가 완료:")
    print(f"    총 스텝: {step}일")
    print(f"    최종 자본: {info['capital']:,.0f}원")
    print(f"    총 수익률: {metrics['total_return']:.2f}%")
    print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"    변동성: {metrics['volatility']:.2f}%")
    
    return {
        'metrics': metrics,
        'capital_history': capital_history,
        'actions_history': actions_history
    }


def baseline_on_december(env, strategy='equal_weight'):
    """
    12월 데이터에서 베이스라인 전략 평가
    """
    print(f"\n🔍 베이스라인 전략 평가: {strategy}")
    
    obs = env.reset()
    done = False
    capital_history = []
    
    n_assets = env.action_space.shape[0]
    
    if strategy == 'equal_weight':
        action = np.ones(n_assets) / n_assets
    elif strategy == 'buy_and_hold':
        action = np.ones(n_assets) / n_assets
        rebalance = True
    
    while not done:
        if strategy == 'buy_and_hold' and 'rebalance' in locals() and not rebalance:
            action = env.get_attr('portfolio_weights')[0]
        
        obs, reward, done, info = env.step([action])
        
        if isinstance(info, list):
            info = info[0]
        
        capital_history.append(info['capital'])
        
        if strategy == 'buy_and_hold' and 'rebalance' in locals():
            rebalance = False
        
        if done:
            break
    
    # 메트릭
    initial_capital = capital_history[0] / (1 + info['total_return'])
    metrics = calculate_metrics(capital_history, initial_capital)
    
    print(f"  Return: {metrics['total_return']:.2f}%, "
          f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
          f"MDD: {metrics['max_drawdown']:.2f}%")
    
    return capital_history


def main():
    """
    12월 평가 메인 함수
    """
    print("=" * 70)
    print("🔮 미래 성능 검증: 2025년 12월 평가")
    print("   (프로젝트 제출 후 추가 검증용)")
    print("=" * 70)
    
    # 1. 12월 데이터 로드
    try:
        df_dec = load_december_data('data/etf_data_december.csv')
    except FileNotFoundError as e:
        print(f"\n{e}")
        return
    
    # 2. 학습된 모델 로드
    try:
        model = load_trained_model('results/ppo_portfolio_final.zip')
    except FileNotFoundError as e:
        print(f"\n{e}")
        return
    
    # 3. Test 환경 생성
    env_dec = create_test_env(
        df_dec,
        initial_capital=100_000_000,
        transaction_cost=0.0015,
        n_history=20,
        cash_return=0.02,
        rebalance_freq=1
    )
    
    # 4. 12월 데이터 평가
    dec_results = evaluate_on_december(model, env_dec)
    
    # 5. 베이스라인 비교
    baseline_equal = baseline_on_december(env_dec, strategy='equal_weight')
    baseline_bh = baseline_on_december(env_dec, strategy='buy_and_hold')
    
    # 6. 시각화
    print("\n📊 결과 시각화 중...")
    
    initial_capital = 100_000_000
    rl_capital = dec_results['capital_history']
    
    # RL 성과
    plot_portfolio_performance(
        rl_capital,
        initial_capital,
        save_path='results/rl_performance_december.png'
    )
    
    # 전략 비교
    compare_strategies(
        {
            'RL Agent (Dec)': rl_capital,
            'Equal Weight (Dec)': baseline_equal,
            'Buy & Hold (Dec)': baseline_bh
        },
        initial_capital,
        save_path='results/strategy_comparison_december.png'
    )
    
    # 7. 결과 저장
    from utils import save_results
    save_results(
        dec_results['metrics'],
        rl_capital,
        output_dir='results/december_results'
    )
    
    # 8. 최종 요약
    print("\n" + "=" * 70)
    print("✅ 12월 평가 완료!")
    print("=" * 70)
    
    print(f"\n📊 12월 미래 성능 (학습에 사용되지 않은 데이터):")
    print(f"   [RL Agent]")
    print(f"     - 총 수익률: {dec_results['metrics']['total_return']:.2f}%")
    print(f"     - Sharpe Ratio: {dec_results['metrics']['sharpe_ratio']:.2f}")
    print(f"     - Max Drawdown: {dec_results['metrics']['max_drawdown']:.2f}%")
    
    print(f"\n📁 결과 파일:")
    print(f"   - 성과 차트: results/rl_performance_december.png")
    print(f"   - 전략 비교: results/strategy_comparison_december.png")
    print(f"   - 메트릭: results/december_results/metrics.csv")
    
    print(f"\n💡 활용 방법:")
    print(f"   1. GitHub README에 12월 성과 추가")
    print(f"   2. 면접 시 '미래 데이터 검증' 어필")
    print(f"   3. 2024.01~2025.11 vs 2025.12 성능 비교 분석")


if __name__ == "__main__":
    main()
