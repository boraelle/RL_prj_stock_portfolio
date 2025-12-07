"""
Main Training Script for RL Portfolio Management (최종 버전)
- 데이터: 2024년 1월 ~ 2025년 11월 (23개월)
- 학습 및 평가: 전체 데이터 사용
- 미래 테스트: 2025년 12월 (제출 후 별도 평가)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# 프로젝트 모듈
from environment import SectorPortfolioEnv
from models import create_ppo_policy_kwargs
from utils import calculate_metrics, plot_portfolio_performance, save_results


def load_data(data_path: str = 'data/etf_data_full.csv'):
    """
    전체 ETF 데이터 로드 (2024.01 ~ 2025.11)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ 데이터 파일이 없습니다: {data_path}\n"
            f"먼저 'python src/collect_data.py'를 실행하세요!"
        )
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✅ 데이터 로드 완료: {len(df):,}개 레코드")
    print(f"   기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print(f"   거래일 수: {df['Date'].nunique()}일 (약 {df['Date'].nunique()/20:.1f}개월)")
    print(f"   종목 수: {df['Ticker'].nunique()}개")
    
    return df


def create_env(df: pd.DataFrame, **kwargs):
    """
    강화학습 환경 생성 (Vectorized)
    """
    def _init():
        return SectorPortfolioEnv(df, **kwargs)
    
    env = DummyVecEnv([_init])
    print(f"✅ 환경 생성 완료")
    print(f"   Observation Space: Dict with {len(env.observation_space.spaces)} modals")
    print(f"   Action Space: {env.action_space.shape}")
    print(f"   Episode Length: {df['Date'].nunique() - kwargs.get('n_history', 20)}일")
    
    return env


def train_agent(
    env,
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    save_dir: str = 'results'
):
    """
    PPO 에이전트 학습
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Policy kwargs (LSTM Feature Extractor)
    policy_kwargs = create_ppo_policy_kwargs(env.observation_space)
    
    # 2. PPO 모델 생성
    print("\n🚀 PPO 모델 생성 중...")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, 'tensorboard')
    )
    
    print(f"✅ 모델 생성 완료")
    print(f"   총 파라미터: {sum(p.numel() for p in model.policy.parameters()):,}개")
    
    # 3. Callbacks 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.join(save_dir, 'checkpoints'),
        name_prefix='ppo_portfolio'
    )
    
    # 4. 학습 시작
    print(f"\n🎓 학습 시작 (총 {total_timesteps:,} timesteps)...")
    print(f"   데이터: 2024.01 ~ 2025.11 (23개월)")
    print(f"   예상 시간: {(total_timesteps / 1000):.0f}분 (Mac CPU 기준)")
    print(f"   TensorBoard: tensorboard --logdir {os.path.join(save_dir, 'tensorboard')}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ 학습이 중단되었습니다. 현재까지 학습된 모델을 저장합니다...")
    
    # 5. 최종 모델 저장
    model_path = os.path.join(save_dir, 'ppo_portfolio_final.zip')
    model.save(model_path)
    print(f"\n✅ 학습 완료! 모델 저장: {model_path}")
    
    return model


def evaluate_agent(model, env, n_eval_episodes: int = 5):
    """
    학습된 에이전트 평가 (전체 데이터)
    """
    print(f"\n📊 에이전트 평가 중 ({n_eval_episodes} 에피소드)...")
    
    all_capital_histories = []
    all_metrics = []
    
    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        capital_history = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # info는 VecEnv에서 리스트로 반환
            if isinstance(info, list):
                info = info[0]
            
            capital_history.append(info['capital'])
            
            if done:
                break
        
        # 메트릭 계산
        initial_capital = capital_history[0] / (1 + info['total_return'])
        metrics = calculate_metrics(capital_history, initial_capital)
        
        all_capital_histories.append(capital_history)
        all_metrics.append(metrics)
        
        print(f"  Episode {ep+1}: Return={metrics['total_return']:.2f}%, "
              f"Sharpe={metrics['sharpe_ratio']:.2f}, MDD={metrics['max_drawdown']:.2f}%")
    
    # 평균 메트릭
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print(f"\n✅ 평가 완료 (평균):")
    print(f"   Total Return: {avg_metrics['total_return']:.2f}%")
    print(f"   Annualized Return: {avg_metrics['annualized_return']:.2f}%")
    print(f"   Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {avg_metrics['max_drawdown']:.2f}%")
    
    return {
        'metrics': avg_metrics,
        'capital_histories': all_capital_histories,
        'all_metrics': all_metrics
    }


def baseline_strategy(env, strategy='equal_weight'):
    """
    베이스라인 전략 평가 (비교용)
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
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    while not done:
        if strategy == 'buy_and_hold' and not rebalance:
            action = env.get_attr('portfolio_weights')[0]
        
        obs, reward, done, info = env.step([action])
        
        if isinstance(info, list):
            info = info[0]
        
        capital_history.append(info['capital'])
        
        if strategy == 'buy_and_hold':
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
    메인 실행 함수
    """
    print("=" * 70)
    print("🎯 RL Portfolio Management Training")
    print("   Algorithm: PPO + LSTM")
    print("   Assets: 14 ETFs + Cash")
    print("   Period: 2024년 1월 ~ 2025년 11월 (23개월)")
    print("=" * 70)
    
    # 1. 데이터 로드
    df = load_data('data/etf_data_full.csv')
    
    # 2. 환경 생성
    env = create_env(
        df,
        initial_capital=100_000_000,  # 1억원
        transaction_cost=0.0015,      # 0.15%
        n_history=20,
        cash_return=0.02,
        rebalance_freq=1
    )
    
    # 3. 에이전트 학습
    model = train_agent(
        env,
        total_timesteps=150_000,  # 23개월 데이터이므로 증가
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        save_dir='results'
    )
    
    # 4. 평가
    eval_results = evaluate_agent(model, env, n_eval_episodes=3)
    
    # 5. 베이스라인 비교
    baseline_equal = baseline_strategy(env, strategy='equal_weight')
    baseline_bh = baseline_strategy(env, strategy='buy_and_hold')
    
    # 6. 시각화
    print("\n📊 결과 시각화 중...")
    
    initial_capital = 100_000_000
    best_capital = eval_results['capital_histories'][0]
    
    plot_portfolio_performance(
        best_capital,
        initial_capital,
        save_path='results/rl_performance.png'
    )
    
    # 전략 비교
    from utils import compare_strategies
    compare_strategies(
        {
            'RL Agent (PPO+LSTM)': best_capital,
            'Equal Weight': baseline_equal,
            'Buy & Hold': baseline_bh
        },
        initial_capital,
        save_path='results/strategy_comparison.png'
    )
    
    # 7. 결과 저장
    save_results(
        eval_results['metrics'],
        best_capital,
        output_dir='results'
    )
    
    print("\n" + "=" * 70)
    print("✅ 모든 작업 완료!")
    print("=" * 70)
    print(f"\n📁 결과물:")
    print(f"   - 모델: results/ppo_portfolio_final.zip")
    print(f"   - 성과 차트: results/rl_performance.png")
    print(f"   - 전략 비교: results/strategy_comparison.png")
    print(f"   - 메트릭: results/metrics.csv")
    print(f"   - 자본 이력: results/capital_history.csv")
    
    print(f"\n💡 TensorBoard 확인:")
    print(f"   tensorboard --logdir results/tensorboard")
    
    print(f"\n🔮 미래 테스트 (제출 후):")
    print(f"   2025년 12월 데이터로 최종 검증:")
    print(f"   1. python src/collect_december.py (12월 후)")
    print(f"   2. python src/evaluate_december.py")
    
    print(f"\n📊 프로젝트 성과 요약:")
    print(f"   기간: 2024.01 ~ 2025.11 (23개월)")
    print(f"   총 수익률: {eval_results['metrics']['total_return']:.2f}%")
    print(f"   연환산 수익률: {eval_results['metrics']['annualized_return']:.2f}%")
    print(f"   Sharpe Ratio: {eval_results['metrics']['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {eval_results['metrics']['max_drawdown']:.2f}%")
    

if __name__ == "__main__":
    main()
