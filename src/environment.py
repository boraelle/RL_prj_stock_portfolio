"""
Multi-Modal Portfolio Gym Environment
15개 자산 (14 ETF + 1 Cash) 포트폴리오 리밸런싱 환경
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class SectorPortfolioEnv(gym.Env):
    """
    포트폴리오 리밸런싱 강화학습 환경
    
    State (Multi-modal):
        - price_history: (n_assets, n_history, 5) - OHLCV 20일치
        - indicators: (n_assets, 6) - MA5, MA20, RSI, MACD, BB_upper, Volume_ratio
        - portfolio: (3,) - [현금비율, 보유종목수, MDD]
        - market: (2,) - [KOSPI200지수변화율, 시장변동성]
    
    Action: (n_assets,) - 각 자산 목표 비중 (0~1, 합=1)
    Reward: 일간 수익률 - 거래비용(0.15%)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000_000,  # 1억원
        transaction_cost: float = 0.0015,      # 0.15%
        n_history: int = 20,                   # 과거 20일 사용
        cash_return: float = 0.02,             # 연 2% 현금 수익률
        rebalance_freq: int = 1                # 매일 리밸런싱 (1=일봉)
    ):
        super().__init__()
        
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.n_history = n_history
        self.daily_cash_return = (1 + cash_return) ** (1/252) - 1  # 일간 수익률 변환
        self.rebalance_freq = rebalance_freq
        
        # 자산 리스트 (14 ETF + Cash)
        self.tickers = sorted(self.df['Ticker'].unique())
        self.n_assets = len(self.tickers) + 1  # +1 for cash
        
        # 날짜 인덱스 생성
        self.dates = sorted(self.df['Date'].unique())
        self.max_steps = len(self.dates) - n_history - 1
        
        # State/Action Space 정의
        self.observation_space = spaces.Dict({
            'price_history': spaces.Box(
                low=0, high=np.inf, 
                shape=(self.n_assets-1, n_history, 5),  # Cash 제외
                dtype=np.float32
            ),
            'indicators': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_assets-1, 6),  # Cash 제외
                dtype=np.float32
            ),
            'portfolio': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
            'market': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(2,),
                dtype=np.float32
            )
        })
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # 내부 상태 초기화
        self.reset()
    
    def reset(self, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.portfolio_weights = np.zeros(self.n_assets)
        self.portfolio_weights[-1] = 1.0  # 처음엔 100% 현금
        
        # 성과 추적
        self.total_return = 0.0
        self.max_capital = self.initial_capital
        self.mdd = 0.0
        self.trade_count = 0
        
        # 이력
        self.capital_history = [self.initial_capital]
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        1스텝 진행
        action: (n_assets,) 목표 포트폴리오 비중
        """
        # 1. Action 정규화 (합 = 1.0)
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # 2. 리밸런싱 비용 계산
        weight_change = np.abs(action - self.portfolio_weights).sum()
        trading_cost = weight_change * self.transaction_cost * self.current_capital
        
        # 3. 포트폴리오 업데이트
        old_weights = self.portfolio_weights.copy()
        self.portfolio_weights = action
        
        # 4. 다음 스텝으로 이동
        self.current_step += 1
        
        # 5. 일간 수익률 계산
        returns = self._calculate_daily_returns()
        
        # 6. 자산별 수익 적용
        portfolio_return = 0.0
        for i, ret in enumerate(returns):
            portfolio_return += self.portfolio_weights[i] * ret
        
        # 7. 자본 업데이트
        self.current_capital = self.current_capital * (1 + portfolio_return) - trading_cost
        self.capital_history.append(self.current_capital)
        
        # 8. MDD 계산
        self.max_capital = max(self.max_capital, self.current_capital)
        current_dd = (self.current_capital - self.max_capital) / self.max_capital
        self.mdd = min(self.mdd, current_dd)
        
        # 9. Reward 계산 (일간 수익률 - 거래비용)
        reward = portfolio_return - (trading_cost / self.current_capital)
        
        # 10. 거래 카운트
        if weight_change > 0.01:  # 1% 이상 변경시에만 거래로 간주
            self.trade_count += 1
        
        # 11. 종료 조건
        terminated = (self.current_step >= self.max_steps) or (self.current_capital <= 0)
        truncated = False
        
        # 12. 관찰
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_daily_returns(self) -> np.ndarray:
        """
        각 자산의 일간 수익률 계산
        Returns: (n_assets,) - 마지막은 현금(고정 수익률)
        """
        current_date = self.dates[self.current_step + self.n_history]
        prev_date = self.dates[self.current_step + self.n_history - 1]
        
        returns = []
        
        for ticker in self.tickers:
            current_price = self.df[
                (self.df['Date'] == current_date) & (self.df['Ticker'] == ticker)
            ]['Close'].values
            
            prev_price = self.df[
                (self.df['Date'] == prev_date) & (self.df['Ticker'] == ticker)
            ]['Close'].values
            
            if len(current_price) > 0 and len(prev_price) > 0:
                ret = (current_price[0] - prev_price[0]) / prev_price[0]
            else:
                ret = 0.0  # 데이터 없으면 0% 수익
            
            returns.append(ret)
        
        # 현금 수익률 추가
        returns.append(self.daily_cash_return)
        
        return np.array(returns, dtype=np.float32)
    
    def _get_observation(self) -> Dict:
        """
        Multi-modal State 생성
        """
        current_idx = self.current_step + self.n_history
        start_idx = self.current_step
        
        # 1. Price History (20일 OHLCV)
        price_history = []
        for ticker in self.tickers:
            ticker_data = self.df[
                (self.df['Ticker'] == ticker) &
                (self.df['Date'].isin(self.dates[start_idx:current_idx]))
            ][['Open', 'High', 'Low', 'Close', 'Volume']].values
            
            if len(ticker_data) < self.n_history:
                # 데이터 부족시 제로 패딩
                padding = np.zeros((self.n_history - len(ticker_data), 5))
                ticker_data = np.vstack([padding, ticker_data])
            
            price_history.append(ticker_data[-self.n_history:])
        
        price_history = np.array(price_history, dtype=np.float32)
        
        # 2. Technical Indicators (각 자산별 6개 지표)
        indicators = []
        for ticker in self.tickers:
            ticker_df = self.df[
                (self.df['Ticker'] == ticker) &
                (self.df['Date'].isin(self.dates[:current_idx+1]))
            ].copy()
            
            if len(ticker_df) >= self.n_history:
                close = ticker_df['Close'].values
                volume = ticker_df['Volume'].values
                
                # MA5, MA20
                ma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
                ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
                
                # RSI (간단 버전)
                if len(close) >= 14:
                    delta = np.diff(close[-15:])
                    gain = np.mean(np.maximum(delta, 0))
                    loss = np.mean(np.abs(np.minimum(delta, 0)))
                    rsi = 100 - (100 / (1 + (gain / (loss + 1e-8))))
                else:
                    rsi = 50.0
                
                # MACD (간단 버전)
                if len(close) >= 26:
                    ema12 = np.mean(close[-12:])
                    ema26 = np.mean(close[-26:])
                    macd = ema12 - ema26
                else:
                    macd = 0.0
                
                # Bollinger Bands
                if len(close) >= 20:
                    bb_mid = np.mean(close[-20:])
                    bb_std = np.std(close[-20:])
                    bb_upper = bb_mid + 2 * bb_std
                else:
                    bb_upper = close[-1]
                
                # Volume Ratio
                if len(volume) >= 20:
                    vol_ratio = volume[-1] / (np.mean(volume[-20:]) + 1e-8)
                else:
                    vol_ratio = 1.0
                
                ind = [ma5, ma20, rsi, macd, bb_upper, vol_ratio]
            else:
                ind = [0.0] * 6
            
            indicators.append(ind)
        
        indicators = np.array(indicators, dtype=np.float32)
        
        # 3. Portfolio State
        cash_ratio = self.portfolio_weights[-1]
        n_holdings = np.sum(self.portfolio_weights[:-1] > 0.01)
        portfolio_state = np.array([cash_ratio, n_holdings, self.mdd], dtype=np.float32)
        
        # 4. Market Context (KOSPI 200 IT를 시장 대리 지수로 사용)
        market_ticker = 'KODEX_200_IT'
        market_data = self.df[
            (self.df['Ticker'] == market_ticker) &
            (self.df['Date'].isin(self.dates[max(0, start_idx-20):current_idx+1]))
        ]['Close'].values
        
        if len(market_data) >= 2:
            market_return = (market_data[-1] - market_data[-2]) / market_data[-2]
        else:
            market_return = 0.0
        
        if len(market_data) >= 20:
            market_volatility = np.std(np.diff(market_data[-20:]) / market_data[-21:-1])
        else:
            market_volatility = 0.0
        
        market_state = np.array([market_return, market_volatility], dtype=np.float32)
        
        return {
            'price_history': price_history,
            'indicators': indicators,
            'portfolio': portfolio_state,
            'market': market_state
        }
    
    def _get_info(self) -> Dict:
        """추가 정보"""
        return {
            'step': self.current_step,
            'capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital) - 1,
            'mdd': self.mdd,
            'trades': self.trade_count,
            'portfolio': self.portfolio_weights.copy()
        }
    
    def render(self, mode='human'):
        """간단한 상태 출력"""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps} | "
                  f"Capital: {self.current_capital:,.0f}원 | "
                  f"Return: {((self.current_capital/self.initial_capital-1)*100):.2f}% | "
                  f"MDD: {self.mdd*100:.2f}%")
