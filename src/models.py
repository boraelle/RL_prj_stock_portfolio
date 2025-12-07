"""
LSTM-based Multi-Modal Policy Network
PPO와 연동되는 커스텀 Feature Extractor
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class LSTMMultiModalExtractor(BaseFeaturesExtractor):
    """
    Multi-Modal State를 처리하는 Feature Extractor
    
    Architecture:
        1. Price History Encoder: LSTM(2 layers, 64 hidden)
        2. Indicator Encoder: Dense(32 hidden)
        3. Portfolio Encoder: Dense(16 hidden)
        4. Market Encoder: Dense(16 hidden)
        5. Fusion Layer: Concat → Dense(256 → 128)
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        # features_dim은 최종 출력 차원 (PPO policy에 전달)
        super().__init__(observation_space, features_dim)
        
        # observation_space에서 각 modal의 shape 추출
        price_shape = observation_space['price_history'].shape  # (n_assets, n_history, 5)
        indicator_shape = observation_space['indicators'].shape  # (n_assets, 6)
        portfolio_shape = observation_space['portfolio'].shape  # (3,)
        market_shape = observation_space['market'].shape  # (2,)
        
        self.n_assets = price_shape[0]
        self.n_history = price_shape[1]
        
        # 1. Price Encoder - LSTM (각 자산별로 독립 처리 후 평균)
        self.price_lstm = nn.LSTM(
            input_size=5,  # OHLCV
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 2. Indicator Encoder
        indicator_dim = self.n_assets * indicator_shape[-1]
        self.indicator_encoder = nn.Sequential(
            nn.Linear(indicator_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # 3. Portfolio Encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_shape[0], 16),
            nn.ReLU()
        )
        
        # 4. Market Encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(market_shape[0], 16),
            nn.ReLU()
        )
        
        # 5. Fusion Layer
        # LSTM(64) + Indicator(32) + Portfolio(16) + Market(16) = 128
        fusion_input_dim = 64 + 32 + 16 + 16
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Multi-modal state를 하나의 feature vector로 인코딩
        
        Args:
            observations: Dict with keys ['price_history', 'indicators', 'portfolio', 'market']
        
        Returns:
            features: (batch_size, features_dim) Tensor
        """
        # 각 modal 추출
        price = observations['price_history']  # (batch, n_assets, n_history, 5)
        indicators = observations['indicators']  # (batch, n_assets, 6)
        portfolio = observations['portfolio']  # (batch, 3)
        market = observations['market']  # (batch, 2)
        
        batch_size = price.shape[0]
        
        # 1. Price LSTM Encoding
        # 각 자산별로 LSTM 통과시킨 후 평균
        price_features = []
        for i in range(self.n_assets):
            asset_price = price[:, i, :, :]  # (batch, n_history, 5)
            lstm_out, (h_n, c_n) = self.price_lstm(asset_price)
            # 마지막 hidden state 사용
            last_hidden = lstm_out[:, -1, :]  # (batch, 64)
            price_features.append(last_hidden)
        
        # 자산별 feature 평균 (간단한 aggregation)
        price_feat = torch.stack(price_features, dim=1).mean(dim=1)  # (batch, 64)
        
        # 2. Indicator Encoding
        indicator_flat = indicators.reshape(batch_size, -1)  # (batch, n_assets*6)
        indicator_feat = self.indicator_encoder(indicator_flat)  # (batch, 32)
        
        # 3. Portfolio Encoding
        portfolio_feat = self.portfolio_encoder(portfolio)  # (batch, 16)
        
        # 4. Market Encoding
        market_feat = self.market_encoder(market)  # (batch, 16)
        
        # 5. Fusion
        combined = torch.cat([
            price_feat,
            indicator_feat,
            portfolio_feat,
            market_feat
        ], dim=-1)  # (batch, 128)
        
        features = self.fusion(combined)  # (batch, features_dim=128)
        
        return features


def create_ppo_policy_kwargs(observation_space: spaces.Dict):
    """
    PPO 모델 생성시 사용할 policy_kwargs 생성
    
    Returns:
        dict: Stable-Baselines3 PPO의 policy_kwargs
    """
    policy_kwargs = dict(
        features_extractor_class=LSTMMultiModalExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 128],  # Actor/Critic 공유 레이어
        activation_fn=nn.ReLU
    )
    
    return policy_kwargs


if __name__ == "__main__":
    # 간단한 테스트
    from gymnasium import spaces
    
    # Dummy observation space
    obs_space = spaces.Dict({
        'price_history': spaces.Box(low=0, high=np.inf, shape=(14, 20, 5), dtype=np.float32),
        'indicators': spaces.Box(low=-np.inf, high=np.inf, shape=(14, 6), dtype=np.float32),
        'portfolio': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        'market': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    })
    
    # Feature Extractor 생성
    extractor = LSTMMultiModalExtractor(obs_space, features_dim=128)
    
    # Dummy 입력
    dummy_obs = {
        'price_history': torch.randn(4, 14, 20, 5),  # batch=4
        'indicators': torch.randn(4, 14, 6),
        'portfolio': torch.randn(4, 3),
        'market': torch.randn(4, 2)
    }
    
    # Forward pass
    features = extractor(dummy_obs)
    print(f"✅ Feature Extractor 테스트 성공!")
    print(f"   Input shapes: {[(k, v.shape) for k, v in dummy_obs.items()]}")
    print(f"   Output shape: {features.shape}")  # Should be (4, 128)
    print(f"   Model parameters: {sum(p.numel() for p in extractor.parameters()):,}")
