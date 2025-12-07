"""
ETF 데이터 수집 스크립트 (수정 버전 - TIGER ETF 사용)
- 학습/평가: 2024년 1월 ~ 2025년 11월 (23개월, 약 460일)
- 미래 테스트: 2025년 12월 (나중에 별도 수집)
"""

import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime
import os

# 15자산 정의 (14 ETFs + 현금) - TIGER ETF 위주로 변경
ASSETS = {
    # TIGER 섹터 ETF (더 안정적)
    'TIGER_200_Energy': '139250',      # TIGER 200 에너지화학
    'TIGER_200_Materials': '139260',   # TIGER 200 IT (소재 대신 IT로)
    'TIGER_200_Industrials': '139270', # TIGER 200 산업재
    'TIGER_200_Financials': '139240',  # TIGER 200 금융
    'TIGER_200_IT': '139260',          # TIGER 200 IT
    'TIGER_Consumer': '143850',        # TIGER 소비재
    'TIGER_Healthcare': '143860',      # TIGER 헬스케어
    
    # KODEX 특수산업 ETF (잘 작동하는 것들)
    'KODEX_Semiconductor': '091160',   # KODEX 반도체
    'KODEX_Battery': '305720',         # KODEX 2차전지
    'KODEX_KoGames': '307510',         # KODEX 게임
    'KODEX_Defense': '367380',         # KODEX 방산
    
    # 추가 ETF
    'TIGER_KOSPI200': '102110',        # TIGER KOSPI200
    'KODEX_200': '069500',             # KODEX 200 (시장 대표)
}


def collect_etf_data(start_date, end_date):
    """
    ETF 일봉 데이터 수집
    
    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: 전체 OHLCV 데이터
    """
    print(f"📊 ETF 데이터 수집 시작 ({start_date} ~ {end_date})...")
    print(f"   기간: 23개월 (약 460 거래일)")
    
    all_data = []
    failed_tickers = []
    
    for name, ticker in ASSETS.items():
        print(f"  - {name} ({ticker}) 수집 중...", end=' ')
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            
            # 필요한 컬럼만 선택 (OHLCV)
            if 'Close' in df.columns and len(df) > 0:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df['Ticker'] = name
                df['Code'] = ticker
                df.reset_index(inplace=True)  # Date를 컬럼으로
                all_data.append(df)
                print(f"✅ {len(df)}일")
            else:
                print(f"⚠️ 데이터 없음")
                failed_tickers.append(name)
        except Exception as e:
            print(f"❌ 실패 - {str(e)[:50]}")
            failed_tickers.append(name)
    
    if not all_data:
        raise ValueError("수집된 데이터가 없습니다!")
    
    if failed_tickers:
        print(f"\n⚠️  일부 종목 수집 실패: {', '.join(failed_tickers)}")
        print(f"   계속 진행하지만 포트폴리오에서 제외됩니다.")
    
    # 전체 데이터프레임 결합
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Date를 datetime 형식으로 변환
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # 날짜별로 정렬
    combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    print(f"\n✅ 수집 완료!")
    print(f"   - 기간: {combined_df['Date'].min().date()} ~ {combined_df['Date'].max().date()}")
    print(f"   - 총 데이터 포인트: {len(combined_df):,}개")
    print(f"   - 성공 종목 수: {combined_df['Ticker'].nunique()}개 / {len(ASSETS)}개")
    print(f"   - 거래일 수: {combined_df['Date'].nunique()}일")
    
    return combined_df


def split_train_val(df, val_ratio=0.2):
    """
    Train/Validation 데이터 분할 (시계열 순서 유지)
    
    Args:
        df: 전체 데이터프레임
        val_ratio: Validation 비율 (기본 20%)
    
    Returns:
        train_df, val_df
    """
    dates = sorted(df['Date'].unique())
    n_dates = len(dates)
    
    # 시계열 순서대로 split (뒤쪽 20%를 validation)
    split_idx = int(n_dates * (1 - val_ratio))
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]
    
    train_df = df[df['Date'].isin(train_dates)].copy()
    val_df = df[df['Date'].isin(val_dates)].copy()
    
    print(f"\n📂 데이터 분할 (시계열 순서 유지):")
    print(f"   [Train] {train_df['Date'].min().date()} ~ {train_df['Date'].max().date()} "
          f"({len(train_dates)}일, {len(train_df):,}개)")
    print(f"   [Val]   {val_df['Date'].min().date()} ~ {val_df['Date'].max().date()} "
          f"({len(val_dates)}일, {len(val_df):,}개)")
    
    return train_df, val_df


def save_data(df, train_df=None, val_df=None, output_dir='data'):
    """
    데이터를 CSV로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 전체 데이터 저장 (학습용)
    full_path = os.path.join(output_dir, 'etf_data_full.csv')
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 전체 데이터 저장: {full_path}")
    print(f"   파일 크기: {os.path.getsize(full_path) / 1024:.1f} KB")
    
    # Train/Val 분할 데이터 저장 (선택적)
    if train_df is not None and val_df is not None:
        train_path = os.path.join(output_dir, 'etf_data_train.csv')
        val_path = os.path.join(output_dir, 'etf_data_val.csv')
        
        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
        
        print(f"💾 Train 데이터 저장: {train_path} ({os.path.getsize(train_path) / 1024:.1f} KB)")
        print(f"💾 Val 데이터 저장: {val_path} ({os.path.getsize(val_path) / 1024:.1f} KB)")
    
    # 미래 테스트용 안내
    print(f"\n🔮 미래 테스트 (제출 후):")
    print(f"   2025년 12월 데이터는 나중에 별도 수집하여 평가")
    print(f"   → python src/collect_december.py (12월 후)")
    
    return full_path


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 ETF 데이터 수집 (TIGER + KODEX)")
    print("   기간: 2024년 1월 ~ 2025년 11월 (23개월)")
    print("   용도: 학습 및 평가 (프로젝트 제출용)")
    print("=" * 70)
    
    # 데이터 수집 (2024.01 ~ 2025.11)
    df = collect_etf_data(start_date='2024-01-01', end_date='2025-11-30')
    
    # Train/Val 분할 (선택적, 80:20)
    train_df, val_df = split_train_val(df, val_ratio=0.2)
    
    # 데이터 저장
    save_data(df, train_df, val_df)
    
    # 간단한 통계
    print("\n📈 전체 데이터 통계:")
    stats = df.groupby('Ticker')['Close'].describe()[['count', 'mean', 'std', 'min', 'max']]
    print(stats.to_string())
    
    print("\n" + "=" * 70)
    print("✅ 데이터 수집 완료!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("   1. 학습: python src/train.py")
    print("   2. 평가: 학습 과정에서 자동 수행")
    print("   3. 12월 테스트: python src/evaluate_december.py (제출 후)")
