"""
ETF 데이터 수집 스크립트 (최종 확정 버전)
- TIGER 섹터 ETF 8개 (통신/유틸리티 제외)
- KODEX 특수산업 ETF 4개
- 현금 1개
- 총 13자산 포트폴리오
"""

import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime
import os

# 13자산 정의 - TIGER 8개 + KODEX 4개 + 현금 1개
ASSETS = {
    # === TIGER 섹터 ETF (8개) - 핵심 섹터만 ===
    'TIGER_200_Energy': '139250',      # 에너지화학
    'TIGER_200_Materials': '252670',   # 소재
    'TIGER_200_Industrials': '139270', # 산업재
    'TIGER_200_Consumer_Staples': '252710',  # 필수소비재
    'TIGER_200_Consumer_Discretionary': '252720',  # 자유소비재
    'TIGER_200_Healthcare': '252730',  # 헬스케어
    'TIGER_200_Financials': '139240',  # 금융
    'TIGER_200_IT': '139260',          # IT
    
    # === KODEX 특수산업 ETF (4개) - 한국 대표 산업 ===
    'KODEX_Semiconductor': '091160',   # 반도체
    'KODEX_Battery': '305720',         # 2차전지
    'KODEX_Defense': '367380',         # 방산
    'KODEX_Bio': '244580',             # 바이오
}

# 현금은 코드에서 자동 추가 (13번째 자산, 연 2% 수익률)


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
    print(f"   TIGER 섹터 ETF: 8개 (에너지, 소재, 산업재, 소비재, 헬스케어, 금융, IT)")
    print(f"   KODEX 특수산업 ETF: 4개 (반도체, 2차전지, 방산, 바이오)")
    print(f"   현금: 1개 (코드에서 추가)")
    print(f"   총 13자산 포트폴리오\n")
    
    all_data = []
    success_count = 0
    failed_tickers = []
    
    for name, ticker in ASSETS.items():
        print(f"  - {name:40s} ({ticker}) ...", end=' ')
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            
            # 필요한 컬럼만 선택 (OHLCV)
            if 'Close' in df.columns and len(df) > 0:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df['Ticker'] = name
                df['Code'] = ticker
                df.reset_index(inplace=True)  # Date를 컬럼으로
                all_data.append(df)
                success_count += 1
                print(f"✅ {len(df)}일")
            else:
                print(f"⚠️ 데이터 없음")
                failed_tickers.append(name)
        except Exception as e:
            error_msg = str(e)[:40]
            print(f"❌ 실패 - {error_msg}")
            failed_tickers.append(name)
    
    if not all_data:
        raise ValueError("❌ 수집된 데이터가 없습니다!")
    
    # 결과 요약
    print(f"\n{'='*70}")
    print(f"✅ 수집 완료: {success_count}개 성공 / {len(ASSETS)}개 시도")
    if failed_tickers:
        print(f"⚠️  실패 종목: {', '.join(failed_tickers)}")
        print(f"   → 포트폴리오에서 제외됩니다.")
    else:
        print(f"🎉 모든 ETF 데이터 수집 성공!")
    print(f"{'='*70}\n")
    
    # 전체 데이터프레임 결합
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Date를 datetime 형식으로 변환
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # 날짜별로 정렬
    combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    print(f"📊 데이터 요약:")
    print(f"   - 기간: {combined_df['Date'].min().date()} ~ {combined_df['Date'].max().date()}")
    print(f"   - 총 데이터 포인트: {len(combined_df):,}개")
    print(f"   - 성공 종목 수: {combined_df['Ticker'].nunique()}개 ETF + 1개 현금 = {combined_df['Ticker'].nunique() + 1}자산")
    print(f"   - 거래일 수: {combined_df['Date'].nunique()}일")
    
    return combined_df


def split_train_val(df, val_ratio=0.2):
    """
    Train/Validation 데이터 분할 (시계열 순서 유지)
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
    print("🎯 ETF 데이터 수집 (최종 확정)")
    print("   기간: 2024년 1월 ~ 2025년 11월 (23개월)")
    print("   구성: TIGER 섹터 8개 + KODEX 특수산업 4개 + 현금 1개")
    print("=" * 70)
    print()
    
    # 데이터 수집 (2024.01 ~ 2025.11)
    df = collect_etf_data(start_date='2024-01-01', end_date='2025-11-30')
    
    # Train/Val 분할 (80:20)
    train_df, val_df = split_train_val(df, val_ratio=0.2)
    
    # 데이터 저장
    save_data(df, train_df, val_df)
    
    # 간단한 통계
    print("\n📈 수집된 ETF 통계:")
    stats = df.groupby('Ticker')['Close'].describe()[['count', 'mean', 'std', 'min', 'max']]
    print(stats.to_string())
    
    print("\n" + "=" * 70)
    print("✅ 데이터 수집 완료!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("   python src/train.py")
    print("\n📊 포트폴리오 구성:")
    print("   - TIGER 8개: 에너지, 소재, 산업재, 소비재, 헬스케어, 금융, IT")
    print("   - KODEX 4개: 반도체, 2차전지, 방산, 바이오")
    print("   - 현금 1개: 연 2% 고정 수익률")
