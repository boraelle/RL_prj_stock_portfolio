"""
2025년 12월 데이터 수집 스크립트 (제출 후 사용)
프로젝트 제출 후 미래 성능 검증용
"""

import pandas as pd
import FinanceDataReader as fdr
import os

# 동일한 15자산
ASSETS = {
    # KOSPI 200 섹터 ETF (10개)
    'KODEX_200_Energy': '140700',
    'KODEX_200_Materials': '140710',
    'KODEX_200_Industrials': '140720',
    'KODEX_200_Consumer_Discretionary': '140730',
    'KODEX_200_Consumer_Staples': '140740',
    'KODEX_200_Healthcare': '140780',
    'KODEX_200_Financials': '140760',
    'KODEX_200_IT': '140770',
    'KODEX_200_Telecom': '140750',
    'KODEX_200_Utilities': '140790',
    
    # 특수산업 ETF (4개)
    'KODEX_Semiconductor': '091160',
    'KODEX_Battery': '305720',
    'KODEX_KoGames': '307510',
    'KODEX_Defense': '367380',
}


def collect_december_data():
    """
    2025년 12월 데이터 수집
    """
    print("=" * 70)
    print("🔮 2025년 12월 데이터 수집 (미래 검증용)")
    print("=" * 70)
    print(f"\n📊 ETF 데이터 수집 시작 (2025-12-01 ~ 2025-12-31)...")
    
    all_data = []
    failed_tickers = []
    
    for name, ticker in ASSETS.items():
        print(f"  - {name} ({ticker}) 수집 중...", end=' ')
        try:
            df = fdr.DataReader(ticker, '2025-12-01', '2025-12-31')
            
            if 'Close' in df.columns and len(df) > 0:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df['Ticker'] = name
                df['Code'] = ticker
                df.reset_index(inplace=True)
                all_data.append(df)
                print(f"✅ {len(df)}일")
            else:
                print(f"⚠️ 데이터 없음")
                failed_tickers.append(name)
        except Exception as e:
            print(f"❌ 실패 - {e}")
            failed_tickers.append(name)
    
    if not all_data:
        raise ValueError("수집된 데이터가 없습니다!")
    
    if failed_tickers:
        print(f"\n⚠️  일부 종목 수집 실패: {', '.join(failed_tickers)}")
    
    # 데이터프레임 결합
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    print(f"\n✅ 수집 완료!")
    print(f"   - 기간: {combined_df['Date'].min().date()} ~ {combined_df['Date'].max().date()}")
    print(f"   - 거래일 수: {combined_df['Date'].nunique()}일")
    print(f"   - 성공 종목 수: {combined_df['Ticker'].nunique()}개")
    
    return combined_df


def save_december_data(df, output_dir='data'):
    """
    12월 데이터 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'etf_data_december.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 12월 데이터 저장: {output_path}")
    print(f"   파일 크기: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path


if __name__ == "__main__":
    # 12월 데이터 수집
    df = collect_december_data()
    
    # 저장
    save_december_data(df)
    
    # 통계
    print("\n📈 12월 데이터 통계:")
    stats = df.groupby('Ticker')['Close'].describe()[['count', 'mean', 'std', 'min', 'max']]
    print(stats.to_string())
    
    print("\n" + "=" * 70)
    print("✅ 12월 데이터 수집 완료!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("   python src/evaluate_december.py")
