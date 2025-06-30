import pandas as pd
import numpy as np
from pathlib import Path


def preprocess_44_latent_tb_data(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path, encoding='utf-8')

    # 컬럼명 정리
    df.columns = df.columns.str.strip()

    # 날짜형 변환
    if '검사일' in df.columns:
        df['검사일'] = pd.to_datetime(df['검사일'], errors='coerce')
        df['검사연도'] = df['검사일'].dt.year

    # 연령 처리
    if '생년' in df.columns:
        df['연령'] = df['검사연도'] - df['생년']

    # IGRA 검사결과 이진화 (기준: 0.35 이상 → 양성)
    if 'IGRA결과값' in df.columns:
        df['IGRA양성'] = np.where(df['IGRA결과값'] >= 0.35, 'Positive', 'Negative')

    # 신장 및 체중 이상치 처리
    if '신장' in df.columns:
        df.loc[(df['신장'] < 130) | (df['신장'] > 210), '신장'] = np.nan
    if '체중' in df.columns:
        df.loc[(df['체중'] < 30) | (df['체중'] > 150), '체중'] = np.nan

    return df

if __name__ == '__main__':
    file_path = Path("/Users/yeowon/Desktop/Data/병무청/44_병무청_잠복결핵_유병률_조사_데이터_제공[Kdata]_20211231.csv")
    df = preprocess_44_latent_tb_data(file_path)
    print(df.head())