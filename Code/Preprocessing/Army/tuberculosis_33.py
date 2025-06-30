import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_33_hepatitis_b_data(file_path: str) -> pd.DataFrame:

    # 데이터 불러오기
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    
    # Unnamed 열 제거
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    # Glucose 텍스트 정리 및 매핑
    if 'Glucose' in df.columns:
        df['Glucose'] = df['Glucose'].astype(str).str.strip().str.upper()
        glucose_map = {
            '음성': 'Negative',
            '양성': 'Positive',
            'POS': 'Positive',
            'NEG': 'Negative',
            'NAN': np.nan,
            '': np.nan,
            'N/A': np.nan,
            'NULL': np.nan
        }
        df['Glucose'] = df['Glucose'].replace(glucose_map)

    # BMI 재계산 (신장, 체중 기준으로 이탈 크면 대체)
    if all(col in df.columns for col in ['bmi', '체중', '검사년도신장']):
        height_m = df['검사년도신장'] / 100
        calculated_bmi = df['체중'] / (height_m ** 2)
        bmi_diff = abs(df['bmi'] - calculated_bmi)
        df.loc[bmi_diff > 5, 'bmi'] = calculated_bmi[bmi_diff > 5].astype(df['bmi'].dtype)

    # 혈압 이상값 처리
    if '수축기(혈압)' in df.columns:
        df.loc[(df['수축기(혈압)'] < 70) | (df['수축기(혈압)'] > 250), '수축기(혈압)'] = np.nan

    if '이완기(혈압)' in df.columns:
        df.loc[(df['이완기(혈압)'] < 40) | (df['이완기(혈압)'] > 150), '이완기(혈압)'] = np.nan

    # ALT, 간염결과, Plt 컬럼 결측 확인
    for col in ['ALT', '간염결과', 'Plt']:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            total = len(df)
            if missing_count == total:
                print(f"'{col}' 컬럼은 존재하지만 모든 값이 비어 있습니다 ({missing_count}/{total} NaN).")
    print(df.head())
    return df

if __name__ == "__main__":
    file_path = Path("/Users/yeowon/Desktop/Data/병무청/33_병무청_B형간염_유병률_연구데이터_제공_20191231.csv")
    df = preprocess_33_hepatitis_b_data(file_path)