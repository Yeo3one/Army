import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_75_igra_analysis(file_path: str) -> pd.DataFrame:
    # Load CSV file
    df = pd.read_csv(file_path, encoding="utf-8")

    # 공백 컬럼 제거
    df.columns = df.columns.str.strip()

    # 필요한 컬럼만 선별
    key_cols = ['입영연도', '생년', '검사연월', '위탁검사결과', 'NIL결과내용', 
                'TBGA결과내용', 'TBAG결과내용2', 'MITOGEN결과내용', 
                'TBAGNIL결과내용', 'TBAGNIL결과내용2']
    df = df[[col for col in key_cols if col in df.columns]]

    # 판정 결과 전처리
    # 판정 결과 전처리 (대문자 통일 후 매핑)
    df['위탁검사결과'] = df['위탁검사결과'].astype(str).str.strip().str.upper()

    # 한글 및 영어 모두 처리
    result_map = {
        'POSITIVE': 'Positive',
        'NEGATIVE': 'Negative',
        '양성': 'Positive',
        '음성': 'Negative',
        '양': 'Positive',
        '음': 'Negative'
    }
    df['위탁검사결과'] = df['위탁검사결과'].map(result_map)

    valid_df = df[df['위탁검사결과'].isin(['Positive', 'Negative'])]

    # 수치 컬럼 처리: 문자열 포함된 경우 제거 및 float 변환
    for col in df.columns:
        if '결과내용' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 양성/음성 값만 필터링 (결측 제거)
    valid_df = df[df['위탁검사결과'].isin(['Positive', 'Negative'])]
    total = len(valid_df)
    positives = (valid_df['위탁검사결과'] == 'Positive').sum()
    positive_rate = positives / total * 100 if total > 0 else np.nan

    print("[결과 요약]")
    print(f"총 검사자 수: {total}명")
    print(f"양성 판정자 수: {positives}명")
    print(f"양성률: {positive_rate:.2f}%")

    # 결과 일부 확인
    print("\n[전처리된 데이터 샘플]")
    print(df.head(2))

    return df

if __name__ == "__main__":
    file_path = Path("/Users/yeowon/Desktop/Data/병무청/75_병무청_잠복결핵검사의_효과분석을_위한_병역판정검사_자료제공.csv")
    preprocess_75_igra_analysis(file_path)