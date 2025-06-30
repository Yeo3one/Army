import pandas as pd
import numpy as np

def preprocess_59_latent_tb_by_region(file_path: str, year: int = 2023) -> pd.DataFrame:

    # CSV 파일 읽기
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    
    # Unnamed 열 제거
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    
    # 연도 정보 추가
    df['연도'] = year

    # 퍼센트(%) 제거 후 수치형 변환
    percent_cols = ['음성비율', '양성비율', '판독불명비율']
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 인원 수 관련 컬럼 숫자형 처리
    count_cols = ['검사인원', '음성', '양성', '판독불명']
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # '구분'이 지역을 의미하므로 이름 변경
    df.rename(columns={'구분': '지역'}, inplace=True)

    # 지역명이 없는 행 제거 (총계일 가능성 있음)
    df = df.dropna(subset=['지역'])

    return df

if __name__ == "__main__":
    file_path = "/Users/yeowon/Desktop/Data/병무청/59_병무청_병역판정검사_잠복결핵검사_실적및청별_현황_20231231.csv"
    df = preprocess_59_latent_tb_by_region(file_path, year=2023)
    print(df.head())
    print(df.describe())