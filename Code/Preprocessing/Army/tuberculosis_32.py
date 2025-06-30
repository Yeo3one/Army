import pandas as pd

def preprocess_32_disease_grading(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path, encoding='utf-8')

    # 컬럼명 정리
    df.columns = df.columns.str.strip()

    # 질병명 공백 제거 및 표준화
    if '질병명' in df.columns:
        df['질병명'] = df['질병명'].str.strip()

    # 부령조항에서 코드만 추출 (필요 시)
    if '부령조항' in df.columns:
        df['부령코드'] = df['부령조항'].astype(str).str.extract(r'(\d{3})')

    # 등급 컬럼 확인
    for grade_col in ['4급', '5급', '6급']:
        if grade_col in df.columns:
            df[grade_col] = pd.to_numeric(df[grade_col], errors='coerce').fillna(0)

    # 전체 합계 컬럼이 존재하면 재계산
    if '계' in df.columns and all(col in df.columns for col in ['4급', '5급', '6급']):
        df['계_재계산'] = df['4급'] + df['5급'] + df['6급']

    return df


if __name__ == "__main__":
    file_path = "/Users/yeowon/Desktop/Data/병무청/32_병무청_병역판정검사_결과(4급5급6급)_20181231.csv"
    df = preprocess_32_disease_grading(file_path)
    print(df.head())
