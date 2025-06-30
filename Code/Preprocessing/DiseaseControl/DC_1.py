import pandas as pd
import os
from typing import Tuple

def preprocess_DC_1(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    결핵 신환자 데이터 전처리 함수
    
    Args:
        file_path (str): CSV 파일 경로
        verbose (bool): 진행상황 출력 여부
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    if verbose:
        print(f"파일 처리 중: {file_path}")
    
    try:
        # CSV 파일 읽기 (다중 헤더 건너뛰기)
        df_raw = pd.read_csv(file_path, skiprows=1, encoding='utf-8-sig')
    except UnicodeDecodeError:
        if verbose:
            print("UTF-8 인코딩 실패, EUC-KR로 재시도...")
        df_raw = pd.read_csv(file_path, skiprows=1, encoding='euc-kr')
    
    if verbose:
        print(f"원본 데이터 형태: {df_raw.shape}")
        print(f"원본 컬럼: {df_raw.columns.tolist()}")
    
    # 컬럼명 설정
    expected_columns = ['연도', '결핵환자수', '결핵환자율', '신환자수', '신환자율']
    
    if len(df_raw.columns) != len(expected_columns):
        print(f"경고: 예상 컬럼 수({len(expected_columns)})와 실제 컬럼 수({len(df_raw.columns)})가 다릅니다.")
        print(f"실제 컬럼: {df_raw.columns.tolist()}")
    
    df_raw.columns = expected_columns[:len(df_raw.columns)]
    
    # 빈 행 제거
    df_raw = df_raw.dropna(subset=['연도']).reset_index(drop=True)
    
    if verbose:
        print(f"빈 행 제거 후: {df_raw.shape}")
    
    # 숫자 데이터 정제
    numeric_columns = ['결핵환자수', '신환자수']
    for col in numeric_columns:
        if col in df_raw.columns:
            # 문자열로 변환 후 쉼표와 따옴표 제거
            df_raw[col] = (df_raw[col].astype(str)
                          .str.replace(',', '', regex=False)
                          .str.replace('"', '', regex=False)
                          .str.strip())
            # 숫자로 변환
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    # 비율 데이터 정제 (괄호 제거)
    rate_columns = ['결핵환자율', '신환자율']
    for col in rate_columns:
        if col in df_raw.columns:
            # 괄호 제거 및 숫자 변환
            df_raw[col] = (df_raw[col].astype(str)
                          .str.replace('(', '', regex=False)
                          .str.replace(')', '', regex=False)
                          .str.strip())
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    # 연도 데이터 정제
    df_raw['연도'] = pd.to_numeric(df_raw['연도'], errors='coerce')
    
    # 유효하지 않은 데이터 제거
    df_clean = df_raw.dropna(subset=['연도']).copy()
    df_clean['연도'] = df_clean['연도'].astype(int)
    
    # 컬럼명 표준화
    df_renamed = df_clean.rename(columns={
        '결핵환자수': '결핵환자_수',
        '결핵환자율': '결핵환자_율',
        '신환자수': '신환자_수',
        '신환자율': '신환자_율'
    })
    
    # 데이터 검증
    if df_renamed.empty:
        raise ValueError("처리된 데이터가 비어있습니다.")
    
    # 연도 범위 검증
    min_year, max_year = df_renamed['연도'].min(), df_renamed['연도'].max()
    if min_year < 1900 or max_year > 2030:
        print(f"경고: 연도 범위가 예상과 다릅니다 ({min_year}-{max_year})")
    
    # 논리적 검증: 신환자 수 <= 전체 결핵환자 수
    if '신환자_수' in df_renamed.columns and '결핵환자_수' in df_renamed.columns:
        invalid_rows = df_renamed[df_renamed['신환자_수'] > df_renamed['결핵환자_수']]
        if not invalid_rows.empty:
            print(f"경고: 신환자 수가 전체 결핵환자 수보다 큰 행이 {len(invalid_rows)}개 있습니다.")
    
    # 데이터 정렬
    df_final = df_renamed.sort_values(by='연도').reset_index(drop=True)
    
    if verbose:
        print(f"처리 완료: {len(df_final)}행, {len(df_final.columns)}열")
        print(f"연도 범위: {df_final['연도'].min()}-{df_final['연도'].max()}")
        print(f"최종 컬럼: {list(df_final.columns)}")
    
    return df_final


def validate_data_quality(df: pd.DataFrame) -> None:
    """
    데이터 품질 검증 함수
    
    Args:
        df (pd.DataFrame): 검증할 데이터프레임
    """
    print("\n" + "="*60)
    print("데이터 품질 검증")
    print("="*60)
    
    # 1. 기본 정보
    print(f"데이터 형태: {df.shape}")
    print(f"연도 범위: {df['연도'].min()}-{df['연도'].max()}")
    
    # 2. 결측값 확인
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n결측값 현황:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"- {col}: {missing}개 ({missing/len(df)*100:.1f}%)")
    else:
        print("\n✅ 결측값 없음")
    
    # 3. 데이터 타입 확인
    print(f"\n데이터 타입:")
    for col, dtype in df.dtypes.items():
        print(f"- {col}: {dtype}")
    
    # 4. 논리적 일관성 검증
    print(f"\n논리적 일관성 검증:")
    
    # 신환자 수 vs 전체 결핵환자 수
    if '신환자_수' in df.columns and '결핵환자_수' in df.columns:
        invalid_count = (df['신환자_수'] > df['결핵환자_수']).sum()
        if invalid_count == 0:
            print("✅ 신환자 수 ≤ 전체 결핵환자 수")
        else:
            print(f"❌ 신환자 수 > 전체 결핵환자 수인 행: {invalid_count}개")
    
    # 비율 데이터 범위 확인
    rate_columns = [col for col in df.columns if '율' in col]
    for col in rate_columns:
        if col in df.columns:
            min_rate = df[col].min()
            max_rate = df[col].max()
            if 0 <= min_rate and max_rate <= 200:  # 인구 10만명당이므로 200 정도가 합리적 상한
                print(f"✅ {col} 범위 적절: {min_rate:.1f} ~ {max_rate:.1f}")
            else:
                print(f"⚠️ {col} 범위 확인 필요: {min_rate:.1f} ~ {max_rate:.1f}")
    
    # 5. 기본 통계
    print(f"\n기본 통계:")
    numeric_cols = df.select_dtypes(include=['number']).columns
    print(df[numeric_cols].describe().round(1))


def calculate_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    추가 지표 계산 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: 추가 지표가 포함된 데이터프레임
    """
    df_enhanced = df.copy()
    
    # 재치료자 수 및 비율 계산
    if '결핵환자_수' in df.columns and '신환자_수' in df.columns:
        df_enhanced['재치료자_수'] = df_enhanced['결핵환자_수'] - df_enhanced['신환자_수']
        df_enhanced['재치료자_비율'] = (df_enhanced['재치료자_수'] / df_enhanced['결핵환자_수']) * 100
    
    # 전년 대비 증감률 계산
    if '결핵환자_수' in df.columns:
        df_enhanced['결핵환자_증감률'] = df_enhanced['결핵환자_수'].pct_change() * 100
        df_enhanced['신환자_증감률'] = df_enhanced['신환자_수'].pct_change() * 100
    
    # 5년 이동평균 계산
    if len(df) >= 5:
        df_enhanced['결핵환자_5년평균'] = df_enhanced['결핵환자_수'].rolling(window=5, center=True).mean()
        df_enhanced['신환자_5년평균'] = df_enhanced['신환자_수'].rolling(window=5, center=True).mean()
    
    return df_enhanced


if __name__ == "__main__":
    file_path = "/Users/yeowon/Desktop/Data/결핵/1_결핵_(신)환자_수및_율_2001-2024.csv"
    
    try:
        # 데이터 전처리
        print("결핵 신환자 데이터 전처리 시작...")
        df_processed = preprocess_DC_1(file_path, verbose=True)
        
        # 처리된 데이터 미리보기
        print("\n" + "="*60)
        print("처리된 데이터 미리보기")
        print("="*60)
        print(df_processed.head(10))
        
        # 데이터 품질 검증
        validate_data_quality(df_processed)
        
        # 추가 지표 계산
        df_enhanced = calculate_additional_metrics(df_processed)
        
        print("\n" + "="*60)
        print("추가 지표가 포함된 데이터 (최근 5년)")
        print("="*60)
        recent_data = df_enhanced.tail(5)[['연도', '결핵환자_수', '신환자_수', '재치료자_수', 
                                          '재치료자_비율', '결핵환자_증감률']].round(2)
        print(recent_data.to_string(index=False))
        
        # 처리된 데이터 저장
        output_path = file_path.replace('.csv', '_processed.csv')
        df_enhanced.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n처리된 데이터 저장: {output_path}")
        
        print(f"\n전처리 완료! 총 {len(df_processed)}개 연도의 데이터를 처리했습니다.")
        
    except FileNotFoundError as e:
        print(f"파일 오류: {e}")
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()