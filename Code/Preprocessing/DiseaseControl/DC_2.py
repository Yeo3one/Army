import pandas as pd
import os
from typing import Optional

def preprocess_dc2(filepath: str, verbose: bool = True) -> pd.DataFrame:
    
    # 파일 존재 확인
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    if verbose:
        print(f"파일 처리 중: {filepath}")
    
    try:
        df = pd.read_csv(filepath, header=[0, 1, 2], encoding='utf-8-sig')
    except UnicodeDecodeError:
        if verbose:
            print("UTF-8 인코딩 실패, EUC-KR로 재시도...")
        df = pd.read_csv(filepath, header=[0, 1, 2], encoding='euc-kr')
    
    if verbose:
        print(f"원본 데이터 형태: {df.shape}")
        print(f"원본 컬럼 구조:\n{df.columns.tolist()[:5]}...")  # 처음 5개만 표시
    
    # 2. 다중 헤더 병합 → 단일 문자열 컬럼명 생성
    new_columns = []
    for col in df.columns.values:
        # NaN이 아닌 값들과 Unnamed가 아닌 값들만 결합
        col_parts = []
        for part in col:
            part_str = str(part).strip()
            # NaN, Unnamed, 빈 문자열 제외
            if (part_str != 'nan' and 
                not part_str.startswith('Unnamed:') and 
                part_str != '' and 
                part_str != 'Unnamed'):
                col_parts.append(part_str)
        
        if col_parts:
            # 공백 제거 및 언더스코어로 연결
            column_name = '_'.join(col_parts).replace(" ", "").replace("__", "_")
            # 시작과 끝의 언더스코어 제거
            column_name = column_name.strip('_')
            new_columns.append(column_name)
        else:
            # 원본 컬럼에서 마지막 의미있는 부분 추출
            last_meaningful = None
            for part in reversed(col):
                part_str = str(part).strip()
                if (part_str != 'nan' and 
                    not part_str.startswith('Unnamed:') and 
                    part_str != '' and 
                    part_str != 'Unnamed'):
                    last_meaningful = part_str
                    break
            
            new_columns.append(last_meaningful if last_meaningful else 'unknown_column')
    
    df.columns = new_columns
    
    if verbose:
        print(f"병합된 컬럼명: {df.columns.tolist()}")
    
    # 3. 컬럼명 정제 및 표준화
    column_mapping = {
        '연도': '연도',
        '결핵환자': '결핵환자_수',
        '신환자': '신환자_수',
        '소계': '재치료자_수',
        '재발자': '재치료자_재발',
        '실패후재치료자': '재치료자_실패후',
        '중단후재치료자': '재치료자_중단후',
        '이전치료결과불명확': '재치료자_불명확결과',
        '과거치료여부불명확': '재치료자_불명확이력',
        '기타환자': '재치료자_기타',
        '과거치료력_재치료자_신환자': '신환자_수',
        '과거치료력_신환자': '신환자_수'
    }
    
    # 실제 존재하는 컬럼만 매핑
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df.rename(columns=existing_mapping, inplace=True)
    
    if verbose and existing_mapping:
        print(f"컬럼명 변경: {existing_mapping}")
    
    # 4. 데이터 정제: 쉼표 제거 및 숫자 변환
    numeric_columns = [col for col in df.columns if col != '연도']
    
    for col in numeric_columns:
        if col in df.columns:
            # 문자열로 변환 후 쉼표 제거
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            # 숫자로 변환 (오류 시 NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 5. 연도 처리
    if '연도' in df.columns:
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce')
        # 유효한 연도만 필터링
        df = df.dropna(subset=['연도'])
        df['연도'] = df['연도'].astype('int')
    
    # 6. 데이터 정렬 및 인덱스 리셋
    if '연도' in df.columns:
        df = df.sort_values(by='연도').reset_index(drop=True)
    
    # 7. 데이터 검증
    if df.empty:
        raise ValueError("처리된 데이터가 비어있습니다.")
    
    # 연도 범위 검증
    if '연도' in df.columns:
        min_year, max_year = df['연도'].min(), df['연도'].max()
        if min_year < 1900 or max_year > 2030:
            print(f"경고: 연도 범위가 예상과 다릅니다 ({min_year}-{max_year})")
    
    if verbose:
        print(f"처리 완료: {len(df)}행, {len(df.columns)}열")
        if '연도' in df.columns:
            print(f"연도 범위: {df['연도'].min()}-{df['연도'].max()}")
        print(f"최종 컬럼명: {list(df.columns)}")
    
    return df


def analyze_data_quality(df: pd.DataFrame) -> None:

    print("\n" + "="*60)
    print("데이터 품질 분석")
    print("="*60)
    
    # 기본 정보
    print(f"데이터 형태: {df.shape}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # 결측값 분석
    print("\n결측값 현황:")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_info = pd.DataFrame({
        '결측값_개수': missing_counts,
        '결측값_비율(%)': missing_percentages.round(2)
    })
    
    missing_info = missing_info[missing_info['결측값_개수'] > 0]
    if not missing_info.empty:
        print(missing_info)
    else:
        print("결측값 없음")
    
    # 데이터 타입 정보
    print(f"\n데이터 타입:")
    print(df.dtypes)
    
    # 수치형 컬럼 기본 통계
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\n기본 통계 (수치형 컬럼):")
        print(df[numeric_cols].describe())


def save_processed_data(df: pd.DataFrame, original_path: str, suffix: str = "_processed") -> str:

    # 출력 파일 경로 생성
    base_name = os.path.splitext(original_path)[0]
    output_path = f"{base_name}{suffix}.csv"
    
    # 데이터 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"처리된 데이터 저장: {output_path}")
    
    return output_path


if __name__ == "__main__":
    file_path = "/Users/yeowon/Desktop/Data/결핵/2_과거_치료력에_따른_결핵환자_수_2001-2024.csv"
    
    try:
        # 데이터 전처리
        print("결핵 데이터 전처리 시작...")
        df = preprocess_dc2(file_path, verbose=True)
        
        # 처리된 데이터 미리보기
        print("\n" + "="*60)
        print("처리된 데이터 미리보기")
        print("="*60)
        print(df.head(10))
        
        # 데이터 품질 분석
        analyze_data_quality(df)
        
        # 최근 데이터 확인
        if '연도' in df.columns and len(df) >= 5:
            print("\n" + "="*60)
            print("최근 5년 데이터")
            print("="*60)
            recent_data = df.tail(5)
            print(recent_data)
        
        # 연도별 총 결핵환자 수 추이 (간단한 분석)
        if '결핵환자_수' in df.columns and '연도' in df.columns:
            print("\n" + "="*60)
            print("연도별 결핵환자 수 추이 (최근 10년)")
            print("="*60)
            recent_trend = df.tail(10)[['연도', '결핵환자_수']]
            print(recent_trend.to_string(index=False))
        
        # 처리된 데이터 저장
        output_file = save_processed_data(df, file_path)
        
        print(f"\n전처리 완료! 총 {len(df)}개 연도의 데이터를 처리했습니다.")
        
    except FileNotFoundError as e:
        print(f"파일 오류: {e}")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        print("데이터 형식이나 파일 구조를 확인해주세요.")