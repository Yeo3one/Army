import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict

def preprocess_regional_tuberculosis_data(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    지역별 결핵 환자 데이터 전처리 함수
    
    Args:
        file_path (str): CSV 파일 경로
        verbose (bool): 진행상황 출력 여부
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    
    if verbose:
        print(f"지역별 결핵 데이터 전처리 시작: {file_path}")
    
    try:
        # CSV 파일 읽기 (헤더 없이)
        df_raw = pd.read_csv(file_path, header=None, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(file_path, header=None, encoding='euc-kr')
    
    if verbose:
        print(f"원본 데이터 형태: {df_raw.shape}")
    
    # 헤더 설정 (첫 번째 행 기준)
    age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                  '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', 
                  '65-69', '70-74', '75-79', '80', '미상']
    
    columns = ['시도', '환자유형', '성별', '계'] + age_groups
    df_raw.columns = columns[:df_raw.shape[1]]  # 실제 컬럼 수에 맞춰 조정
    
    # 빈 행 제거
    df_clean = df_raw.dropna(how='all').reset_index(drop=True)
    
    if verbose:
        print(f"빈 행 제거 후: {df_clean.shape}")
    
    # 데이터 정제된 리스트 생성
    processed_rows = []
    current_region = None
    current_patient_type = None
    
    for idx, row in df_clean.iterrows():
        # 시도명 확인 (첫 번째 컬럼이 비어있지 않은 경우)
        if pd.notna(row['시도']) and row['시도'].strip():
            current_region = row['시도'].strip()
            
        # 환자유형 확인 (두 번째 컬럼이 비어있지 않은 경우)
        if pd.notna(row['환자유형']) and row['환자유형'].strip():
            current_patient_type = row['환자유형'].strip()
        
        # 성별 정보가 있는 행만 처리
        if pd.notna(row['성별']) and row['성별'].strip() in ['계', '남', '여']:
            new_row = {
                '시도': current_region,
                '환자유형': current_patient_type,
                '성별': row['성별'].strip()
            }
            
            # 연령대별 데이터 처리
            for col in ['계'] + age_groups:
                if col in df_clean.columns and pd.notna(row[col]):
                    value_str = str(row[col]).strip()
                    
                    # 숫자 추출 (쉼표가 있는 숫자 처리)
                    if value_str and value_str != 'nan':
                        # 괄호로 둘러싸인 비율 데이터 분리
                        if '(' in value_str and ')' in value_str:
                            # 괄호 앞의 숫자 (환자 수)
                            count_match = re.search(r'^"?([0-9,]+)"?', value_str)
                            if count_match:
                                count = count_match.group(1).replace(',', '').replace('"', '')
                                new_row[f'{col}_수'] = int(count) if count.isdigit() else 0
                            else:
                                new_row[f'{col}_수'] = 0
                            
                            # 괄호 안의 비율
                            rate_match = re.search(r'\(([0-9.-]+)\)', value_str)
                            if rate_match:
                                rate = rate_match.group(1)
                                new_row[f'{col}_율'] = float(rate) if rate.replace('.', '').replace('-', '').isdigit() else 0.0
                            else:
                                new_row[f'{col}_율'] = 0.0
                        else:
                            # 숫자만 있는 경우
                            clean_value = value_str.replace(',', '').replace('"', '')
                            if clean_value.isdigit():
                                new_row[f'{col}_수'] = int(clean_value)
                                new_row[f'{col}_율'] = 0.0
                            else:
                                new_row[f'{col}_수'] = 0
                                new_row[f'{col}_율'] = 0.0
                    else:
                        new_row[f'{col}_수'] = 0
                        new_row[f'{col}_율'] = 0.0
            
            processed_rows.append(new_row)
    
    # DataFrame 생성
    df_processed = pd.DataFrame(processed_rows)
    
    # 데이터 검증 및 정제
    df_processed = df_processed[df_processed['시도'].notna()].copy()
    df_processed = df_processed[df_processed['환자유형'].notna()].copy()
    
    if verbose:
        print(f"처리된 데이터 형태: {df_processed.shape}")
        print(f"시도 수: {df_processed['시도'].nunique()}")
        print(f"환자유형: {df_processed['환자유형'].unique()}")
        print(f"성별: {df_processed['성별'].unique()}")
    
    return df_processed


def validate_regional_data(df: pd.DataFrame) -> None:
    """
    지역별 데이터 검증 함수
    
    Args:
        df (pd.DataFrame): 검증할 데이터프레임
    """
    print("\n" + "="*60)
    print("지역별 데이터 검증")
    print("="*60)
    
    # 1. 기본 정보
    print(f"데이터 형태: {df.shape}")
    print(f"지역 수: {df['시도'].nunique()}")
    print(f"지역 목록: {', '.join(df['시도'].unique())}")
    
    # 2. 환자유형별 데이터 확인
    print(f"\n환자유형별 데이터 수:")
    type_counts = df['환자유형'].value_counts()
    for patient_type, count in type_counts.items():
        print(f"- {patient_type}: {count}개")
    
    # 3. 성별 분포 확인
    print(f"\n성별 분포:")
    gender_counts = df['성별'].value_counts()
    for gender, count in gender_counts.items():
        print(f"- {gender}: {count}개")
    
    # 4. 결측값 확인
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if len(missing_cols) > 0:
        print(f"\n결측값 있는 컬럼:")
        for col, missing in missing_cols.items():
            print(f"- {col}: {missing}개")
    else:
        print(f"\n✅ 결측값 없음")
    
    # 5. 전국 vs 지역별 데이터 일치성 검증 (계 컬럼 기준)
    if '전국' in df['시도'].values and '계_수' in df.columns:
        national_data = df[(df['시도'] == '전국') & (df['성별'] == '계')]
        regional_data = df[(df['시도'] != '전국') & (df['성별'] == '계')]
        
        if len(national_data) > 0 and len(regional_data) > 0:
            for patient_type in df['환자유형'].unique():
                national_total = national_data[national_data['환자유형'] == patient_type]['계_수'].sum()
                regional_total = regional_data[regional_data['환자유형'] == patient_type]['계_수'].sum()
                
                diff = abs(national_total - regional_total)
                diff_rate = (diff / national_total * 100) if national_total > 0 else 0
                
                status = "✅" if diff_rate < 5 else "⚠️"
                print(f"\n{status} {patient_type} 합계 검증:")
                print(f"  전국: {national_total:,}명")
                print(f"  지역합계: {regional_total:,}명")
                print(f"  차이: {diff:,}명 ({diff_rate:.1f}%)")


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    요약 통계 생성 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: 요약 통계 데이터프레임
    """
    
    # 지역별 환자유형별 총계 계산
    summary_data = []
    
    for region in df['시도'].unique():
        if region == '전국':
            continue
            
        region_data = df[df['시도'] == region]
        
        for patient_type in df['환자유형'].unique():
            type_data = region_data[region_data['환자유형'] == patient_type]
            
            # 계 데이터 (전체)
            total_data = type_data[type_data['성별'] == '계']
            if len(total_data) > 0:
                total_patients = total_data['계_수'].iloc[0]
                total_rate = total_data['계_율'].iloc[0]
                
                # 남녀 데이터
                male_data = type_data[type_data['성별'] == '남']
                female_data = type_data[type_data['성별'] == '여']
                
                male_patients = male_data['계_수'].iloc[0] if len(male_data) > 0 else 0
                female_patients = female_data['계_수'].iloc[0] if len(female_data) > 0 else 0
                
                # 주요 연령대 분석 (40-64세)
                middle_age_cols = ['40-44_수', '45-49_수', '50-54_수', '55-59_수', '60-64_수']
                middle_age_total = sum([total_data[col].iloc[0] for col in middle_age_cols if col in total_data.columns])
                middle_age_ratio = (middle_age_total / total_patients * 100) if total_patients > 0 else 0
                
                # 고령자 분석 (65세 이상)
                elderly_cols = ['65-69_수', '70-74_수', '75-79_수', '80_수']
                elderly_total = sum([total_data[col].iloc[0] for col in elderly_cols if col in total_data.columns])
                elderly_ratio = (elderly_total / total_patients * 100) if total_patients > 0 else 0
                
                summary_data.append({
                    '시도': region,
                    '환자유형': patient_type,
                    '총_환자수': total_patients,
                    '발생률': total_rate,
                    '남성_환자수': male_patients,
                    '여성_환자수': female_patients,
                    '남성_비율': (male_patients / total_patients * 100) if total_patients > 0 else 0,
                    '중장년_환자수_40_64세': middle_age_total,
                    '중장년_비율': middle_age_ratio,
                    '고령_환자수_65세이상': elderly_total,
                    '고령_비율': elderly_ratio
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 정렬 (환자 수 기준 내림차순)
    summary_df = summary_df.sort_values(['환자유형', '총_환자수'], ascending=[True, False])
    
    return summary_df


def analyze_age_distribution(df: pd.DataFrame, region: str = '전국') -> None:
    """
    연령대별 분포 분석 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        region (str): 분석할 지역 (기본값: 전국)
    """
    
    print(f"\n" + "="*60)
    print(f"{region} 연령대별 결핵 환자 분포 분석")
    print("="*60)
    
    # 해당 지역의 계 데이터만 추출
    region_data = df[(df['시도'] == region) & (df['성별'] == '계')]
    
    if len(region_data) == 0:
        print(f"'{region}' 데이터를 찾을 수 없습니다.")
        return
    
    age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                  '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', 
                  '65-69', '70-74', '75-79', '80']
    
    for patient_type in region_data['환자유형'].unique():
        type_data = region_data[region_data['환자유형'] == patient_type].iloc[0]
        
        print(f"\n📊 {patient_type} 연령대별 분포:")
        print(f"총 환자 수: {type_data['계_수']:,}명")
        
        age_analysis = []
        for age in age_groups:
            if f'{age}_수' in type_data:
                count = type_data[f'{age}_수']
                rate = type_data[f'{age}_율']
                percentage = (count / type_data['계_수'] * 100) if type_data['계_수'] > 0 else 0
                
                age_analysis.append({
                    '연령대': age,
                    '환자수': count,
                    '구성비(%)': percentage,
                    '발생률': rate
                })
        
        # 상위 5개 연령대
        age_df = pd.DataFrame(age_analysis)
        top_5_ages = age_df.nlargest(5, '환자수')
        
        print("\n상위 5개 연령대:")
        for _, row in top_5_ages.iterrows():
            print(f"- {row['연령대']}세: {row['환자수']:,}명 ({row['구성비(%)']:.1f}%, 발생률 {row['발생률']:.1f})")


if __name__ == "__main__":
    # 파일 경로 설정 (실제 파일 경로로 변경 필요)
    file_path = "/Users/yeowon/Desktop/Data/결핵/3_시도별_성별_연령별_결핵_(신)환자수_및_율_2024.csv"
    
    try:
        # 데이터 전처리
        print("지역별 결핵 데이터 전처리 시작...")
        df_processed = preprocess_regional_tuberculosis_data(file_path, verbose=True)
        
        # 처리된 데이터 미리보기
        print("\n" + "="*60)
        print("처리된 데이터 미리보기")
        print("="*60)
        print(df_processed.head(10))
        
        # 데이터 검증
        validate_regional_data(df_processed)
        
        # 요약 통계 생성
        summary_stats = create_summary_statistics(df_processed)
        print("\n" + "="*60)
        print("지역별 요약 통계 (결핵환자 상위 10개 지역)")
        print("="*60)
        
        tb_summary = summary_stats[summary_stats['환자유형'] == '결핵환자'].head(10)
        print(tb_summary[['시도', '총_환자수', '발생률', '남성_비율', '고령_비율']].to_string(index=False))
        
        # 전국 연령대별 분석
        analyze_age_distribution(df_processed, '전국')
        
        # 처리된 데이터 저장
        output_path = file_path.replace('.csv', '_processed.csv')
        df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 요약 통계 저장
        summary_path = file_path.replace('.csv', '_summary.csv')
        summary_stats.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n처리 완료!")
        print(f"전처리된 데이터: {output_path}")
        print(f"요약 통계: {summary_path}")
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()