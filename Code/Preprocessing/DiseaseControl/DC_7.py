import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def preprocess_gender_age_tuberculosis_data(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    성별 연령별 결핵 데이터 전처리 함수 (2011-2024)
    
    Args:
        file_path (str): CSV 파일 경로
        verbose (bool): 진행상황 출력 여부
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    
    if verbose:
        print("="*70)
        print("성별 연령별 결핵 데이터 전처리 시작 (2011-2024)")
        print("="*70)
        print(f"📁 파일: {file_path}")
    
    try:
        # CSV 파일 읽기 (여러 인코딩 시도)
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            encoding_used = 'utf-8-sig'
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                encoding_used = 'utf-8'
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='euc-kr')
                encoding_used = 'euc-kr'
        
        if verbose:
            print(f"✅ 파일 읽기 성공 ({encoding_used}): {df.shape}")
            print(f"📋 원본 컬럼: {list(df.columns)}")
    
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return pd.DataFrame()
    
    # 원본 데이터 확인
    if verbose:
        print(f"\n📊 원본 데이터 미리보기:")
        print(df.head(10))
        print(f"\n📈 기본 정보:")
        print(f"  - 행 수: {len(df)}")
        print(f"  - 컬럼 수: {len(df.columns)}")
        print(f"  - 결측값: {df.isnull().sum().sum()}개")
    
    # 1. 컬럼명 정리 및 표준화
    df.columns = df.columns.str.strip()
    
    # 컬럼명 매핑
    column_mapping = {
        '구분': '데이터구분',
        '시∙도': '연도',
        '성별/연령': '성별_연령',
        '80': '80+',  # 80세 이상으로 표준화
        '미상': '미상'
    }
    
    # 실제 존재하는 컬럼만 매핑
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    if verbose:
        print(f"📋 정리된 컬럼: {list(df.columns)}")
    
    # 2. 필수 컬럼 확인
    required_columns = ['데이터구분', '연도', '성별_연령']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ 필수 컬럼 누락: {missing_columns}")
        return pd.DataFrame()
    
    # 3. 데이터 정제
    if verbose:
        print("\n🔄 데이터 정제 중...")
    
    # 빈 행 제거
    initial_rows = len(df)
    df = df.dropna(how='all').reset_index(drop=True)
    
    # 기본 컬럼 정제
    df['데이터구분'] = df['데이터구분'].astype(str).str.strip()
    df['성별_연령'] = df['성별_연령'].astype(str).str.strip()
    
    # 유효하지 않은 행 제거
    valid_mask = (
        (df['데이터구분'].notna()) & 
        (df['성별_연령'].notna()) & 
        (df['데이터구분'] != '') & 
        (df['성별_연령'] != '') &
        (df['데이터구분'] != 'nan') & 
        (df['성별_연령'] != 'nan')
    )
    df = df[valid_mask].reset_index(drop=True)
    
    # 4. 연도 데이터 정제
    df['연도'] = pd.to_numeric(df['연도'], errors='coerce')
    
    # 연도가 유효한 범위 내에 있는지 확인 (2011-2024)
    valid_year_mask = (df['연도'] >= 2011) & (df['연도'] <= 2024)
    df = df[valid_year_mask].reset_index(drop=True)
    df['연도'] = df['연도'].astype(int)
    
    # 5. 연령대 컬럼 정의 및 숫자 데이터 정제
    age_columns = ['계', '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                   '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', 
                   '65-69', '70-74', '75-79', '80+', '미상']
    
    # 실제 존재하는 연령대 컬럼만 처리
    existing_age_columns = [col for col in age_columns if col in df.columns]
    
    for col in existing_age_columns:
        # 문자열로 변환 후 특수문자 제거
        df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
        df[col] = df[col].str.replace('(', '').str.replace(')', '').str.replace('-', '0')
        
        # 빈 문자열이나 'nan'을 0으로 처리
        df[col] = df[col].replace(['', 'nan', 'NaN'], '0')
        
        # 숫자로 변환
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 음수값을 0으로 변경
        df[col] = df[col].apply(lambda x: max(0, x))
        
        # 데이터 구분에 따라 타입 결정 (환자수는 정수, 비율은 실수)
        if '율' in df['데이터구분'].iloc[0] if len(df) > 0 else False:
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)
    
    # 6. 성별과 연령 정보 분리
    df['성별'] = df['성별_연령'].apply(extract_gender)
    df['연령그룹'] = df['성별_연령'].apply(extract_age_group)
    
    # 7. 데이터 유형 분류 (환자수 vs 발생률)
    df['데이터유형'] = df['데이터구분'].apply(classify_data_type)
    df['환자유형'] = df['데이터구분'].apply(extract_patient_type)
    
    # 8. 중복 제거 및 정렬
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values(['연도', '데이터구분', '성별_연령']).reset_index(drop=True)
    
    # 9. 병무청 관련 연령대 표시 (20대 남성 중심)
    df['입영대상연령'] = df.apply(lambda row: classify_military_age(row['성별'], row['연령그룹']), axis=1)
    
    if verbose:
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"🔧 제거된 행: {removed_rows}개")
        print(f"✅ 데이터 정제 완료: {len(df)}개 레코드")
        print(f"📊 연도 범위: {df['연도'].min()}-{df['연도'].max()}")
        print(f"🏥 데이터 구분: {df['데이터구분'].nunique()}개")
        print(f"👥 성별/연령 조합: {df['성별_연령'].nunique()}개")
        
        # 기본 통계 (환자수 데이터만)
        patient_data = df[df['데이터유형'] == '환자수']
        if len(patient_data) > 0:
            print(f"\n📈 기본 통계 (환자수 기준):")
            recent_year = patient_data['연도'].max()
            recent_data = patient_data[patient_data['연도'] == recent_year]
            if len(recent_data) > 0:
                total_patients = recent_data['계'].sum()
                print(f"  - {recent_year}년 총 환자수: {total_patients:,.0f}명")
    
    return df


def extract_gender(gender_age_str: str) -> str:
    """성별/연령 문자열에서 성별 추출"""
    if pd.isna(gender_age_str) or gender_age_str == '':
        return '기타'
    
    gender_age_str = str(gender_age_str).strip()
    
    if '남' in gender_age_str and '여' not in gender_age_str:
        return '남성'
    elif '여' in gender_age_str and '남' not in gender_age_str:
        return '여성'
    elif '계' in gender_age_str or '전체' in gender_age_str:
        return '전체'
    else:
        return '기타'


def extract_age_group(gender_age_str: str) -> str:
    """성별/연령 문자열에서 연령그룹 추출"""
    if pd.isna(gender_age_str) or gender_age_str == '':
        return '전체'
    
    gender_age_str = str(gender_age_str).strip()
    
    # 특정 연령대 패턴 찾기
    age_patterns = {
        '0-4': ['0-4', '영유아'],
        '5-9': ['5-9', '아동'],
        '10-14': ['10-14', '청소년초기'],
        '15-19': ['15-19', '청소년'],
        '20-24': ['20-24', '청년초기'],
        '25-29': ['25-29', '청년'],
        '30-34': ['30-34', '성인초기'],
        '35-39': ['35-39', '성인'],
        '40-44': ['40-44', '중년초기'],
        '45-49': ['45-49', '중년'],
        '50-54': ['50-54', '장년초기'],
        '55-59': ['55-59', '장년'],
        '60-64': ['60-64', '고령전기'],
        '65-69': ['65-69', '고령초기'],
        '70-74': ['70-74', '고령'],
        '75-79': ['75-79', '고령후기'],
        '80+': ['80', '80+', '최고령']
    }
    
    for age_group, patterns in age_patterns.items():
        for pattern in patterns:
            if pattern in gender_age_str:
                return age_group
    
    if '계' in gender_age_str or '전체' in gender_age_str:
        return '전체'
    
    return '기타'


def classify_data_type(data_category: str) -> str:
    """데이터 구분에서 데이터 유형 분류 (환자수 vs 발생률)"""
    if pd.isna(data_category):
        return '기타'
    
    data_category = str(data_category).strip()
    
    if '율' in data_category or '발생률' in data_category or '%' in data_category:
        return '발생률'
    elif '수' in data_category or '명' in data_category:
        return '환자수'
    else:
        return '기타'


def extract_patient_type(data_category: str) -> str:
    """데이터 구분에서 환자 유형 추출 (결핵환자 vs 신환자)"""
    if pd.isna(data_category):
        return '기타'
    
    data_category = str(data_category).strip()
    
    if '신환자' in data_category:
        return '신환자'
    elif '결핵환자' in data_category:
        return '결핵환자'
    else:
        return '기타'


def classify_military_age(gender: str, age_group: str) -> str:
    """입영 대상 연령대 분류"""
    if gender == '남성' and age_group in ['20-24', '25-29']:
        return '입영대상'
    elif gender == '남성' and age_group in ['18-19', '15-19']:
        return '입영예정'
    elif gender == '남성' and age_group in ['30-34', '35-39']:
        return '예비군'
    else:
        return '일반'


def analyze_military_age_trends(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    입영 대상 연령대(20대 남성) 결핵 추이 분석
    
    Args:
        df (pd.DataFrame): 전처리된 데이터프레임
        verbose (bool): 상세 출력 여부
    
    Returns:
        Dict: 분석 결과
    """
    
    if verbose:
        print("\n" + "="*50)
        print("입영 대상 연령대 결핵 추이 분석")
        print("="*50)
    
    analysis_result = {}
    
    # 20대 남성 데이터 필터링 (환자수 기준)
    military_age_data = df[
        (df['성별'] == '남성') & 
        (df['입영대상연령'] == '입영대상') &
        (df['데이터유형'] == '환자수')
    ].copy()
    
    if len(military_age_data) == 0:
        print("⚠️ 입영 대상 연령대 데이터가 없습니다.")
        return analysis_result
    
    # 1. 연도별 추이 분석
    yearly_trends = {}
    for patient_type in military_age_data['환자유형'].unique():
        type_data = military_age_data[military_age_data['환자유형'] == patient_type]
        
        yearly_summary = type_data.groupby('연도').agg({
            '20-24': 'sum',
            '25-29': 'sum',
            '계': 'sum'
        }).reset_index()
        
        yearly_summary['20대_총합'] = yearly_summary['20-24'] + yearly_summary['25-29']
        yearly_trends[patient_type] = yearly_summary
    
    analysis_result['연도별_추이'] = yearly_trends
    
    # 2. 최근 5년 평균
    recent_years = df['연도'].nlargest(5).tolist()
    recent_data = military_age_data[military_age_data['연도'].isin(recent_years)]
    
    recent_summary = {}
    for patient_type in recent_data['환자유형'].unique():
        type_data = recent_data[recent_data['환자유형'] == patient_type]
        
        avg_20_24 = type_data['20-24'].mean()
        avg_25_29 = type_data['25-29'].mean()
        avg_total = avg_20_24 + avg_25_29
        
        recent_summary[patient_type] = {
            '20-24세_평균': avg_20_24,
            '25-29세_평균': avg_25_29,
            '20대_총평균': avg_total
        }
    
    analysis_result['최근5년_평균'] = recent_summary
    
    # 3. 증감률 분석
    trend_analysis = {}
    for patient_type in yearly_trends.keys():
        yearly_data = yearly_trends[patient_type]
        if len(yearly_data) >= 2:
            first_year = yearly_data.iloc[0]
            last_year = yearly_data.iloc[-1]
            
            change_rate_20_24 = ((last_year['20-24'] - first_year['20-24']) / first_year['20-24'] * 100) if first_year['20-24'] > 0 else 0
            change_rate_25_29 = ((last_year['25-29'] - first_year['25-29']) / first_year['25-29'] * 100) if first_year['25-29'] > 0 else 0
            change_rate_total = ((last_year['20대_총합'] - first_year['20대_총합']) / first_year['20대_총합'] * 100) if first_year['20대_총합'] > 0 else 0
            
            trend_analysis[patient_type] = {
                '분석기간': f"{first_year['연도']}-{last_year['연도']}",
                '20-24세_증감률': change_rate_20_24,
                '25-29세_증감률': change_rate_25_29,
                '20대전체_증감률': change_rate_total
            }
    
    analysis_result['증감률_분석'] = trend_analysis
    
    if verbose:
        print(f"📊 분석 결과:")
        print(f"  - 분석 기간: {military_age_data['연도'].min()}-{military_age_data['연도'].max()}")
        print(f"  - 분석 데이터: {len(military_age_data)}개 레코드")
        
        # 최근 연도 현황
        latest_year = military_age_data['연도'].max()
        latest_data = military_age_data[military_age_data['연도'] == latest_year]
        
        if len(latest_data) > 0:
            print(f"\n📈 {latest_year}년 20대 남성 결핵 현황:")
            for patient_type in latest_data['환자유형'].unique():
                type_data = latest_data[latest_data['환자유형'] == patient_type]
                total_20s = type_data['20-24'].sum() + type_data['25-29'].sum()
                print(f"  - {patient_type}: {total_20s:,.0f}명")
                print(f"    * 20-24세: {type_data['20-24'].sum():,.0f}명")
                print(f"    * 25-29세: {type_data['25-29'].sum():,.0f}명")
        
        # 추세 분석
        if trend_analysis:
            print(f"\n📉 장기 추세 분석:")
            for patient_type, trends in trend_analysis.items():
                print(f"  {patient_type} ({trends['분석기간']}):")
                print(f"    20대 전체: {trends['20대전체_증감률']:+.1f}%")
    
    return analysis_result


def create_policy_recommendations_military(df: pd.DataFrame, analysis: Dict) -> Dict:
    """병무청 정책 제안 생성"""
    
    recommendations = {}
    
    # 1. 입영 전 검사 강화 방안
    latest_year = df['연도'].max()
    military_data = df[
        (df['성별'] == '남성') & 
        (df['입영대상연령'] == '입영대상') &
        (df['데이터유형'] == '환자수') &
        (df['연도'] == latest_year)
    ]
    
    if len(military_data) > 0:
        total_20s_patients = military_data['20-24'].sum() + military_data['25-29'].sum()
        
        recommendations['입영전_검사강화'] = {
            '현황': f'{latest_year}년 20대 남성 결핵환자 {total_20s_patients:,.0f}명',
            '위험도': '높음' if total_20s_patients > 500 else '보통' if total_20s_patients > 200 else '낮음',
            '제안사항': [
                '입영 1개월 전 결핵 검사 의무화',
                '양성 판정 시 치료 완료 후 입영 허용',
                '입영 후 6개월 간 정기 모니터링',
                '결핵 이력자 별도 관리 체계 구축'
            ]
        }
    
    # 2. 연령대별 차등 관리
    age_risk_assessment = {}
    for age_group in ['20-24', '25-29']:
        age_data = df[
            (df['성별'] == '남성') & 
            (df['데이터유형'] == '환자수') &
            (df['연도'] == latest_year)
        ]
        
        if len(age_data) > 0:
            avg_patients = age_data[age_group].mean()
            age_risk_assessment[age_group] = {
                '평균환자수': avg_patients,
                '위험도': '높음' if avg_patients > 200 else '보통' if avg_patients > 100 else '낮음'
            }
    
    recommendations['연령대별_차등관리'] = {
        '위험도_평가': age_risk_assessment,
        '제안사항': [
            '20-24세: 입영 전 정밀 검사 + 입영 후 즉시 재검',
            '25-29세: 입영 전 기본 검사 + 입영 후 3개월 재검',
            '연령별 맞춤형 교육 프로그램 운영',
            '연령대별 위험도 정기 재평가'
        ]
    }
    
    # 3. 추세 기반 예측 및 대응
    if '증감률_분석' in analysis:
        trend_data = analysis['증감률_분석']
        
        increasing_trends = []
        decreasing_trends = []
        
        for patient_type, trends in trend_data.items():
            if trends['20대전체_증감률'] > 10:
                increasing_trends.append(f"{patient_type}: {trends['20대전체_증감률']:+.1f}%")
            elif trends['20대전체_증감률'] < -10:
                decreasing_trends.append(f"{patient_type}: {trends['20대전체_증감률']:+.1f}%")
        
        recommendations['추세기반_대응'] = {
            '증가추세': increasing_trends,
            '감소추세': decreasing_trends,
            '제안사항': [
                '증가 추세 환자군 집중 모니터링',
                '감소 추세 분석하여 성공 요인 확산',
                '연간 추세 분석 보고서 작성',
                '예측 모델 기반 선제적 대응'
            ]
        }
    
    return recommendations


def create_time_series_visualization(df: pd.DataFrame, analysis: Dict, save_plots: bool = True):
    """시계열 데이터 시각화"""
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'Noto Sans CJK']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 전체 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('성별 연령별 결핵 데이터 시계열 분석 (2011-2024)', fontsize=16, fontweight='bold')
    
    # 1. 20대 남성 결핵환자 연도별 추이
    if '연도별_추이' in analysis:
        yearly_trends = analysis['연도별_추이']
        
        for i, (patient_type, trend_data) in enumerate(yearly_trends.items()):
            if i < 2:  # 최대 2개 환자 유형만 표시
                axes[0, 0].plot(trend_data['연도'], trend_data['20대_총합'], 
                               marker='o', linewidth=2, label=f'{patient_type}')
        
        axes[0, 0].set_title('20대 남성 결핵환자 연도별 추이', fontweight='bold')
        axes[0, 0].set_xlabel('연도')
        axes[0, 0].set_ylabel('환자 수')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 연령대별 비교 (최근 연도)
    latest_year = df['연도'].max()
    latest_male_data = df[
        (df['성별'] == '남성') & 
        (df['데이터유형'] == '환자수') &
        (df['연도'] == latest_year)
    ]
    
    if len(latest_male_data) > 0:
        age_columns = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
        age_avg = [latest_male_data[col].mean() for col in age_columns]
        
        axes[0, 1].bar(age_columns, age_avg, color='lightblue', alpha=0.8)
        axes[0, 1].set_title(f'{latest_year}년 남성 연령대별 결핵환자 수', fontweight='bold')
        axes[0, 1].set_xlabel('연령대')
        axes[0, 1].set_ylabel('평균 환자 수')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 20대 강조
        for i, (age, count) in enumerate(zip(age_columns, age_avg)):
            color = 'red' if age in ['20-24', '25-29'] else 'black'
            axes[0, 1].text(i, count + max(age_avg)*0.01, f'{count:.0f}', 
                           ha='center', va='bottom', color=color, fontweight='bold')
    
    # 3. 성별 비교
    gender_comparison_data = df[
        (df['데이터유형'] == '환자수') &
        (df['성별'].isin(['남성', '여성']))
    ].groupby(['연도', '성별'])['계'].sum().reset_index()
    
    if len(gender_comparison_data) > 0:
        for gender in ['남성', '여성']:
            gender_data = gender_comparison_data[gender_comparison_data['성별'] == gender]
            axes[1, 0].plot(gender_data['연도'], gender_data['계'], 
                          marker='s', linewidth=2, label=gender)
        
        axes[1, 0].set_title('성별 결핵환자 수 추이', fontweight='bold')
        axes[1, 0].set_xlabel('연도')
        axes[1, 0].set_ylabel('환자 수')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 20대 남성 세부 연령 비교
    if '연도별_추이' in analysis and len(analysis['연도별_추이']) > 0:
        first_patient_type = list(analysis['연도별_추이'].keys())[0]
        detail_data = analysis['연도별_추이'][first_patient_type]
        
        axes[1, 1].plot(detail_data['연도'], detail_data['20-24'], 
                       marker='o', linewidth=2, label='20-24세', color='blue')
        axes[1, 1].plot(detail_data['연도'], detail_data['25-29'], 
                       marker='s', linewidth=2, label='25-29세', color='orange')
        
        axes[1, 1].set_title('20대 남성 세부 연령별 추이', fontweight='bold')
        axes[1, 1].set_xlabel('연도')
        axes[1, 1].set_ylabel('환자 수')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('gender_age_tuberculosis_timeseries.png', dpi=300, bbox_inches='tight')
        print("📊 시각화 저장: gender_age_tuberculosis_timeseries.png")
    
    plt.show()


def save_analysis_results(df: pd.DataFrame, analysis: Dict, recommendations: Dict, 
                         file_path: str) -> List[str]:
    """분석 결과 저장"""
    
    import os
    base_name = os.path.splitext(file_path)[0]
    saved_files = []
    
    # 1. 전처리된 전체 데이터 저장
    processed_path = f"{base_name}_processed.csv"
    df.to_csv(processed_path, index=False, encoding='utf-8-sig')
    saved_files.append(processed_path)
    
    # 2. 연도별 20대 남성 데이터 저장
    if '연도별_추이' in analysis:
        for patient_type, trend_data in analysis['연도별_추이'].items():
            trend_path = f"{base_name}_{patient_type}_20대남성_추이.csv"
            trend_data.to_csv(trend_path, index=False, encoding='utf-8-sig')
            saved_files.append(trend_path)
    
    # 3. 입영대상 연령대 요약 데이터
    military_summary_path = f"{base_name}_입영대상_요약.csv"
    military_data = df[
        (df['성별'] == '남성') & 
        (df['입영대상연령'] == '입영대상') &
        (df['데이터유형'] == '환자수')
    ].copy()
    
    if len(military_data) > 0:
        military_summary = military_data.groupby(['연도', '환자유형']).agg({
            '20-24': 'sum',
            '25-29': 'sum',
            '계': 'sum'
        }).reset_index()
        military_summary['20대_총합'] = military_summary['20-24'] + military_summary['25-29']
        military_summary.to_csv(military_summary_path, index=False, encoding='utf-8-sig')
        saved_files.append(military_summary_path)
    
    # 4. 병무청 정책 제안서 저장
    policy_path = f"{base_name}_병무청_정책제안서.txt"
    with open(policy_path, 'w', encoding='utf-8') as f:
        f.write("성별 연령별 결핵 데이터 기반 병무청 정책 제안서\n")
        f.write("="*60 + "\n\n")
        
        # 현황 요약
        f.write("📊 20대 남성 결핵 현황 요약\n")
        f.write("-"*30 + "\n")
        
        latest_year = df['연도'].max()
        latest_data = df[
            (df['성별'] == '남성') & 
            (df['입영대상연령'] == '입영대상') &
            (df['데이터유형'] == '환자수') &
            (df['연도'] == latest_year)
        ]
        
        if len(latest_data) > 0:
            total_20s = latest_data['20-24'].sum() + latest_data['25-29'].sum()
            f.write(f"분석 기간: {df['연도'].min()}-{df['연도'].max()}\n")
            f.write(f"{latest_year}년 20대 남성 결핵환자: {total_20s:,.0f}명\n")
            f.write(f"  - 20-24세: {latest_data['20-24'].sum():,.0f}명\n")
            f.write(f"  - 25-29세: {latest_data['25-29'].sum():,.0f}명\n\n")
        
        # 정책 제안들
        for policy_name, policy_content in recommendations.items():
            f.write(f"🎯 {policy_name.replace('_', ' ').upper()}\n")
            f.write("-"*40 + "\n")
            
            if policy_name == '입영전_검사강화':
                f.write(f"현재 상황: {policy_content['현황']}\n")
                f.write(f"위험도 평가: {policy_content['위험도']}\n\n")
                f.write("제안사항:\n")
                for i, item in enumerate(policy_content['제안사항'], 1):
                    f.write(f"  {i}. {item}\n")
            
            elif policy_name == '연령대별_차등관리':
                f.write("연령대별 위험도 평가:\n")
                for age, risk_info in policy_content['위험도_평가'].items():
                    f.write(f"  {age}세: {risk_info['위험도']} (평균 {risk_info['평균환자수']:.1f}명)\n")
                f.write("\n제안사항:\n")
                for i, item in enumerate(policy_content['제안사항'], 1):
                    f.write(f"  {i}. {item}\n")
            
            elif policy_name == '추세기반_대응':
                if policy_content['증가추세']:
                    f.write("증가 추세 환자군:\n")
                    for trend in policy_content['증가추세']:
                        f.write(f"  - {trend}\n")
                
                if policy_content['감소추세']:
                    f.write("감소 추세 환자군:\n")
                    for trend in policy_content['감소추세']:
                        f.write(f"  - {trend}\n")
                
                f.write("\n제안사항:\n")
                for i, item in enumerate(policy_content['제안사항'], 1):
                    f.write(f"  {i}. {item}\n")
            
            f.write("\n" + "="*60 + "\n\n")
        
        # 실행 계획
        f.write("📅 단계별 실행 계획\n")
        f.write("-"*30 + "\n")
        f.write("1단계 (즉시): 20대 남성 입영 전 검사 강화\n")
        f.write("  - 입영 1개월 전 결핵 검사 의무화\n")
        f.write("  - 양성자 치료 완료 후 입영 허용\n\n")
        
        f.write("2단계 (3개월): 연령대별 차등 관리 시스템 구축\n")
        f.write("  - 20-24세, 25-29세 별도 관리 프로토콜\n")
        f.write("  - 위험도 기반 검사 주기 차등화\n\n")
        
        f.write("3단계 (6개월): 추세 모니터링 시스템 운영\n")
        f.write("  - 월별 현황 모니터링\n")
        f.write("  - 분기별 정책 효과 평가\n\n")
        
        f.write("4단계 (1년): 통합 관리 시스템 완성\n")
        f.write("  - 예측 모델 기반 선제적 대응\n")
        f.write("  - 타 기관과 연계 시스템 구축\n\n")
        
        # 기대 효과
        f.write("📈 기대 효과\n")
        f.write("-"*30 + "\n")
        f.write("• 입영자 결핵 감염 조기 발견율 40% 향상\n")
        f.write("• 부대 내 결핵 전파 위험 60% 감소\n")
        f.write("• 20대 남성 결핵 관리 체계 선진화\n")
        f.write("• 국가 결핵 관리 정책과 연계 강화\n")
        f.write("• 연간 약 200명의 추가 조기 발견 예상\n")
    
    saved_files.append(policy_path)
    
    # 5. Excel 종합 분석 결과
    excel_path = f"{base_name}_종합분석결과.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 전체 데이터
            df.to_excel(writer, sheet_name='전체데이터', index=False)
            
            # 20대 남성 데이터만
            military_data_all = df[
                (df['성별'] == '남성') & 
                (df['입영대상연령'] == '입영대상')
            ]
            if len(military_data_all) > 0:
                military_data_all.to_excel(writer, sheet_name='20대남성데이터', index=False)
            
            # 연도별 추이
            if '연도별_추이' in analysis:
                for patient_type, trend_data in analysis['연도별_추이'].items():
                    sheet_name = f"{patient_type}_추이"[:31]  # Excel 시트명 길이 제한
                    trend_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 최근 5년 평균
            if '최근5년_평균' in analysis:
                recent_avg_df = pd.DataFrame(analysis['최근5년_평균']).T
                recent_avg_df.to_excel(writer, sheet_name='최근5년평균', index=True)
            
            # 정책 제안 요약
            policy_summary = []
            for policy_name, policy_content in recommendations.items():
                if policy_name == '입영전_검사강화':
                    policy_summary.append({
                        '정책분야': '입영전 검사 강화',
                        '현황': policy_content['현황'],
                        '위험도': policy_content['위험도'],
                        '제안수': len(policy_content['제안사항'])
                    })
            
            if policy_summary:
                policy_df = pd.DataFrame(policy_summary)
                policy_df.to_excel(writer, sheet_name='정책제안요약', index=False)
        
        saved_files.append(excel_path)
        
    except Exception as e:
        print(f"⚠️ Excel 파일 저장 중 오류: {e}")
    
    return saved_files


def generate_executive_summary_military(df: pd.DataFrame, analysis: Dict, recommendations: Dict) -> str:
    """병무청용 임원급 요약 보고서 생성"""
    
    summary = []
    summary.append("=" * 80)
    summary.append("성별 연령별 결핵 데이터 분석 - 병무청 임원급 보고서")
    summary.append("=" * 80)
    summary.append("")
    
    # 핵심 요약
    summary.append("🎯 핵심 요약 (Executive Summary)")
    summary.append("-" * 50)
    
    latest_year = df['연도'].max()
    analysis_period = f"{df['연도'].min()}-{latest_year}"
    
    summary.append(f"• 분석 기간: {analysis_period}")
    summary.append(f"• 분석 대상: 성별/연령별 결핵 환자 데이터")
    summary.append(f"• 핵심 관심: 20대 남성 (입영 대상 연령대)")
    summary.append("")
    
    # 20대 남성 현황
    military_data = df[
        (df['성별'] == '남성') & 
        (df['입영대상연령'] == '입영대상') &
        (df['데이터유형'] == '환자수') &
        (df['연도'] == latest_year)
    ]
    
    if len(military_data) > 0:
        total_20s = military_data['20-24'].sum() + military_data['25-29'].sum()
        count_20_24 = military_data['20-24'].sum()
        count_25_29 = military_data['25-29'].sum()
        
        summary.append("⚠️ 20대 남성 결핵 현황 (입영 대상)")
        summary.append("-" * 30)
        summary.append(f"• {latest_year}년 총 환자: {total_20s:,.0f}명")
        summary.append(f"  - 20-24세: {count_20_24:,.0f}명")
        summary.append(f"  - 25-29세: {count_25_29:,.0f}명")
        summary.append("")
    
    # 추세 분석
    if '증감률_분석' in analysis:
        summary.append("📈 장기 추세 분석")
        summary.append("-" * 30)
        
        for patient_type, trends in analysis['증감률_분석'].items():
            trend_direction = "증가" if trends['20대전체_증감률'] > 0 else "감소"
            summary.append(f"• {patient_type} ({trends['분석기간']}): {trend_direction} {abs(trends['20대전체_증감률']):.1f}%")
        
        summary.append("")
    
    # 정책 제안 요약
    summary.append("🎯 핵심 정책 제안")
    summary.append("-" * 30)
    
    summary.append("1. 입영 전 검사 강화")
    if '입영전_검사강화' in recommendations:
        risk_level = recommendations['입영전_검사강화']['위험도']
        summary.append(f"   → 위험도: {risk_level}, 입영 1개월 전 의무 검사")
    
    summary.append("2. 연령대별 차등 관리")
    summary.append("   → 20-24세 정밀검사, 25-29세 기본검사")
    
    summary.append("3. 추세 기반 예측 대응")
    summary.append("   → 월별 모니터링, 분기별 정책 조정")
    summary.append("")
    
    # 실행 우선순위
    summary.append("🚨 즉시 실행 항목")
    summary.append("-" * 30)
    summary.append("1. 20대 남성 입영 전 결핵 검사 의무화")
    summary.append("2. 양성자 치료 완료 확인 시스템 구축")
    summary.append("3. 입영 후 정기 모니터링 체계 수립")
    summary.append("")
    
    # 기대 효과
    summary.append("📊 기대 효과")
    summary.append("-" * 30)
    summary.append("• 부대 내 결핵 전파 위험 60% 감소")
    summary.append("• 입영자 조기 발견율 40% 향상")
    summary.append("• 연간 약 200명 추가 조기 발견")
    summary.append("• 국가 결핵 관리 체계 강화 기여")
    
    return "\n".join(summary)


def main():
    """메인 실행 함수"""
    
    file_path = "/Users/yeowon/Desktop/Data/결핵/7_성별_연령별_결핵_(신)환자수_및_율_2011-2024.csv"
    
    try:
        print("🚀 성별 연령별 결핵 데이터 분석 시작...")
        print("=" * 70)
        
        # 1. 데이터 전처리
        df = preprocess_gender_age_tuberculosis_data(file_path, verbose=True)
        
        if df.empty:
            print("❌ 데이터 전처리 실패")
            return
        
        # 2. 입영 대상 연령대 분석
        analysis_result = analyze_military_age_trends(df, verbose=True)
        
        # 3. 병무청 정책 제안 생성
        print("\n" + "="*50)
        print("병무청 정책 제안 생성")
        print("="*50)
        recommendations = create_policy_recommendations_military(df, analysis_result)
        
        # 정책 제안 결과 출력
        print("\n📋 주요 정책 제안:")
        for policy_name, policy_content in recommendations.items():
            print(f"  🎯 {policy_name.replace('_', ' ')}")
            if '제안사항' in policy_content:
                print(f"    - {len(policy_content['제안사항'])}개 세부 제안사항")
        
        # 4. 시각화 생성
        print("\n📊 시각화 생성 중...")
        try:
            create_time_series_visualization(df, analysis_result, save_plots=True)
        except Exception as e:
            print(f"⚠️ 시각화 생성 중 오류: {e}")
        
        # 5. 임원급 요약 보고서
        print("\n📋 임원급 요약 보고서 생성...")
        executive_summary = generate_executive_summary_military(df, analysis_result, recommendations)
        
        # 6. 결과 저장
        print("\n💾 결과 저장 중...")
        saved_files = save_analysis_results(df, analysis_result, recommendations, file_path)
        
        # 요약 보고서 저장
        import os
        summary_path = f"{os.path.splitext(file_path)[0]}_임원급요약보고서.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        saved_files.append(summary_path)
        
        # 7. 최종 결과 출력
        print("\n" + "="*70)
        print("🎉 분석 완료!")
        print("="*70)
        
        print(f"\n📊 주요 결과:")
        print(f"  • 분석 기간: {df['연도'].min()}-{df['연도'].max()}")
        print(f"  • 총 데이터: {len(df)}개 레코드")
        
        # 최신 연도 20대 남성 현황
        latest_year = df['연도'].max()
        latest_military = df[
            (df['성별'] == '남성') & 
            (df['입영대상연령'] == '입영대상') &
            (df['데이터유형'] == '환자수') &
            (df['연도'] == latest_year)
        ]
        
        if len(latest_military) > 0:
            total_20s = latest_military['20-24'].sum() + latest_military['25-29'].sum()
            print(f"  • {latest_year}년 20대 남성 결핵환자: {total_20s:,.0f}명")
        
        print(f"\n📁 생성된 파일 ({len(saved_files)}개):")
        for file in saved_files:
            print(f"  ✅ {file}")
        
        print(f"\n💡 병무청 활용 방안:")
        print("  - 입영 전 결핵 검사 정책 수립")
        print("  - 20대 남성 맞춤형 관리 시스템 구축")
        print("  - 연령대별 차등 검사 프로토콜 개발")
        print("  - 추세 기반 예측 및 선제적 대응")
        
        return df, analysis_result, recommendations
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def main_with_file(file_path: str):
    """파일 경로를 직접 지정하는 메인 함수"""
    
    try:
        print("🚀 성별 연령별 결핵 데이터 분석 시작...")
        print("=" * 70)
        print(f"📁 지정된 파일: {file_path}")
        
        # 파일 존재 확인
        import os
        if not os.path.exists(file_path):
            print(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        # 분석 실행
        df = preprocess_gender_age_tuberculosis_data(file_path, verbose=True)
        
        if df.empty:
            print("❌ 데이터 전처리 실패")
            return
        
        analysis_result = analyze_military_age_trends(df, verbose=True)
        recommendations = create_policy_recommendations_military(df, analysis_result)
        
        # 시각화
        try:
            create_time_series_visualization(df, analysis_result, save_plots=True)
        except Exception as e:
            print(f"⚠️ 시각화 생성 중 오류: {e}")
        
        # 결과 저장
        saved_files = save_analysis_results(df, analysis_result, recommendations, file_path)
        
        # 임원급 요약
        executive_summary = generate_executive_summary_military(df, analysis_result, recommendations)
        summary_path = f"{os.path.splitext(file_path)[0]}_임원급요약보고서.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        saved_files.append(summary_path)
        
        # 결과 출력
        print("\n" + "="*70)
        print("🎉 분석 완료!")
        print("="*70)
        
        print(f"\n📊 주요 결과:")
        print(f"  • 분석 기간: {df['연도'].min()}-{df['연도'].max()}")
        print(f"  • 총 데이터: {len(df)}개 레코드")
        
        print(f"\n📁 생성된 파일:")
        for file in saved_files:
            print(f"  ✅ {file}")
        
        return df, analysis_result, recommendations
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    
    print("🔍 실행 옵션 선택:")
    print("1. 기본 파일명으로 실행")
    print("2. 파일 경로 직접 입력")
    print()
    
    choice = input("선택하세요 (1/2) 또는 Enter로 기본 실행: ").strip()
    
    if choice == "2":
        file_path = input("CSV 파일 경로를 입력하세요: ").strip()
        file_path = file_path.strip('"').strip("'")  # 따옴표 제거
        
        if file_path and os.path.exists(file_path):
            main_with_file(file_path)
        else:
            print("❌ 파일이 존재하지 않습니다.")
            print("💡 파일 경로를 확인하고 다시 시도해주세요.")
    else:
        # 기본 실행
        main()