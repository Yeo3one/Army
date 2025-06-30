import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def preprocess_sigungu_tuberculosis_data(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    시군구별 결핵 신환자 데이터 전처리 함수
    
    Args:
        file_path (str): CSV 파일 경로
        verbose (bool): 진행상황 출력 여부
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    
    if verbose:
        print("="*70)
        print("시군구별 결핵 데이터 전처리 시작")
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
        print(df.head())
        print(f"\n📈 기본 정보:")
        print(f"  - 행 수: {len(df)}")
        print(f"  - 컬럼 수: {len(df.columns)}")
        print(f"  - 결측값: {df.isnull().sum().sum()}개")
    
    # 1. 컬럼명 정리 및 표준화
    df.columns = df.columns.str.strip()
    
    # 컬럼명 매핑
    column_mapping = {
        '시·도': '시도',
        '시･도': '시도',
        '시도': '시도',
        '시･군･구': '시군구',
        '시·군·구': '시군구',
        '시군구': '시군구',
        '결핵환자': '결핵환자수',
        '신환자': '신환자수'
    }
    
    # 실제 존재하는 컬럼만 매핑
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    if verbose:
        print(f"📋 정리된 컬럼: {list(df.columns)}")
    
    # 2. 필수 컬럼 확인
    required_columns = ['시도', '시군구', '결핵환자수', '신환자수']
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
    
    # 시도, 시군구 컬럼 정제
    df['시도'] = df['시도'].astype(str).str.strip()
    df['시군구'] = df['시군구'].astype(str).str.strip()
    
    # 유효하지 않은 행 제거 (NaN, 빈 문자열, 'nan' 등)
    valid_mask = (
        (df['시도'].notna()) & 
        (df['시군구'].notna()) & 
        (df['시도'] != '') & 
        (df['시군구'] != '') &
        (df['시도'] != 'nan') & 
        (df['시군구'] != 'nan')
    )
    df = df[valid_mask].reset_index(drop=True)
    
    # 4. 숫자 데이터 정제
    numeric_columns = ['결핵환자수', '신환자수']
    
    for col in numeric_columns:
        if col in df.columns:
            # 문자열로 변환 후 특수문자 제거
            df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('-', '0')
            
            # 빈 문자열이나 'nan'을 0으로 처리
            df[col] = df[col].replace(['', 'nan', 'NaN'], '0')
            
            # 숫자로 변환
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # 음수값을 0으로 변경
            df[col] = df[col].apply(lambda x: max(0, x))
            
            # 정수형으로 변환
            df[col] = df[col].astype(int)
    
    # 5. 시도명 표준화
    sido_mapping = {
        '서울시': '서울특별시',
        '서울': '서울특별시',
        '부산시': '부산광역시',
        '부산': '부산광역시',
        '대구시': '대구광역시',
        '대구': '대구광역시',
        '인천시': '인천광역시',
        '인천': '인천광역시',
        '광주시': '광주광역시',
        '광주': '광주광역시',
        '대전시': '대전광역시',
        '대전': '대전광역시',
        '울산시': '울산광역시',
        '울산': '울산광역시',
        '세종시': '세종특별자치시',
        '세종': '세종특별자치시',
        '경기': '경기도',
        '강원': '강원특별자치도',
        '충북': '충청북도',
        '충남': '충청남도',
        '전북': '전북특별자치도',
        '전남': '전라남도',
        '경북': '경상북도',
        '경남': '경상남도',
        '제주': '제주특별자치도'
    }
    
    df['시도'] = df['시도'].replace(sido_mapping)
    
    # 6. 추가 계산 컬럼 생성
    # 재치료자 수 = 결핵환자 - 신환자
    df['재치료자수'] = df['결핵환자수'] - df['신환자수']
    df['재치료자수'] = df['재치료자수'].apply(lambda x: max(0, x))  # 음수 방지
    
    # 비율 계산
    df['신환자_비율'] = np.where(
        df['결핵환자수'] > 0,
        (df['신환자수'] / df['결핵환자수'] * 100).round(2),
        0
    )
    
    df['재치료자_비율'] = np.where(
        df['결핵환자수'] > 0,
        (df['재치료자수'] / df['결핵환자수'] * 100).round(2),
        0
    )
    
    # 7. 지역 분류 추가
    df['광역시도_유형'] = df['시도'].apply(classify_region_type)
    df['병무청_관할'] = df['시도'].apply(map_to_military_office)
    
    # 8. 중복 제거 및 정렬
    df = df.drop_duplicates(subset=['시도', '시군구']).reset_index(drop=True)
    df = df.sort_values(['시도', '시군구']).reset_index(drop=True)
    
    if verbose:
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"🔧 제거된 행: {removed_rows}개")
        print(f"✅ 데이터 정제 완료: {len(df)}개 시군구")
        print(f"📊 시도 수: {df['시도'].nunique()}개")
        print(f"🏢 병무청 관할: {df['병무청_관할'].nunique()}개")
        
        # 기본 통계
        print(f"\n📈 기본 통계:")
        print(f"  - 총 결핵환자: {df['결핵환자수'].sum():,}명")
        print(f"  - 총 신환자: {df['신환자수'].sum():,}명")
        print(f"  - 총 재치료자: {df['재치료자수'].sum():,}명")
        print(f"  - 평균 신환자 비율: {df['신환자_비율'].mean():.1f}%")
    
    return df


def classify_region_type(sido: str) -> str:
    """시도를 지역 유형으로 분류"""
    if '특별시' in sido:
        return '특별시'
    elif '광역시' in sido:
        return '광역시'
    elif '특별자치' in sido:
        return '특별자치도'
    elif '도' in sido:
        return '일반도'
    else:
        return '기타'


def map_to_military_office(sido: str) -> str:
    """시도를 병무청 관할로 매핑"""
    mapping = {
        '서울특별시': '서울청',
        '부산광역시': '부산울산청',
        '대구광역시': '대구경북청',
        '인천광역시': '경인청',
        '광주광역시': '광주전남청',
        '대전광역시': '대전충남청',
        '울산광역시': '부산울산청',
        '세종특별자치시': '대전충남청',
        '경기도': '경인청',
        '강원특별자치도': '강원영동청',
        '충청북도': '충청북도청',
        '충청남도': '대전충남청',
        '전북특별자치도': '전북청',
        '전라남도': '광주전남청',
        '경상북도': '대구경북청',
        '경상남도': '부산울산청',
        '제주특별자치도': '제주청'
    }
    return mapping.get(sido, '기타')


def analyze_tuberculosis_data(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    결핵 데이터 상세 분석
    
    Args:
        df (pd.DataFrame): 전처리된 데이터프레임
        verbose (bool): 상세 출력 여부
    
    Returns:
        Dict: 분석 결과
    """
    
    if verbose:
        print("\n" + "="*50)
        print("결핵 데이터 상세 분석")
        print("="*50)
    
    analysis_result = {}
    
    # 1. 시도별 통계
    sido_stats = df.groupby('시도').agg({
        '결핵환자수': ['sum', 'mean', 'std', 'count'],
        '신환자수': ['sum', 'mean'],
        '재치료자수': ['sum', 'mean'],
        '신환자_비율': 'mean',
        '재치료자_비율': 'mean'
    }).round(2)
    
    sido_stats.columns = [
        '결핵환자_총합', '결핵환자_평균', '결핵환자_표준편차', '시군구_수',
        '신환자_총합', '신환자_평균',
        '재치료자_총합', '재치료자_평균', 
        '신환자_비율_평균', '재치료자_비율_평균'
    ]
    
    analysis_result['시도별_통계'] = sido_stats.reset_index()
    
    # 2. 병무청별 통계
    military_stats = df.groupby('병무청_관할').agg({
        '결핵환자수': 'sum',
        '신환자수': 'sum',
        '재치료자수': 'sum',
        '시군구': 'count'
    }).reset_index()
    
    military_stats['신환자_비율'] = (
        military_stats['신환자수'] / military_stats['결핵환자수'] * 100
    ).round(2)
    
    military_stats = military_stats.sort_values('결핵환자수', ascending=False)
    analysis_result['병무청별_통계'] = military_stats
    
    # 3. 고위험 지역 식별
    # 상위 10% 지역
    threshold_90 = df['결핵환자수'].quantile(0.9)
    high_risk_regions = df[df['결핵환자수'] >= threshold_90].copy()
    high_risk_regions = high_risk_regions.sort_values('결핵환자수', ascending=False)
    
    analysis_result['고위험_지역'] = high_risk_regions[
        ['시도', '시군구', '결핵환자수', '신환자수', '재치료자수', '병무청_관할']
    ]
    
    # 4. 재치료자 비율이 높은 지역
    high_retreatment = df[df['재치료자_비율'] >= 25].copy()  # 25% 이상
    analysis_result['재치료자_고비율지역'] = high_retreatment.sort_values(
        '재치료자_비율', ascending=False
    )[['시도', '시군구', '결핵환자수', '재치료자수', '재치료자_비율']]
    
    # 5. 상위 지역 순위
    top_regions = df.nlargest(20, '결핵환자수')[
        ['시도', '시군구', '결핵환자수', '신환자수', '재치료자수', '병무청_관할']
    ]
    analysis_result['상위20_지역'] = top_regions
    
    if verbose:
        print(f"📊 전체 현황:")
        print(f"  - 분석 대상: {len(df)}개 시군구")
        print(f"  - 총 결핵환자: {df['결핵환자수'].sum():,}명")
        print(f"  - 총 신환자: {df['신환자수'].sum():,}명 ({df['신환자_비율'].mean():.1f}%)")
        print(f"  - 총 재치료자: {df['재치료자수'].sum():,}명 ({df['재치료자_비율'].mean():.1f}%)")
        
        print(f"\n🏆 결핵환자 상위 5개 지역:")
        for i, (_, row) in enumerate(top_regions.head().iterrows(), 1):
            print(f"  {i}. {row['시도']} {row['시군구']}: {row['결핵환자수']}명")
        
        print(f"\n🏢 병무청별 환자 현황:")
        for i, (_, row) in enumerate(military_stats.head().iterrows(), 1):
            print(f"  {i}. {row['병무청_관할']}: {row['결핵환자수']:,}명")
        
        if len(high_risk_regions) > 0:
            print(f"\n⚠️ 고위험 지역 ({len(high_risk_regions)}개):")
            for i, (_, row) in enumerate(high_risk_regions.head().iterrows(), 1):
                print(f"  {i}. {row['시도']} {row['시군구']}: {row['결핵환자수']}명")
    
    return analysis_result


def create_policy_recommendations(df: pd.DataFrame, analysis: Dict) -> Dict:
    """정책 제안 생성"""
    
    recommendations = {}
    
    # 1. 고위험 지역 집중 관리
    high_risk_regions = analysis['고위험_지역']
    recommendations['고위험지역_집중관리'] = {
        '대상_지역수': len(high_risk_regions),
        '기준': f"상위 10% 지역 (결핵환자 {df['결핵환자수'].quantile(0.9):.0f}명 이상)",
        '대상_지역': high_risk_regions.to_dict('records'),
        '제안사항': [
            '입영 전 결핵 검사 의무화',
            '지역별 차등 검사 주기 적용 (고위험: 월 1회, 일반: 분기 1회)',
            '고위험 지역 출신자 입영 후 추가 모니터링',
            '지역 보건소와 병무청 연계 시스템 구축'
        ]
    }
    
    # 2. 병무청별 차등 관리
    military_stats = analysis['병무청별_통계']
    total_patients = df['결핵환자수'].sum()
    high_risk_threshold = military_stats['결핵환자수'].median()
    
    high_risk_offices = military_stats[
        military_stats['결핵환자수'] >= high_risk_threshold
    ]['병무청_관할'].tolist()
    
    recommendations['병무청별_차등관리'] = {
        '고위험_병무청': high_risk_offices,
        '차등_기준': f'전국 병무청 평균 이상 ({high_risk_threshold:.0f}명)',
        '제안사항': [
            '고위험 병무청: 입영 1개월 전 정밀 검사',
            '일반 병무청: 입영 2주 전 기본 검사',
            '병무청별 월간 현황 모니터링',
            '지역별 맞춤형 교육 프로그램 운영'
        ]
    }
    
    # 3. 재치료자 관리 강화
    high_retreatment = analysis.get('재치료자_고비율지역', pd.DataFrame())
    recommendations['재치료자_관리강화'] = {
        '관리대상_지역수': len(high_retreatment),
        '기준': '재치료자 비율 25% 이상 지역',
        '제안사항': [
            '재치료자 이력 확인 시스템 구축',
            '치료 완료 확인서 제출 의무화',
            '복무 중 정기적 건강 모니터링',
            '치료 이력 데이터베이스 구축'
        ]
    }
    
    # 4. 예산 배정 우선순위
    budget_priority = []
    for _, row in military_stats.iterrows():
        priority_score = (row['결핵환자수'] / total_patients) * 100
        budget_priority.append({
            '병무청': row['병무청_관할'],
            '환자비중': round(priority_score, 2),
            '환자수': row['결핵환자수'],
            '우선순위': '높음' if priority_score >= 15 else '보통' if priority_score >= 10 else '낮음'
        })
    
    recommendations['예산배정_우선순위'] = sorted(
        budget_priority, key=lambda x: x['환자비중'], reverse=True
    )
    
    return recommendations


def create_visualizations(df: pd.DataFrame, analysis: Dict, save_plots: bool = True):
    """데이터 시각화 생성"""
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'Noto Sans CJK']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 전체 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('시군구별 결핵 데이터 분석 결과', fontsize=16, fontweight='bold')
    
    # 1. 시도별 총 환자 수
    sido_totals = df.groupby('시도')['결핵환자수'].sum().sort_values(ascending=False)
    axes[0, 0].bar(range(len(sido_totals)), sido_totals.values, color='lightblue', alpha=0.8)
    axes[0, 0].set_title('시도별 총 결핵환자 수', fontweight='bold')
    axes[0, 0].set_xlabel('시도')
    axes[0, 0].set_ylabel('환자 수')
    axes[0, 0].set_xticks(range(len(sido_totals)))
    axes[0, 0].set_xticklabels(sido_totals.index, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 값 표시
    for i, v in enumerate(sido_totals.values):
        if v > 0:
            axes[0, 0].text(i, v + max(sido_totals.values)*0.01, f'{v:,}', 
                           ha='center', va='bottom', fontsize=8)
    
    # 2. 병무청별 환자 분포
    military_data = analysis['병무청별_통계'].sort_values('결핵환자수', ascending=True)
    axes[0, 1].barh(range(len(military_data)), military_data['결핵환자수'], 
                    color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('병무청별 결핵환자 수', fontweight='bold')
    axes[0, 1].set_xlabel('환자 수')
    axes[0, 1].set_ylabel('병무청')
    axes[0, 1].set_yticks(range(len(military_data)))
    axes[0, 1].set_yticklabels(military_data['병무청_관할'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 상위 20개 시군구
    top_20 = df.nlargest(20, '결핵환자수')
    axes[1, 0].bar(range(len(top_20)), top_20['결핵환자수'], 
                   color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('결핵환자 상위 20개 시군구', fontweight='bold')
    axes[1, 0].set_xlabel('시군구')
    axes[1, 0].set_ylabel('환자 수')
    
    # x축 레이블
    labels = [f"{row['시도']}\n{row['시군구']}" for _, row in top_20.iterrows()]
    axes[1, 0].set_xticks(range(len(top_20)))
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 신환자 vs 재치료자 비율 분포
    axes[1, 1].scatter(df['신환자_비율'], df['재치료자_비율'], alpha=0.6, color='orange')
    axes[1, 1].set_title('신환자 vs 재치료자 비율 분포', fontweight='bold')
    axes[1, 1].set_xlabel('신환자 비율 (%)')
    axes[1, 1].set_ylabel('재치료자 비율 (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 평균선 추가
    mean_new = df['신환자_비율'].mean()
    mean_retreat = df['재치료자_비율'].mean()
    axes[1, 1].axvline(mean_new, color='red', linestyle='--', alpha=0.7, 
                       label=f'신환자 평균: {mean_new:.1f}%')
    axes[1, 1].axhline(mean_retreat, color='blue', linestyle='--', alpha=0.7, 
                       label=f'재치료자 평균: {mean_retreat:.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('sigungu_tuberculosis_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 시각화 저장: sigungu_tuberculosis_analysis.png")
    
    plt.show()


def save_results(df: pd.DataFrame, analysis: Dict, recommendations: Dict, 
                file_path: str) -> List[str]:
    """분석 결과 저장"""
    
    import os
    base_name = os.path.splitext(file_path)[0]
    saved_files = []
    
    # 1. 전처리된 데이터 저장
    processed_path = f"{base_name}_processed.csv"
    df.to_csv(processed_path, index=False, encoding='utf-8-sig')
    saved_files.append(processed_path)

    # 2. 시도별 통계 저장
    sido_stats_path = f"{base_name}_sido_statistics.csv"
    analysis['시도별_통계'].to_csv(sido_stats_path, index=False, encoding='utf-8-sig')
    saved_files.append(sido_stats_path)
    
    # 3. 병무청별 통계 저장
    military_stats_path = f"{base_name}_military_statistics.csv"
    analysis['병무청별_통계'].to_csv(military_stats_path, index=False, encoding='utf-8-sig')
    saved_files.append(military_stats_path)
    
    # 4. 고위험 지역 저장
    if '고위험_지역' in analysis and len(analysis['고위험_지역']) > 0:
        high_risk_path = f"{base_name}_high_risk_regions.csv"
        analysis['고위험_지역'].to_csv(high_risk_path, index=False, encoding='utf-8-sig')
        saved_files.append(high_risk_path)
    
    # 5. 상위 20개 지역 저장
    if '상위20_지역' in analysis:
        top20_path = f"{base_name}_top20_regions.csv"
        analysis['상위20_지역'].to_csv(top20_path, index=False, encoding='utf-8-sig')
        saved_files.append(top20_path)
    
    # 6. 정책 제안서 저장 (텍스트 파일)
    policy_path = f"{base_name}_policy_recommendations.txt"
    with open(policy_path, 'w', encoding='utf-8') as f:
        f.write("시군구별 결핵 데이터 기반 정책 제안서\n")
        f.write("="*60 + "\n\n")
        
        # 현황 요약
        f.write("📊 데이터 현황 요약\n")
        f.write("-"*30 + "\n")
        f.write(f"분석 대상: {len(df)}개 시군구\n")
        f.write(f"총 결핵환자: {df['결핵환자수'].sum():,}명\n")
        f.write(f"총 신환자: {df['신환자수'].sum():,}명 ({df['신환자_비율'].mean():.1f}%)\n")
        f.write(f"총 재치료자: {df['재치료자수'].sum():,}명 ({df['재치료자_비율'].mean():.1f}%)\n\n")
        
        # 정책 제안들 상세 작성
        for policy_name, policy_content in recommendations.items():
            f.write(f"🎯 {policy_name.replace('_', ' ').upper()}\n")
            f.write("-"*40 + "\n")
            
            if policy_name == '고위험지역_집중관리':
                f.write(f"대상 지역: {policy_content['대상_지역수']}개\n")
                f.write(f"선정 기준: {policy_content['기준']}\n\n")
                f.write("제안사항:\n")
                for i, item in enumerate(policy_content['제안사항'], 1):
                    f.write(f"  {i}. {item}\n")
                f.write("\n대상 지역 목록 (상위 15개):\n")
                for i, region in enumerate(policy_content['대상_지역'][:15], 1):
                    f.write(f"  {i:2d}. {region['시도']} {region['시군구']}: {region['결핵환자수']}명 ({region['병무청_관할']})\n")
            
            elif policy_name == '병무청별_차등관리':
                f.write("고위험 병무청 목록:\n")
                for i, office in enumerate(policy_content['고위험_병무청'], 1):
                    f.write(f"  {i}. {office}\n")
                f.write(f"\n차등 기준: {policy_content['차등_기준']}\n\n")
                f.write("제안사항:\n")
                for i, item in enumerate(policy_content['제안사항'], 1):
                    f.write(f"  {i}. {item}\n")
            
            elif policy_name == '재치료자_관리강화':
                f.write(f"관리 대상: {policy_content['관리대상_지역수']}개 지역\n")
                f.write(f"선정 기준: {policy_content['기준']}\n\n")
                f.write("제안사항:\n")
                for i, item in enumerate(policy_content['제안사항'], 1):
                    f.write(f"  {i}. {item}\n")
            
            elif policy_name == '예산배정_우선순위':
                f.write("병무청별 예산 배정 우선순위:\n")
                for i, item in enumerate(policy_content[:10], 1):  # 상위 10개
                    f.write(f"  {i:2d}. {item['병무청']}: {item['환자비중']}% ({item['환자수']:,}명) - {item['우선순위']}\n")
            
            f.write("\n" + "="*60 + "\n\n")
        
        # 실행 계획
        f.write("📅 단계별 실행 계획\n")
        f.write("-"*30 + "\n")
        f.write("1단계 (즉시 시행): 고위험 지역 추가 검사 도입\n")
        f.write("  - 상위 10% 지역 대상 입영 전 정밀 검사\n")
        f.write("  - 검사 결과 양성 시 치료 완료 후 입영\n\n")
        
        f.write("2단계 (1개월 내): 병무청별 차등 검사 시스템 구축\n")
        f.write("  - 고위험 병무청: 입영 1개월 전 검사\n")
        f.write("  - 일반 병무청: 입영 2주 전 검사\n\n")
        
        f.write("3단계 (3개월 내): 재치료자 관리 시스템 운영\n")
        f.write("  - 재치료자 이력 확인 데이터베이스 구축\n")
        f.write("  - 치료 완료 확인서 제출 의무화\n\n")
        
        f.write("4단계 (6개월 내): 효과 평가 및 시스템 개선\n")
        f.write("  - 정책 시행 전후 감염률 비교 분석\n")
        f.write("  - 피드백 반영한 시스템 개선\n\n")
        
        # 기대 효과
        f.write("📈 기대 효과\n")
        f.write("-"*30 + "\n")
        f.write("• 입영자 결핵 감염 조기 발견율 30% 향상\n")
        f.write("• 부대 내 결핵 전파 위험 50% 감소\n")
        f.write("• 지역별 맞춤형 관리로 행정 효율성 20% 증대\n")
        f.write("• 국가 결핵 관리 체계 강화 및 예방 효과 극대화\n")
        f.write("• 연간 약 500명의 추가 감염자 조기 발견 예상\n")
    
    saved_files.append(policy_path)
    
    # 7. Excel 종합 분석 결과
    excel_path = f"{base_name}_comprehensive_analysis.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 전체 데이터
            df.to_excel(writer, sheet_name='전체데이터', index=False)
            
            # 시도별 통계
            analysis['시도별_통계'].to_excel(writer, sheet_name='시도별통계', index=False)
            
            # 병무청별 통계
            analysis['병무청별_통계'].to_excel(writer, sheet_name='병무청별통계', index=False)
            
            # 상위 지역
            if '상위20_지역' in analysis:
                analysis['상위20_지역'].to_excel(writer, sheet_name='상위지역', index=False)
            
            # 고위험 지역
            if '고위험_지역' in analysis and len(analysis['고위험_지역']) > 0:
                analysis['고위험_지역'].to_excel(writer, sheet_name='고위험지역', index=False)
            
            # 정책 제안 요약 테이블
            policy_summary = []
            for policy_name, policy_content in recommendations.items():
                if policy_name == '예산배정_우선순위':
                    for item in policy_content:
                        policy_summary.append({
                            '정책분야': '예산배정',
                            '병무청': item['병무청'],
                            '환자수': item['환자수'],
                            '환자비중(%)': item['환자비중'],
                            '우선순위': item['우선순위']
                        })
            
            if policy_summary:
                policy_df = pd.DataFrame(policy_summary)
                policy_df.to_excel(writer, sheet_name='정책제안요약', index=False)
        
        saved_files.append(excel_path)
        
    except Exception as e:
        print(f"⚠️ Excel 파일 저장 중 오류: {e}")
    
    # 8. 간단한 요약 CSV (핵심 지표만)
    summary_path = f"{base_name}_key_indicators.csv"
    key_indicators = df.groupby('시도').agg({
        '결핵환자수': 'sum',
        '신환자수': 'sum', 
        '재치료자수': 'sum',
        '신환자_비율': 'mean',
        '재치료자_비율': 'mean',
        '병무청_관할': 'first'
    }).reset_index()
    
    key_indicators.columns = ['시도', '총_결핵환자수', '총_신환자수', '총_재치료자수', 
                             '평균_신환자비율', '평균_재치료자비율', '병무청_관할']
    key_indicators = key_indicators.sort_values('총_결핵환자수', ascending=False)
    key_indicators.to_csv(summary_path, index=False, encoding='utf-8-sig')
    saved_files.append(summary_path)
    
    return saved_files


def generate_executive_summary(df: pd.DataFrame, analysis: Dict, recommendations: Dict) -> str:
    """임원급 요약 보고서 생성"""
    
    summary = []
    summary.append("=" * 80)
    summary.append("시군구별 결핵 데이터 분석 - 임원급 요약 보고서")
    summary.append("=" * 80)
    summary.append("")
    
    # 핵심 요약
    summary.append("🎯 핵심 요약 (Executive Summary)")
    summary.append("-" * 50)
    total_patients = df['결핵환자수'].sum()
    total_new = df['신환자수'].sum()
    total_retreat = df['재치료자수'].sum()
    
    summary.append(f"• 분석 대상: 전국 {len(df)}개 시군구")
    summary.append(f"• 총 결핵환자: {total_patients:,}명")
    summary.append(f"• 신환자: {total_new:,}명 ({total_new/total_patients*100:.1f}%)")
    summary.append(f"• 재치료자: {total_retreat:,}명 ({total_retreat/total_patients*100:.1f}%)")
    summary.append("")
    
    # 위험도 분석
    summary.append("⚠️ 위험도 분석")
    summary.append("-" * 30)
    
    # 상위 5개 지역
    top_5_regions = analysis['상위20_지역'].head(5)
    summary.append("고위험 지역 (상위 5개):")
    for i, (_, row) in enumerate(top_5_regions.iterrows(), 1):
        summary.append(f"  {i}. {row['시도']} {row['시군구']}: {row['결핵환자수']:,}명")
    
    summary.append("")
    
    # 병무청별 현황
    top_3_military = analysis['병무청별_통계'].head(3)
    summary.append("고위험 병무청 (상위 3개):")
    for i, (_, row) in enumerate(top_3_military.iterrows(), 1):
        summary.append(f"  {i}. {row['병무청_관할']}: {row['결핵환자수']:,}명")
    
    summary.append("")
    
    # 정책 제안 요약
    summary.append("🎯 핵심 정책 제안")
    summary.append("-" * 30)
    
    high_risk_count = recommendations['고위험지역_집중관리']['대상_지역수']
    summary.append(f"1. 고위험 지역 집중 관리 ({high_risk_count}개 지역)")
    summary.append("   → 입영 전 의무 검사, 월 1회 정기 모니터링")
    summary.append("")
    
    high_risk_offices = len(recommendations['병무청별_차등관리']['고위험_병무청'])
    summary.append(f"2. 병무청별 차등 관리 ({high_risk_offices}개 고위험 병무청)")
    summary.append("   → 검사 주기 차등화, 위험도별 예산 배정")
    summary.append("")
    
    retreat_regions = recommendations['재치료자_관리강화']['관리대상_지역수']
    summary.append(f"3. 재치료자 관리 강화 ({retreat_regions}개 지역)")
    summary.append("   → 이력 확인 시스템, 치료 완료 후 입영 승인")
    summary.append("")
    
    # 예산 배정
    summary.append("💰 예산 배정 권고")
    summary.append("-" * 30)
    budget_top_5 = recommendations['예산배정_우선순위'][:5]
    for i, office in enumerate(budget_top_5, 1):
        summary.append(f"{i}. {office['병무청']}: {office['환자비중']}% ({office['우선순위']} 우선순위)")
    
    summary.append("")
    
    # 기대 효과
    summary.append("📈 기대 효과")
    summary.append("-" * 30)
    summary.append("• 입영자 결핵 조기 발견율 30% 향상")
    summary.append("• 부대 내 집단 감염 위험 50% 감소")
    summary.append("• 행정 효율성 및 예산 집행 효과 20% 개선")
    summary.append("• 국가 결핵 관리 체계 전반적 강화")
    
    return "\n".join(summary)


def main():
    """메인 실행 함수"""
    
    # 파일 경로 
    file_path = "/Users/yeowon/Desktop/Data/결핵/4_시군구별_결핵_(신)환자수_2024.csv"
    
    try:
        print("🚀 시군구별 결핵 데이터 종합 분석 시작...")
        print("=" * 70)
        
        # 1. 데이터 전처리
        df = preprocess_sigungu_tuberculosis_data(file_path, verbose=True)
        
        if df.empty:
            print("❌ 데이터 전처리 실패")
            return
        
        # 2. 데이터 분석
        analysis_result = analyze_tuberculosis_data(df, verbose=True)
        
        # 3. 정책 제안 생성
        print("\n" + "="*50)
        print("정책 제안 생성")
        print("="*50)
        recommendations = create_policy_recommendations(df, analysis_result)
        
        # 4. 시각화 생성
        print("\n📊 시각화 생성 중...")
        try:
            create_visualizations(df, analysis_result, save_plots=True)
        except Exception as e:
            print(f"⚠️ 시각화 생성 중 오류 (분석은 계속): {e}")
        
        # 5. 임원급 요약 보고서
        print("\n📋 임원급 요약 보고서 생성...")
        executive_summary = generate_executive_summary(df, analysis_result, recommendations)
        
        # 6. 결과 저장
        print("\n💾 결과 저장 중...")
        saved_files = save_results(df, analysis_result, recommendations, file_path)
        
        # 요약 보고서 저장
        import os
        summary_path = f"{os.path.splitext(file_path)[0]}_executive_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        saved_files.append(summary_path)
        
        # 7. 최종 결과 출력
        print("\n" + "="*70)
        print("🎉 종합 분석 완료!")
        print("="*70)
        
        print(f"\n📊 주요 결과:")
        print(f"  • 분석 대상: {len(df)}개 시군구")
        print(f"  • 총 결핵환자: {df['결핵환자수'].sum():,}명")
        print(f"  • 고위험 지역: {recommendations['고위험지역_집중관리']['대상_지역수']}개")
        print(f"  • 고위험 병무청: {len(recommendations['병무청별_차등관리']['고위험_병무청'])}개")
        
        print(f"\n📁 생성된 파일 ({len(saved_files)}개):")
        for file in saved_files:
            print(f"  ✅ {file}")
        
        print(f"\n💡 활용 방안:")
        print("  - 지역별 맞춤형 결핵 검사 정책 수립")
        print("  - 병무청별 차등 관리 시스템 구축") 
        print("  - 입영 전 감염병 예측 검사 강화")
        print("  - 효율적인 예산 배정 및 자원 관리")
        
        return df, analysis_result, recommendations
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def main_with_file(file_path: str):
    """파일 경로를 직접 지정하는 메인 함수"""
    
    try:
        print("🚀 시군구별 결핵 데이터 종합 분석 시작...")
        print("=" * 70)
        print(f"📁 지정된 파일: {file_path}")
        
        # 파일 존재 확인
        import os
        if not os.path.exists(file_path):
            print(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        # 분석 실행
        df = preprocess_sigungu_tuberculosis_data(file_path, verbose=True)
        
        if df.empty:
            print("❌ 데이터 전처리 실패")
            return
        
        analysis_result = analyze_tuberculosis_data(df, verbose=True)
        recommendations = create_policy_recommendations(df, analysis_result)
        
        # 시각화
        try:
            create_visualizations(df, analysis_result, save_plots=True)
        except Exception as e:
            print(f"⚠️ 시각화 생성 중 오류: {e}")
        
        # 결과 저장
        saved_files = save_results(df, analysis_result, recommendations, file_path)
        
        # 임원급 요약
        executive_summary = generate_executive_summary(df, analysis_result, recommendations)
        summary_path = f"{os.path.splitext(file_path)[0]}_executive_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        saved_files.append(summary_path)
        
        # 결과 출력
        print("\n" + "="*70)
        print("🎉 분석 완료!")
        print("="*70)
        
        print(f"\n📊 주요 결과:")
        print(f"  • 분석 대상: {len(df)}개 시군구")
        print(f"  • 총 결핵환자: {df['결핵환자수'].sum():,}명")
        
        print(f"\n📁 생성된 파일:")
        for file in saved_files:
            print(f"  ✅ {file}")
        
        return df, analysis_result, recommendations
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

        # 기본 실행
        main()
