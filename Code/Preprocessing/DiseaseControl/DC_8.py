import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows의 경우)
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TuberculosisAnalyzer:
    def __init__(self, file_path):
        """결핵 데이터 분석기 초기화"""
        self.file_path = file_path
        self.raw_data = None
        self.new_cases_data = None
        self.incidence_rate_data = None
        self.regions = ['전국', '서울', '부산', '대구', '인천', '광주', '대전', '울산', 
                       '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
    
    def load_data(self):
        """데이터 로드 및 기본 전처리"""
        print("📁 데이터 로딩 중...")
        
        # CSV 파일 읽기 (인코딩 자동 감지)
        try:
            self.raw_data = pd.read_csv(self.file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.raw_data = pd.read_csv(self.file_path, encoding='cp949')
            except UnicodeDecodeError:
                self.raw_data = pd.read_csv(self.file_path, encoding='euc-kr')
        
        print(f"✅ 데이터 로드 완료: {self.raw_data.shape}")
        print(f"📋 컬럼: {list(self.raw_data.columns)}")
        
        return self.raw_data
    
    def preprocess_data(self):
        """데이터 전처리"""
        print("\n🔧 데이터 전처리 시작...")
        
        # 컬럼명 정리 (공백 제거)
        self.raw_data.columns = self.raw_data.columns.str.strip()
        
        # 연도 컬럼 확인 및 정리
        if '연도' in self.raw_data.columns:
            # 유효한 연도만 필터링
            self.raw_data = self.raw_data[
                (self.raw_data['연도'].notna()) & 
                (self.raw_data['연도'] >= 2011) & 
                (self.raw_data['연도'] <= 2024)
            ].copy()
            
            self.raw_data['연도'] = self.raw_data['연도'].astype(int)
        
        # 숫자 데이터 정리 (쉼표 제거 및 숫자 변환)
        for region in self.regions:
            if region in self.raw_data.columns:
                # 문자열인 경우 쉼표 제거
                self.raw_data[region] = self.raw_data[region].astype(str).str.replace(',', '')
                # 숫자 변환 (변환 불가능한 값은 NaN)
                self.raw_data[region] = pd.to_numeric(self.raw_data[region], errors='coerce')
        
        print(f"✅ 전처리 완료: {self.raw_data.shape}")
        return self.raw_data
    
    def separate_data_types(self):
        """신환자 수와 발생률 데이터 분리"""
        print("\n📊 데이터 타입 분리 중...")
        
        # 전국 데이터의 평균값으로 신환자 수와 발생률 구분
        # 일반적으로 신환자 수는 수천~수만, 발생률은 100 이하
        threshold = 1000  # 구분 기준값
        
        new_cases_mask = self.raw_data['전국'] >= threshold
        incidence_mask = self.raw_data['전국'] < threshold
        
        self.new_cases_data = self.raw_data[new_cases_mask].copy()
        self.incidence_rate_data = self.raw_data[incidence_mask].copy()
        
        print(f"📈 신환자 수 데이터: {len(self.new_cases_data)}년")
        print(f"📉 발생률 데이터: {len(self.incidence_rate_data)}년")
        
        return self.new_cases_data, self.incidence_rate_data
    
    def data_quality_report(self):
        """데이터 품질 리포트"""
        print("\n📋 데이터 품질 분석")
        print("=" * 50)
        
        for data_type, data in [("신환자 수", self.new_cases_data), ("발생률", self.incidence_rate_data)]:
            if data is not None and not data.empty:
                print(f"\n🔍 {data_type} 데이터 품질:")
                
                # 결측값 분석
                missing_info = []
                for region in self.regions:
                    if region in data.columns:
                        total = len(data)
                        missing = data[region].isna().sum()
                        completeness = ((total - missing) / total * 100)
                        missing_info.append({
                            '지역': region,
                            '총 데이터': total,
                            '결측값': missing,
                            '완성도(%)': round(completeness, 1)
                        })
                
                missing_df = pd.DataFrame(missing_info)
                print(missing_df.to_string(index=False))
    
    def trend_analysis(self):
        """트렌드 분석"""
        print("\n📈 트렌드 분석")
        print("=" * 50)
        
        for data_type, data in [("신환자 수", self.new_cases_data), ("발생률", self.incidence_rate_data)]:
            if data is not None and not data.empty:
                print(f"\n📊 {data_type} 변화 추이:")
                
                # 연도순 정렬
                data_sorted = data.sort_values('연도')
                first_year = data_sorted.iloc[0]
                last_year = data_sorted.iloc[-1]
                
                trend_info = []
                for region in self.regions:
                    if region in data.columns:
                        first_val = first_year[region]
                        last_val = last_year[region]
                        
                        if pd.notna(first_val) and pd.notna(last_val):
                            change = last_val - first_val
                            pct_change = (change / first_val * 100)
                            trend = "증가" if change > 0 else "감소" if change < 0 else "변화없음"
                            
                            trend_info.append({
                                '지역': region,
                                f'{first_year["연도"]}년': first_val,
                                f'{last_year["연도"]}년': last_val,
                                '변화량': round(change, 1),
                                '변화율(%)': round(pct_change, 1),
                                '추세': trend
                            })
                
                trend_df = pd.DataFrame(trend_info)
                print(trend_df.to_string(index=False))
    
    def statistical_summary(self):
        """통계 요약"""
        print("\n📊 통계 요약")
        print("=" * 50)
        
        for data_type, data in [("신환자 수", self.new_cases_data), ("발생률", self.incidence_rate_data)]:
            if data is not None and not data.empty:
                print(f"\n📈 {data_type} 통계:")
                
                # 주요 지역만 선택 (전국, 서울, 부산, 경기)
                key_regions = ['전국', '서울', '부산', '경기']
                available_regions = [r for r in key_regions if r in data.columns]
                
                if available_regions:
                    summary = data[available_regions].describe()
                    print(summary.round(1))
    
    def create_visualizations(self):
        """시각화 생성"""
        print("\n📊 시각화 생성 중...")
        
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('결핵 신환자 현황 분석', fontsize=16, fontweight='bold')
        
        # 1. 전국 신환자 수 추이
        if self.new_cases_data is not None and not self.new_cases_data.empty:
            data_sorted = self.new_cases_data.sort_values('연도')
            axes[0, 0].plot(data_sorted['연도'], data_sorted['전국'], 
                           marker='o', linewidth=2, markersize=6)
            axes[0, 0].set_title('전국 결핵 신환자 수 추이')
            axes[0, 0].set_xlabel('연도')
            axes[0, 0].set_ylabel('신환자 수 (명)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 전국 발생률 추이
        if self.incidence_rate_data is not None and not self.incidence_rate_data.empty:
            data_sorted = self.incidence_rate_data.sort_values('연도')
            axes[0, 1].plot(data_sorted['연도'], data_sorted['전국'], 
                           marker='s', color='orange', linewidth=2, markersize=6)
            axes[0, 1].set_title('전국 결핵 발생률 추이')
            axes[0, 1].set_xlabel('연도')
            axes[0, 1].set_ylabel('발생률 (인구 10만명당)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 주요 지역별 최신 신환자 수 비교
        if self.new_cases_data is not None and not self.new_cases_data.empty:
            latest_data = self.new_cases_data.iloc[-1]
            major_regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기']
            region_data = [(r, latest_data[r]) for r in major_regions if r in latest_data.index and pd.notna(latest_data[r])]
            
            if region_data:
                regions, values = zip(*region_data)
                axes[1, 0].bar(regions, values, color='skyblue', alpha=0.7)
                axes[1, 0].set_title(f'{latest_data["연도"]}년 지역별 신환자 수')
                axes[1, 0].set_xlabel('지역')
                axes[1, 0].set_ylabel('신환자 수 (명)')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 주요 지역별 최신 발생률 비교
        if self.incidence_rate_data is not None and not self.incidence_rate_data.empty:
            latest_data = self.incidence_rate_data.iloc[-1]
            major_regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기']
            region_data = [(r, latest_data[r]) for r in major_regions if r in latest_data.index and pd.notna(latest_data[r])]
            
            if region_data:
                regions, values = zip(*region_data)
                axes[1, 1].bar(regions, values, color='lightcoral', alpha=0.7)
                axes[1, 1].set_title(f'{latest_data["연도"]}년 지역별 발생률')
                axes[1, 1].set_xlabel('지역')
                axes[1, 1].set_ylabel('발생률 (인구 10만명당)')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("🚀 결핵 신환자 데이터 분석 시작")
        print("=" * 60)
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 전처리
        self.preprocess_data()
        
        # 3. 데이터 타입 분리
        self.separate_data_types()
        
        # 4. 품질 분석
        self.data_quality_report()
        
        # 5. 트렌드 분석
        self.trend_analysis()
        
        # 6. 통계 요약
        self.statistical_summary()
        
        # 7. 시각화
        self.create_visualizations()
        
        print("\n✅ 분석 완료!")
        
        # 분석된 데이터 반환
        return {
            'new_cases': self.new_cases_data,
            'incidence_rate': self.incidence_rate_data,
            'raw_data': self.raw_data
        }

# 사용 예시
if __name__ == "__main__":
    # 분석기 초기화 및 실행
    analyzer = TuberculosisAnalyzer('/Users/yeowon/Desktop/Data/결핵/8_시도별_결핵_(신)환자수_및_율_2011-2024.csv')
    results = analyzer.run_full_analysis()
    
    # 추가 분석이 필요한 경우
    print("\n📌 추가 분석 예시:")
    print("# 특정 연도 데이터 필터링")
    print("recent_data = results['new_cases'][results['new_cases']['연도'] >= 2020]")
    
    print("\n# 특정 지역 추이 분석")
    print("seoul_trend = results['new_cases'][['연도', '서울']].dropna()")
    print("seoul_trend.plot(x='연도', y='서울', kind='line')")
    
    print("\n# 상관관계 분석")
    print("correlation = results['new_cases'][['서울', '부산', '경기']].corr()")
    print("print(correlation)")