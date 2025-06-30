import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TuberculosisCSVAnalyzer:
    def __init__(self, file_path):
        """결핵 종류별 CSV 데이터 분석기"""
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.year_column = None
        self.numeric_columns = []
        self.data_type = '미확인'
        
    def load_csv_data(self):
        """CSV 파일 로드 및 기본 탐색"""
        print("📁 CSV 파일 로딩 중...")
        
        # 다양한 인코딩으로 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                print(f"🔧 인코딩 '{encoding}' 시도 중...")
                self.raw_data = pd.read_csv(self.file_path, encoding=encoding)
                print(f"✅ '{encoding}' 인코딩으로 로드 성공!")
                break
            except UnicodeDecodeError:
                print(f"⚠️ '{encoding}' 인코딩 실패")
                continue
            except Exception as e:
                print(f"❌ '{encoding}' 인코딩 오류: {e}")
                continue
        
        if self.raw_data is None:
            print("❌ 모든 인코딩 시도 실패")
            return None
        
        # 기본 정보 출력
        print(f"\n📊 데이터 기본 정보:")
        print(f"- 형태: {self.raw_data.shape}")
        print(f"- 컬럼: {list(self.raw_data.columns)}")
        
        # 데이터 미리보기
        print(f"\n🔍 데이터 미리보기:")
        print(self.raw_data.head())
        
        # 데이터 타입 확인
        print(f"\n📋 데이터 타입:")
        print(self.raw_data.dtypes)
        
        # 결측값 확인
        missing_data = self.raw_data.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n❌ 결측값 현황:")
            print(missing_data[missing_data > 0])
        else:
            print(f"\n✅ 결측값 없음")
        
        return self.raw_data
    
    def detect_data_structure(self):
        """데이터 구조 자동 감지"""
        print("\n🔍 데이터 구조 분석...")
        
        # 컬럼명 정리
        self.raw_data.columns = self.raw_data.columns.astype(str).str.strip()
        
        # 1. 연도 컬럼 찾기
        print("📅 연도 컬럼 탐지...")
        
        # 연도 키워드로 찾기
        year_keywords = ['연도', 'year', '년도', '년', 'YEAR']
        year_candidates = []
        
        for col in self.raw_data.columns:
            if any(keyword in str(col).lower() for keyword in year_keywords):
                year_candidates.append(col)
        
        # 첫 번째 컬럼이 연도인지 확인
        first_col = self.raw_data.columns[0]
        if not year_candidates:
            try:
                # 첫 번째 컬럼의 값들이 연도 범위인지 확인
                first_col_numeric = pd.to_numeric(self.raw_data[first_col], errors='coerce')
                valid_years = first_col_numeric.dropna()
                if len(valid_years) > 0 and all(2000 <= year <= 2030 for year in valid_years):
                    year_candidates.append(first_col)
                    print(f"✅ 첫 번째 컬럼을 연도로 인식: {first_col}")
            except:
                pass
        
        self.year_column = year_candidates[0] if year_candidates else None
        print(f"📅 연도 컬럼: {self.year_column}")
        
        # 2. 결핵 종류 컬럼 식별
        print("\n🦠 결핵 종류 컬럼 식별...")
        
        tb_keywords = {
            '폐결핵': ['폐결핵', 'pulmonary', '폐'],
            '폐외결핵': ['폐외결핵', 'extrapulmonary', '폐외'],
            '도말양성': ['도말양성', 'smear_positive', '양성'],
            '도말음성': ['도말음성', 'smear_negative', '음성'],
            '배양양성': ['배양양성', 'culture_positive'],
            '배양음성': ['배양음성', 'culture_negative'],
            '신도말양성': ['신도말양성'],
            '신도말음성': ['신도말음성'],
            '재치료': ['재치료', 'retreatment'],
            '초치료': ['초치료', 'initial']
        }
        
        identified_types = {}
        for tb_type, keywords in tb_keywords.items():
            matching_cols = []
            for col in self.raw_data.columns:
                if col != self.year_column:
                    for keyword in keywords:
                        if keyword in str(col).lower():
                            matching_cols.append(col)
                            break
            if matching_cols:
                identified_types[tb_type] = matching_cols
        
        print("📋 식별된 결핵 종류:")
        for tb_type, cols in identified_types.items():
            print(f"  {tb_type}: {cols}")
        
        # 3. 숫자 컬럼 식별
        print(f"\n🔢 숫자 컬럼 식별...")
        self.numeric_columns = []
        
        for col in self.raw_data.columns:
            if col != self.year_column:
                try:
                    # 문자열 정리 후 숫자 변환 테스트
                    test_series = self.raw_data[col].astype(str).str.replace(',', '').str.replace('nan', '')
                    test_numeric = pd.to_numeric(test_series, errors='coerce')
                    
                    # 유효한 숫자가 50% 이상이면 숫자 컬럼으로 간주
                    valid_ratio = test_numeric.notna().sum() / len(test_numeric)
                    if valid_ratio >= 0.5:
                        self.numeric_columns.append(col)
                except:
                    pass
        
        print(f"📊 숫자 컬럼 ({len(self.numeric_columns)}개): {self.numeric_columns}")
        
        # 4. 데이터 타입 추정
        if self.numeric_columns:
            # 평균값으로 신환자수 vs 발생률 판단
            avg_values = []
            for col in self.numeric_columns[:3]:  # 처음 3개 컬럼만 확인
                try:
                    col_data = pd.to_numeric(self.raw_data[col].astype(str).str.replace(',', ''), errors='coerce')
                    avg_val = col_data.mean()
                    if pd.notna(avg_val):
                        avg_values.append(avg_val)
                except:
                    pass
            
            if avg_values:
                overall_avg = np.mean(avg_values)
                if overall_avg >= 1000:
                    self.data_type = '신환자수'
                elif overall_avg >= 100:
                    self.data_type = '신환자수'
                else:
                    self.data_type = '발생률'
                    
                print(f"📊 데이터 타입: {self.data_type} (평균값: {overall_avg:.1f})")
        
        return identified_types
    
    def preprocess_data(self):
        """데이터 전처리"""
        print("\n🔧 데이터 전처리 시작...")
        
        self.processed_data = self.raw_data.copy()
        
        # 1. 연도 데이터 정리
        if self.year_column:
            print(f"📅 연도 컬럼 '{self.year_column}' 처리 중...")
            
            # 연도를 숫자로 변환
            self.processed_data[self.year_column] = pd.to_numeric(
                self.processed_data[self.year_column], errors='coerce'
            )
            
            # 유효한 연도만 필터링
            valid_data = self.processed_data[self.processed_data[self.year_column].notna()]
            if not valid_data.empty:
                self.processed_data = valid_data
                self.processed_data[self.year_column] = self.processed_data[self.year_column].astype(int)
                
                # 2011-2024 범위 필터링
                year_filtered = self.processed_data[
                    (self.processed_data[self.year_column] >= 2011) & 
                    (self.processed_data[self.year_column] <= 2024)
                ]
                
                if not year_filtered.empty:
                    self.processed_data = year_filtered
                    print(f"✅ 연도 범위: {self.processed_data[self.year_column].min()}~{self.processed_data[self.year_column].max()}")
                else:
                    print(f"⚠️ 2011-2024 범위 데이터 없음. 전체 범위 사용: {self.processed_data[self.year_column].min()}~{self.processed_data[self.year_column].max()}")
        
        # 2. 숫자 컬럼 정리
        print(f"🔢 숫자 컬럼 정리 중...")
        
        cleaned_numeric_cols = []
        for col in self.numeric_columns:
            try:
                # 문자열 정리 (쉼표, 공백 제거)
                self.processed_data[col] = (
                    self.processed_data[col]
                    .astype(str)
                    .str.replace(',', '')
                    .str.replace(' ', '')
                    .str.replace('nan', '')
                    .replace('', np.nan)
                )
                
                # 숫자 변환
                self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')
                
                # 유효한 데이터가 있는 컬럼만 유지
                if self.processed_data[col].notna().sum() > 0:
                    cleaned_numeric_cols.append(col)
                    
            except Exception as e:
                print(f"⚠️ 컬럼 '{col}' 처리 중 오류: {e}")
        
        self.numeric_columns = cleaned_numeric_cols
        print(f"✅ 정리된 숫자 컬럼: {len(self.numeric_columns)}개")
        
        # 3. 빈 행 제거
        self.processed_data = self.processed_data.dropna(how='all')
        
        print(f"✅ 전처리 완료: {self.processed_data.shape}")
        return self.processed_data
    
    def analyze_tuberculosis_data(self):
        """결핵 데이터 분석"""
        print("\n📊 결핵 데이터 분석")
        print("=" * 50)
        
        if not self.year_column or not self.numeric_columns:
            print("❌ 분석 필요 데이터 부족 (연도 또는 숫자 컬럼 없음)")
            return None, None
        
        try:
            # 연도별 집계
            yearly_data = self.processed_data.groupby(self.year_column)[self.numeric_columns].sum()
            
            print(f"📈 {self.data_type} 연도별 분석:")
            print(f"📋 분석 항목: {self.numeric_columns}")
            print(f"📅 분석 기간: {yearly_data.index.min()}~{yearly_data.index.max()}년 ({len(yearly_data)}년간)")
            
            # 최근 데이터 출력
            recent_years = min(5, len(yearly_data))
            print(f"\n최근 {recent_years}년 데이터:")
            print(yearly_data.tail(recent_years).round(1))
            
            # 전체 통계
            print(f"\n📊 전체 기간 통계:")
            stats = yearly_data.describe()
            print(stats.round(1))
            
            # 증감 분석
            change_analysis = None
            if len(yearly_data) >= 2:
                print(f"\n📈 증감 분석 ({yearly_data.index.min()}년 → {yearly_data.index.max()}년):")
                
                first_year = yearly_data.iloc[0]
                last_year = yearly_data.iloc[-1]
                
                changes = []
                for col in self.numeric_columns:
                    if (pd.notna(first_year[col]) and pd.notna(last_year[col]) and 
                        first_year[col] > 0):
                        
                        change = last_year[col] - first_year[col]
                        pct_change = (change / first_year[col]) * 100
                        trend = "📈증가" if change > 0 else "📉감소" if change < 0 else "➡️변화없음"
                        
                        changes.append({
                            '항목': col,
                            '초기값': round(first_year[col], 1),
                            '최종값': round(last_year[col], 1),
                            '변화량': round(change, 1),
                            '변화율(%)': round(pct_change, 1),
                            '추세': trend
                        })
                        
                        print(f"  {col}: {first_year[col]:.1f} → {last_year[col]:.1f} "
                              f"({pct_change:+.1f}%) {trend}")
                
                if changes:
                    change_analysis = pd.DataFrame(changes)
            
            return yearly_data, change_analysis
            
        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
            return None, None
    
    def create_visualizations(self, yearly_data):
        """시각화 생성"""
        print("\n📊 시각화 생성 중...")
        
        if yearly_data is None or yearly_data.empty:
            print("❌ 시각화할 데이터가 없습니다.")
            return
        
        try:
            # 메인 시각화
            n_cols = min(len(self.numeric_columns), 4)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'결핵 종류별 {self.data_type} 분석', fontsize=14, fontweight='bold')
            
            axes = axes.flatten()
            
            for i, col in enumerate(self.numeric_columns[:4]):
                if col in yearly_data.columns:
                    yearly_data[col].plot(kind='line', ax=axes[i], marker='o', linewidth=2, markersize=6)
                    axes[i].set_title(f'{col} 추이')
                    axes[i].set_xlabel('연도')
                    axes[i].set_ylabel(self.data_type)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
            
            # 빈 서브플롯 숨기기
            for i in range(n_cols, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # 비교 차트
            if len(self.numeric_columns) > 1:
                plt.figure(figsize=(14, 6))
                
                # 최신 연도 비교
                latest_data = yearly_data.iloc[-1]
                latest_year = yearly_data.index[-1]
                
                # 상위 8개 항목
                valid_data = latest_data[latest_data.notna() & (latest_data > 0)]
                if len(valid_data) > 0:
                    top_items = valid_data.nlargest(min(8, len(valid_data)))
                    
                    plt.subplot(1, 2, 1)
                    bars = plt.bar(range(len(top_items)), top_items.values, color='skyblue', alpha=0.7)
                    plt.title(f'{latest_year}년 결핵 종류별 {self.data_type} (상위 {len(top_items)}개)')
                    plt.xlabel('결핵 종류')
                    plt.ylabel(self.data_type)
                    plt.xticks(range(len(top_items)), [col.replace('_', '\n') for col in top_items.index], 
                              rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    
                    # 값 표시
                    for bar, value in zip(bars, top_items.values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                                f'{value:.0f}', ha='center', va='bottom', fontsize=9)
                    
                    # 파이차트
                    plt.subplot(1, 2, 2)
                    plt.pie(top_items.values, labels=top_items.index, autopct='%1.1f%%', startangle=90)
                    plt.title(f'{latest_year}년 구성비')
                    
                    plt.tight_layout()
                    plt.show()
                    
        except Exception as e:
            print(f"❌ 시각화 생성 오류: {e}")
    
    def export_results(self, yearly_data, change_analysis):
        """결과 내보내기"""
        print(f"\n💾 분석 결과 내보내기...")
        
        try:
            # 연도별 데이터 저장
            if yearly_data is not None and not yearly_data.empty:
                yearly_filename = 'tuberculosis_yearly_analysis.csv'
                yearly_data.to_csv(yearly_filename, encoding='utf-8-sig')
                print(f"📄 연도별 분석 결과: {yearly_filename}")
            
            # 변화 분석 저장
            if change_analysis is not None and not change_analysis.empty:
                change_filename = 'tuberculosis_change_analysis.csv'
                change_analysis.to_csv(change_filename, index=False, encoding='utf-8-sig')
                print(f"📊 변화 분석 결과: {change_filename}")
            
            # 전처리된 원본 데이터 저장
            processed_filename = 'tuberculosis_processed_data.csv'
            self.processed_data.to_csv(processed_filename, index=False, encoding='utf-8-sig')
            print(f"🔧 전처리된 데이터: {processed_filename}")
            
        except Exception as e:
            print(f"❌ 파일 저장 오류: {e}")
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("🚀 결핵 종류별 CSV 데이터 전체 분석 시작")
        print("=" * 60)
        
        try:
            # 1. 데이터 로드
            if not self.load_csv_data():
                return None
            
            # 2. 구조 분석
            tb_types = self.detect_data_structure()
            
            # 3. 전처리
            self.preprocess_data()
            
            # 4. 분석
            yearly_data, change_analysis = self.analyze_tuberculosis_data()
            
            # 5. 시각화
            if yearly_data is not None:
                self.create_visualizations(yearly_data)
            
            # 6. 결과 저장
            self.export_results(yearly_data, change_analysis)
            
            print("\n✅ 전체 분석 완료!")
            
            return {
                'raw_data': self.raw_data,
                'processed_data': self.processed_data,
                'yearly_data': yearly_data,
                'change_analysis': change_analysis,
                'data_type': self.data_type,
                'numeric_columns': self.numeric_columns
            }
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

# 사용 예시
if __name__ == "__main__":
    # 분석 실행
    analyzer = TuberculosisCSVAnalyzer('/Users/yeowon/Desktop/Data/결핵/7_성별_연령별_결핵_(신)환자수_및_율_2011-2024.csv')
    results = analyzer.run_full_analysis()
    
    if results:
        print("\n🎯 분석 완료! 결과 활용 방법:")
        print("- results['yearly_data']: 연도별 집계 데이터")
        print("- results['change_analysis']: 증감 분석 결과")
        print("- results['processed_data']: 전처리된 원본 데이터")
        
        print("\n📌 추가 분석 예시:")
        print("# 특정 결핵 종류 트렌드")
        print("results['yearly_data']['폐결핵'].plot(kind='line')")
        print("\n# 상관관계 분석")
        print("results['yearly_data'].corr()")