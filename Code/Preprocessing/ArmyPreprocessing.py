from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

# 실제 함수 이름들로 정확히 import
from Army.tuberculosis_32 import preprocess_32_disease_grading
from Army.tuberculosis_33 import preprocess_33_hepatitis_b_data
from Army.tuberculosis_44 import preprocess_44_latent_tb_data  
from Army.tuberculosis_59 import preprocess_59_latent_tb_by_region  
from Army.tuberculosis_75 import preprocess_75_igra_analysis


def run_all_preprocessing():
    base = Path("/Users/yeowon/Desktop/Data/병무청")

    df32 = preprocess_32_disease_grading(base / "32_병무청_병역판정검사_결과(4급5급6급)_20181231.csv")
    df33 = preprocess_33_hepatitis_b_data(base / "33_병무청_B형간염_유병률_연구데이터_제공_20191231.csv")
    df44 = preprocess_44_latent_tb_data(base / "44_병무청_잠복결핵_유병률_조사_데이터_제공[Kdata]_20211231.csv")
    df59 = preprocess_59_latent_tb_by_region(base / "59_병무청_병역판정검사_잠복결핵검사_실적및청별_현황_20231231.csv")
    df75 = preprocess_75_igra_analysis(base / "75_병무청_잠복결핵검사의_효과분석을_위한_병역판정검사_자료제공.csv")

    return {
        "df32": df32,
        "df33": df33,
        "df44": df44,
        "df59": df59,
        "df75": df75
    }

if __name__ == "__main__":
    print("데이터 전처리를 시작합니다...")
    try:
        results = run_all_preprocessing()
        print("모든 데이터셋 전처리 완료")
        print("\n" + "="*50)
        
        for name, df in results.items():
            print(f"[{name}] shape: {df.shape}")
            print(f"컬럼들: {list(df.columns)}")
            print(df.head(2))
            print("-" * 30)
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()