from src.inference_api import run_inference

if __name__ == "__main__":
    df_out = run_inference()
    print(df_out.head())
