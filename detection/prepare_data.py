import pandas as pd

files = [
  "datasets/temperature_1/run_3/chatgpt_results_exp_chatgpt_1771584462.csv",
  "datasets/temperature_1/run_3/Qwen_Qwen2.5-72B-Instruct_1771584467.csv",
  "datasets/temperature_1/run_3/meta-llama_llama-3.1-70b-instruct_1771584467.csv"
]
dfs = [pd.read_csv(f) for f in files]
pd.concat(dfs, ignore_index=True).to_csv("datasets/generated_descriptions.csv", index=False)
