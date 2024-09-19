import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import requests
import pandas as pd
from tqdm import tqdm
from lib.vllm_client import VLLMClient
from mcts_llm.mctsr import MCTSrLlama38B
from mcts_llm.prompt_configs import llama_3_8b_prompt_config, llama_3_70b_prompt_config



def check_api_key(api_key: str, base_url: str):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.get(f'{base_url}/models', headers=headers)
        response.raise_for_status()
        print("API Key is valid and has available credits.")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def load_custom_dataset(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            if line.strip():  
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Ошибка при декодировании строки: {line}")
                    print(e)
    return pd.DataFrame(data)

def run_vllm_on_custom_dataset(file_path: str, model_config) -> None:
    print("Loading custom dataset...")
    # Load the custom dataset
    custom_data = load_custom_dataset(file_path)
    print(f"Loaded {len(custom_data)} questions from dataset.")

    # Initialize VLLM client
    print("Initializing VLLM client...")
    client = VLLMClient(base_url=model_config.base_url, api_key="f7ef6fdd18cf8b61c4d665da8165e32ef99ca884845f66814280f468a7310c2c")

    results = []

    for _, item in tqdm(custom_data.iterrows(), total=len(custom_data)):
        question = item['question']
        print(f"Processing question: {question}")
        
        # Initialize MCTS with VLLM
        try:
            mctsr = MCTSrLlama38B(
                problem=question,
                max_rollouts=8,
                max_children=2,
                selection_policy=2,
                initialize_strategy=1,
                model_client=client,
                model_name=model_config.model
            )

            best_answer = mctsr.run()
            print(f"Best answer: {best_answer}")

            results.append({
                'question_id': item['question_id'],
                'cluster': item['cluster'],
                'category': item['category'],
                'question': question,
                'mcts_answer': best_answer,
                'best_q_value': mctsr.root.Q
            })
        except Exception as e:
            print(f"Error processing question: {question}")
            print(e)

    # Save results to CSV
    print("Saving results to CSV...")
    df = pd.DataFrame(results)
    df.to_csv('results/custom_dataset_vllm_results.csv', index=False)
    print("Results saved to results/custom_dataset_vllm_results.csv")

if __name__ == "__main__":
    api_key = "f7ef6fdd18cf8b61c4d665da8165e32ef99ca884845f66814280f468a7310c2c"
    base_url = "https://api.together.xyz/v1"
    check_api_key(api_key, base_url)

    # Пример запуска с использованием конфигурации модели llama_3_8b
    run_vllm_on_custom_dataset('../data/test_dataset.jsonl', llama_3_8b_prompt_config)
