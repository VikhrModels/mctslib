import json
from mcts_llm.mctsr import MCTSrLlama38B
from dataset_utils import load_aime
from tqdm import tqdm
import pandas as pd

def load_aime_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def run_mcts_on_aime():
    # Load the AIME dataset
    aime_data = load_aime(2022)

    results = []
    # print(aime_data)
    aime_data.reset_index(inplace=True)
    # print(aime_data)

    for problem in tqdm(range(1,len(aime_data))):
        mctsr = MCTSrLlama38B(
            problem=aime_data.at[problem, 'question'],
            max_rollouts=8,
            max_children=2,
            selection_policy=2,
            initialize_strategy=1
        )

        best_answer = mctsr.run()

        results.append({
            'question': aime_data.at[problem,'question'],
            'correct_answer': aime_data.at[problem,'answer'],
            'mcts_answer': best_answer,
            'best_q_value': mctsr.root.Q
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('results/AIME_2022_llama_3_70b.csv', index=False)

import concurrent.futures

def run_mcts_on_custom_dataset(file_path):
    # Load the custom dataset
    with open(file_path, 'r') as f:
        custom_data = [json.loads(fl) for fl in f][250:]

    results = []

    def process_item(item):
        question = item['turns'][0]['content']
        mctsr = MCTSrLlama38B(
            problem=question,
            max_rollouts=4,
            max_children=2,
            selection_policy=2,
            initialize_strategy=1
        )

        best_answer = mctsr.run()

        return {
            'question_id': item['question_id'],
            'cluster': item['cluster'],
            'category': item['category'],
            'question': question,
            'mcts_answer': best_answer,
            'best_q_value': mctsr.root.Q
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        results = list(tqdm(executor.map(process_item, custom_data), total=len(custom_data)))

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('results/custom_dataset_results_500.csv', index=False)


if __name__ == "__main__":
    # run_mcts_on_aime()
    run_mcts_on_custom_dataset('/root/kostya/mcts/mcts-llm/datasets/question-4.jsonl')
