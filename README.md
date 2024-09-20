# mcts-llm

## MCTSr

Based on [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394) by Zhang, et al.

At a high level, MCTSr iteratively generates solutions to a specified (math) problem.

In a MCTSr tree, nodes correspond to attempted answers, and edges correspond to attempts to improve the answer.


### Initialize
Generate an solution to the problem. This paper uses a "dummy" solution (e.g. `"I don't know"`).

### Select a node to expand
We gather a set of candidate nodes which haven't been fully expanded.

A node is fully expanded if either:
1. it has `max_children`
2. any of its children have a Q value which is greater than its own

Once we've gathered the candidates, we compute UCT scores for each candidate node.
There are a few ways we can make our selection:
1. Greedily (choose the node with the highest UCT)
2. Importance sampling (sample from the set of candidates, weighted by their UCT score)
3. Pairwise importance sampling (sample the max from a pair of nodes from the set of candidates, weighted by the difference between the pair's UCT scores)

The authors mention that they perform greedy selection in the paper. In their [repo](https://github.com/trotsky1997/MathBlackBox/blob/main/gen_mcts_dpo.py#L182), they also perform pairwise sampling and save the (question, answer1, answer2) tuples for use in DPO.

### Expand the node

Expansion involves several steps:
1. Generate a critique of the current solution.
2. Refine the solution based on the critique.
3. Add a new child, corresponding to the refined solution.
4. Self-evaluate the `reward` of the new child.
5. Backpropagate the reward from the new child through its parents, through to the root.


# Results
| Model          | Arena Hard RU | Arena Hard EN |
|----------------|---------------|---------------|
| LLaMA 8B       | 35.07       |  20.6      |
| LLaMA 8B MCTS  | 45.71      | 321      |
| Gemma 9B       | Value 5       | Value 6       |
| Gemma 9B MCTS  | Value 7       | Value 8       |


# how to use our repo

1. **Clone the repository** (assuming the repository URL is provided):
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install dependencies** (from a `requirements.txt` file):
   ```bash
   pip install -r requirements.txt
   ```

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```


```python
import os
from src.mcts_llm.mctsr import MCTSrLlama38B
from src.dataset_utils import load_aime
from tqdm import tqdm
import pandas as pd
from src.mcts_llm.mctsr import print_tree

# Set your API key
os.environ['TOGETHER_API_KEY'] = '<your key>'

# Define the problem/question
q = """Your Task"""

# Initialize the MCTS model with the given parameters
mctsr = MCTSrLlama38B(
    problem=q,
    max_rollouts=4,
    max_children=8,
    selection_policy=2,
    initialize_strategy=2
)

# Run the MCTS model
best_answer = mctsr.run()

# Print the best answer found
print(best_answer)
```

# How to use 
- `max_rollouts=8`
- `max_children=2`

