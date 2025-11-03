hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

### Task Title

**Hyperparameter Tuning for Model Performance Improvement**

### Objective

The goal of this task is to train an LLM to perform **hyperparameter tuning** — a common and valuable skill for ML engineers and researchers. The model is given a baseline classifier with a fixed accuracy and is tasked with improving its performance by adjusting key hyperparameters (e.g., learning rate, optimizer, or batch size). The model must propose code or configuration changes that achieve an accuracy improvement beyond a specified threshold.

### Task Design

* **Prompt:** The model is instructed to improve the baseline model’s validation accuracy above a given target (e.g., from 0.675 to ≥0.695).
* **Tools/Data:** The model is provided with access to a small dataset (e.g., a subset of scikit-learn’s iris or digits dataset) and a standard training script.
* **Verification (Grader):** The grader runs the model’s proposed configuration, evaluates accuracy, and checks if the final accuracy meets or exceeds the target threshold.
* **Scoring:** The model passes if its tuned configuration achieves ≥ target accuracy.

### Skill Taught

This task reinforces practical ML experimentation habits: reasoning about hyperparameters, understanding learning dynamics, and iterating on model configurations to meet performance goals.

### Evaluation & Pass Rate

The task was executed 10 times to compute a success rate:

```
Baseline accuracy: 0.675
Target accuracy:   0.695
--------------------------------------------------
Pass Rate: 20.0% (2/10)
```

This pass rate falls within the required **10–40% range**, indicating the task is well-calibrated in difficulty.

### Summary

This task satisfies all requirements:

* Teaches a relevant ML engineering skill
* Evaluates against measurable success criteria
* Allows multiple valid solutions
* Maintains moderate (10–40%) success difficulty
* Concise (<300 lines of code) and easy to review

