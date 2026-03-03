"""
Quick sanity-check tests (no Docker/LLM required).
Run: python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.parser import parse_llm_response, extract_metrics, extract_error
from core.state import ExperimentState


def test_parse_code_block():
    text = """
I will run a linear regression.
Hypothesis: Linear models outperform random baselines.

```python RUN
import json
print(json.dumps({"accuracy": 0.92, "loss": 0.08}))
```

ACTION: ITERATE
"""
    result = parse_llm_response(text)
    assert result.code is not None
    assert "accuracy" in result.code
    assert result.action == "ITERATE"
    assert result.hypothesis is not None
    print("✅ parse_code_block passed")


def test_parse_done():
    text = "The results are conclusive.\nACTION: DONE\n"
    result = parse_llm_response(text)
    assert result.action == "DONE"
    print("✅ parse_done passed")


def test_extract_metrics():
    output = "Training...\nEpoch 5\n{\"accuracy\": 0.95, \"f1\": 0.94}\n"
    metrics = extract_metrics(output)
    assert metrics["accuracy"] == 0.95
    print("✅ extract_metrics passed")


def test_extract_error():
    output = "Running...\nTraceback (most recent call last):\n  File 'x.py'\nValueError: bad input\n"
    error = extract_error(output, exit_code=1)
    assert "Traceback" in error
    print("✅ extract_error passed")


def test_state():
    state = ExperimentState(goal="Test goal")
    state.add_iteration(
        code="print('hello')",
        output="hello\n{\"acc\": 0.9}",
        metrics={"acc": 0.9},
        success=True,
    )
    assert len(state.history) == 1
    assert state.latest_metrics["acc"] == 0.9
    assert state.successful_runs == 1
    print("✅ state passed")


def test_parse_latex():
    text = r"""
Here is the paper.

```latex PAPER
\documentclass{article}
\begin{document}
Hello
\end{document}
```
"""
    result = parse_llm_response(text)
    assert result.latex is not None
    assert "documentclass" in result.latex
    print("✅ parse_latex passed")


def test_parse_experiment_spec():
    text = """
I will compare Adam vs SGD.
Hypothesis: Adam converges faster on MNIST.

```spec
Task: Train a simple MLP on MNIST for 3 epochs
Dataset: MNIST via torchvision
Model: 2-layer MLP (784 → 128 → 10)
Hyperparameters: lr=0.001, batch_size=64, optimizer=Adam
Metrics to report: accuracy, loss
Files to save: plot_loss.png to /results/
```

ACTION: ITERATE
"""
    result = parse_llm_response(text)
    assert result.experiment_spec is not None
    assert "Task" in result.experiment_spec
    assert "Adam" in result.experiment_spec
    assert result.action == "ITERATE"
    assert result.hypothesis is not None
    print("✅ parse_experiment_spec passed")


def test_parse_experiment_spec_plain_delimiter():
    text = """
Hypothesis: SGD with momentum beats vanilla SGD.

SPEC:
Task: Compare SGD vs SGD+momentum on CIFAR-10
Dataset: CIFAR-10
Metrics to report: accuracy
END SPEC

ACTION: ITERATE
"""
    result = parse_llm_response(text)
    assert result.experiment_spec is not None
    assert "SGD" in result.experiment_spec
    print("✅ parse_experiment_spec_plain_delimiter passed")


if __name__ == "__main__":
    test_parse_code_block()
    test_parse_done()
    test_extract_metrics()
    test_extract_error()
    test_state()
    test_parse_latex()
    test_parse_experiment_spec()
    test_parse_experiment_spec_plain_delimiter()
    print("\n🎉 All tests passed!")
