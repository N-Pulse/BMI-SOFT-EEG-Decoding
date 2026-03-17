# Tests

TODO: Write tests.

Suggested coverage:

- **Data pipeline**: `src.data.processing`, `src.data.simplify_labels` — test label maps, NPZ I/O, simplify output shape.
- **Models / training**: `src.models` (or `src.train`) — test pipeline build, bandpower/CSP feature shape, train/val split, metrics.
- **Evaluation**: `src.evaluation.compare_training` — test that comparison runs and produces summary CSV/metrics.

Run from project root:

```bash
pytest tests/ -v
```
