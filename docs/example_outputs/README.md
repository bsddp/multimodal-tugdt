# Public example outputs

This directory contains aggregate artifacts generated from the committed synthetic demonstration.
They are versioned so a reviewer can inspect the expected output without running the project.

The examples contain no participant-level feature values, raw paths, faces, voices, or clinical
claims. Regenerate the research summary with:

```bash
tugdt run-all --config configs/example.yaml
tugdt generate-report --config configs/example.yaml \
  --output docs/example_outputs/synthetic_research_summary.md
```
