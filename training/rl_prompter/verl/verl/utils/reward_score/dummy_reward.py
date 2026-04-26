"""Smoke-only dummy reward: returns a constant per sample so we can verify the GRPO loop."""
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 1.0
