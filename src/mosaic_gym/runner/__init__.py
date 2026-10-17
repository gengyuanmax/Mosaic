"""Runner package — drive the agent over data (config-first).

    inference.py  the loop: run_episode (one episode) + run_task / run_inference (batch) + build_agent
    scoring.py    answer_reward (1/0 correctness — the only reward there is)
    data.py       thin loader: load_samples / sample_to_case (canonical -> CaseInput)

Import from the submodules directly (e.g. ``from mosaic_gym.runner.inference
import run_episode``) — this package __init__ stays import-light so ``python -m
mosaic_gym.runner.inference`` doesn't double-import.
"""
