"""Read one value out of a config YAML — for the serving sbatch (bash).

    python -m mosaic_gym.config get model.serving.max_model_len configs/qwen3vl_mmiu.yaml
"""

import sys

import yaml


def _get(path: str, dotted: str):
    cur = yaml.safe_load(open(path)) or {}
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "get":
        value = _get(sys.argv[3], sys.argv[2])
        print("" if value is None else value)
    else:
        sys.stderr.write("usage: python -m mosaic_gym.config get <dotted.path> <yaml>\n")
        raise SystemExit(2)
