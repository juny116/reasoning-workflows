import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from reasoning_workflows.data.answer_io import read_jsonl, solve_math_problems


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_path", type=str, default="grade_school_math/data/test.jsonl"
    )
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="gsm.jsonl")
    args.add_argument("--max_examples", type=int, default=100)
    args = args.parse_args()

    np.random.seed(args.seed)

    data = read_jsonl(args.data_path)

    # np ramdom select max examples
    indices = np.random.choice(range(len(data)), args.max_examples, replace=False)
    data = [data[i] for i in indices]

    # for each example in data solve math problem for answer and add None option
    for d in data:
        d["answer"] = solve_math_problems(d["answer"])
        d["options"] = None

    # dump data as jsonl
    with open(args.out_file, "w") as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")
