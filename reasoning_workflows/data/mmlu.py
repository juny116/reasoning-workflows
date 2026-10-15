import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from reasoning_workflows.data.answer_io import parse_option


def parse_mmlu(df, ix):
    question = df.iloc[ix, 0]

    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    options = {"A": a, "B": b, "C": c, "D": d}
    answer = df.iloc[ix, 5]

    return {"question": question, "answer": answer, "options": options}


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="mmlu/test")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_dir", type=str, default="mmlu/test")

    args = args.parse_args()

    np.random.seed(args.seed)

    tasks = glob(f"{args.data_dir}/*.csv")
    task_names = [task.split("/")[-1].split(".")[0] for task in tasks]
    task_names = [' '.join(task.split('_')[:-1]) for task in task_names]
    dfs = [pd.read_csv(task, header=None) for task in tasks]
    tasks_with_options = []
    data_per_task = {}
    for t, d in zip(task_names, dfs):
        data_with_options = []

        try:
            for i in range(len(d)):
                data_with_options.append(parse_mmlu(d, i))
            tasks_with_options.append(t)
            data_per_task[t] = data_with_options
        except:
            continue

    for k, data in data_per_task.items():
        # dump data as jsonl
        with open(f"{args.out_dir}/{k}.jsonl", "w") as f:
            for d in data:
                json.dump(d, f)
                f.write("\n")
