import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from reasoning_workflows.data.answer_io import parse_option


def parse_bbh(d):
    question = d["examples"]["input"]
    options = question.split("Options:", 2)[1].strip()
    options = options.split("\n")
    options = [parse_option(o) for o in options if o != ""]
    answer = d["examples"]["target"]
    answer = parse_option(answer)

    return {"question": question, "answer": answer, "options": options}


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="BIG-Bench-Hard/bbh")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="bbh.jsonl")
    args.add_argument("--max_examples_per_task", type=int, default=5)
    args.add_argument("--max_examples", type=int, default=100)

    args = args.parse_args()

    np.random.seed(args.seed)

    tasks = glob(f"{args.data_dir}/*.json")
    task_names = [task.split("/")[-1].split(".")[0] for task in tasks]
    dfs = [pd.read_json(task) for task in tasks]
    tasks_with_options = []
    dfs_with_options = []
    data = []
    for t, d in zip(task_names, dfs):
        try:
            questions, options = d.iloc[0]["examples"]["input"].split("Options:", 2)
            options = options.strip().split("\n")
            if parse_option(options[0]) is None:
                # change raw options to A, B options
                for instance in d["examples"]:
                    questions, options = instance["input"].split("Options:", 2)
                    options = options.strip().split("\n")
                    options = [o.replace("-", "").strip() for o in options]
                    options = [f"({chr(65+i)}) {o}" for i, o in enumerate(options)]
                    # get alphabet of answer
                    answer = instance["target"]
                    answer_option = None
                    for o in options:
                        if answer in o:
                            answer_option = o[:3]
                            break
                    instance["input"] = f"{questions}Options:\n" + "\n".join(options)
                    instance["target"] = answer_option
            tasks_with_options.append(t)
            dfs_with_options.append(d)
        except:
            continue
    print(tasks_with_options)
    for t, d in zip(tasks_with_options, dfs_with_options):
        indices = np.random.choice(
            range(len(d)), args.max_examples_per_task, replace=False
        )
        dfs_list = d.iloc[indices].to_dict(orient="records")
        for instance in dfs_list:
            instance["task"] = t
            data.append(parse_bbh(instance))

    # dump data as jsonl
    with open(args.out_file, "w") as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")
