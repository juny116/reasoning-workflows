import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from reasoning_workflows.data.answer_io import read_jsonl, solve_math_problems, parse_answer, convert_to_number
import random


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="samples/gsm/chatGPT.jsonl")
    args.add_argument("--seed", type=int, default=21)
    args.add_argument("--out_file", type=str, default="gsm_chat.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    # load dataset
    data = read_jsonl(args.data_path)

    new_data = []

    valid_solutions = 0
    correct_solutions = 0
    unique_answers = 0
    converged = 0
    count_case = 0
    for d in data:
        answer_set = set()
        correct = False
        # get 2 correct solutions and 2 incorrect solutions
        correct_pred = []
        incorrect_pred = []
        selected_answer_set = set()
        for solution in d["pred"]:
            if parse_answer(solution) != None:
                ans = convert_to_number(parse_answer(solution))

                valid_solutions += 1
                answer_set.add(ans)
                if ans == convert_to_number(d["gold"]):
                    correct_solutions += 1
                    if len(correct_pred) < 2:
                        correct_pred.append(solution)
                        selected_answer_set.add(ans)
                else:
                    if len(incorrect_pred) < 2 and ans not in selected_answer_set:
                        incorrect_pred.append(solution)
                        selected_answer_set.add(ans)
        # check if converged
        if len(answer_set) == 1:
            converged += 1
        if len(answer_set) > 2 and len(correct_pred) == 2:
            count_case += 1
        unique_answers += len(answer_set)

        if (
            len(correct_pred) == 2
            and len(incorrect_pred) == 2
            and len(selected_answer_set) == 3
        ):
            # add to data
            correct_path_correct_answer = correct_pred[0]
            incorrect_path_wrong_answer = incorrect_pred[0]
            correct_answer_2 = parse_answer(correct_pred[1])
            incorrect_answer_2 = parse_answer(incorrect_pred[1])
            correct_path_wrong_answer = correct_pred[1].replace(
                correct_answer_2, incorrect_answer_2
            )
            incorrect_path_correct_answer = incorrect_pred[1].replace(
                incorrect_answer_2, correct_answer_2
            )
            pred = [
                correct_path_correct_answer,
                correct_path_wrong_answer,
                incorrect_path_correct_answer,
                incorrect_path_wrong_answer,
            ]
            x = list(enumerate(pred))
            random.shuffle(x)
            indices, shuffled_pred = zip(*x)
            new_data.append(
                {
                    "question": d["question"],
                    "gold": d["gold"],
                    "pred": shuffled_pred,
                    "indices": indices,
                }
            )

    # average valid solutions
    average_valid_solutions = valid_solutions / len(data)
    # average correct solutions
    average_correct_solutions = correct_solutions / len(data)
    # average unique answers
    average_unique_answers = unique_answers / len(data)

    print(f"Average valid solutions: {average_valid_solutions}")
    print(f"Average correct solutions: {average_correct_solutions}")
    print(f"Average unique answers: {average_unique_answers}")
    print(f"Converged: {converged}")
    print(f"Count case: {count_case}")

    print(f"Total data: {len(new_data)}")
    # dump data as jsonl
    with open(args.out_file, "w") as f:
        for d in new_data:
            json.dump(d, f)
            f.write("\n")
