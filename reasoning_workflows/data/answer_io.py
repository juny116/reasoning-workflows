import re
import json


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def convert_to_number(string):
    string = string.replace("$", "")
    string = string.replace(",", "")
    try:
        float_n = float(string)
        int_n = int(float_n)
    except ValueError:
        return None

    if float_n == int_n:
        return int_n
    else:
        return float_n

def parse_option(input_str):
    pattern = r"\((\w)\)"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution


def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None
