import json
import random
from tqdm import tqdm
import random
from reasoning_workflows.configuration import configured
from omegaconf import DictConfig
from reasoning_workflows.data import read_jsonl, parse_answer
from reasoning_workflows.dict_messages import messages_to_string
import os
import time
from reasoning_workflows.models import AutoModelForAPI
import logging


logger = logging.getLogger("httpx")
logger.propagate = False


@configured
def main(config: DictConfig) -> None:
    # set seed
    max_examples = 9999
    # load model
    model_config = config["model"]
    model = AutoModelForAPI(
        model_type=model_config["model_type"],
        api_key=model_config["api_key"],
        model=model_config["model_name_or_path"],
        base_url=model_config["base_url"],
    )
    # max_examples = 100
    instruction = config["template"]["instruction"]
    form = config["template"]["format"]
    train_data = read_jsonl(config["train_path"])
    test_data = read_jsonl(config["test_path"])
    prompt = instruction
    for d in train_data:
        prompt += form.replace("[SOLUTION]", d["pred"][0]).replace(
            "[ANSWER]", f' {d["extract"]}'
        )
        prompt += "\n\n"

    # if the result file exists, read it
    if os.path.exists(config["out_path"]):
        results = read_jsonl(config["out_path"])
    else:
        results = []
    # if len(results) > 0 continue from last checkpoint
    start = True
    for i, d in enumerate(tqdm(test_data[:max_examples])):
        if len(results) > 0 and i < len(results):
            continue
        answers = []
        for pred in d["pred"]:
            query = form.replace("[SOLUTION]", pred).replace("[ANSWER]", "")
            context = prompt + query
            messages = [
                {
                    "role": "user",
                    "content": context,
                },
            ]
            if start:
                print(messages_to_string(messages))
                print("================ EOE =====================")
                start = False
            result = model(
                messages,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
            )
            result = result["messages"][-1]["content"]
            result = result.split("\n")[0].replace("$", "").strip()
            answers.append(result)

        results.append(
            {
                "question": d["question"],
                "input_tokens": d["input_tokens"],
                "pred": d["pred"],
                "token_lengths": d["token_lengths"],
                "gold": d["gold"],
                "options": d["options"],
                "extract": answers,
            }
        )
        # write results to jsonl
        out_path = config["out_path"]
        with open(out_path, "w") as f:
            for d in results:
                json.dump(d, f)
                f.write("\n")


if __name__ == "__main__":
    main()
