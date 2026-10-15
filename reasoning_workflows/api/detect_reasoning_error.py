from reasoning_workflows.models import OpenAIGPT, GoogleGemini, AutoModelForAPI
import asyncio
import json
from tqdm import tqdm
from reasoning_workflows.configuration import configured
from omegaconf import DictConfig
from reasoning_workflows.data import read_jsonl
from reasoning_workflows.dict_messages import messages_to_string
import os
import time
import logging

logger = logging.getLogger("httpx")
logger.propagate = False


@configured
def main(config: DictConfig) -> None:
    max_examples = config["max_examples"]
    # load model
    model_config = config["model"]
    model = AutoModelForAPI(
        model_type=model_config["model_type"],
        api_key=model_config["api_key"],
        model=model_config["model_name_or_path"],
        base_url=model_config["base_url"],
    )

    instruction = config["template"]["instruction"]
    form = config["template"]["format"]

    prompt = instruction

    test_data = read_jsonl(config["test_path"])
    # if the result file exists, read it
    if os.path.exists(config["out_path"]):
        results = read_jsonl(config["out_path"])
    else:
        results = []
    # if len(results) > 0 continue from last checkpoint
    for i, d in enumerate(tqdm(test_data[:max_examples])):
        if len(results) > 0 and i < len(results):
            continue
        answers = []
        token_lengths = []
        for p in d["pred"]:
            query = form.replace("[QUESTION]", d["question"]).replace("[ANSWER]", p)
            context = query + prompt
            messages = [
                {
                    "role": "user",
                    "content": context,
                },
            ]
            if i == 0:
                print(messages_to_string(messages))
                print("=====================================")
            result = model(
                messages,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
            )
            token_len = result["tokens"]["prompt_tokens"]
            gen_result = result["messages"][-1]["content"]
            token_lengths.append(token_len)
            answers.append(gen_result)

        results.append(
            {
                "question": d["question"],
                "input": messages[0]["content"],
                "input_tokens": result["tokens"]["prompt_tokens"],
                "pred": d["pred"],
                "error": answers,
                "token_lengths": token_lengths,
                "gold": d["gold"],
                "options": d["options"],
                "extract": d["extract"],
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
