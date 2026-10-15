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
    )

    instruction = config["template"]["instruction"]
    feedback_inst = config["template"]["feedback"]
    revise_inst = config["template"]["revise"]
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
        feedbacks = []
        feedback_token_lengths = []
        query = form.replace("[QUESTION]", d["question"]).replace("[ANSWER]", "")
        context = prompt + query
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
        for r in range(0, config["rounds"]):
            # set 30 second timeout
            query = form.replace("[QUESTION]", d["question"]).replace(
                "[ANSWER]", f"{gen_result}"
            )
            context = prompt + query + "\n\n" + feedback_inst
            messages = [
                {
                    "role": "user",
                    "content": context,
                },
            ]
            # feedback
            feedback_result = model(messages)
            token_len = feedback_result["tokens"]["prompt_tokens"]
            gen_result = feedback_result["messages"][-1]["content"]
            feedbacks.append(gen_result)
            feedback_token_lengths.append(token_len)
            if "[ACCURATE]" in gen_result:
                break

            # revise
            revise_query = revise_inst.replace("[FEEDBACK]", gen_result)
            context = prompt + query + "\n\n" + revise_query
            messages = [
                {
                    "role": "user",
                    "content": context,
                },
            ]
            revised_result = model(messages)
            token_len = revised_result["tokens"]["prompt_tokens"]
            gen_result = revised_result["messages"][-1]["content"]
            answers.append(gen_result)
            token_lengths.append(token_len)
            if i == 0:
                print(messages_to_string(messages))
                print(gen_result)
                print("=====================================")

        results.append(
            {
                "question": d["question"],
                "input": messages[0]["content"],
                "input_tokens": result["tokens"]["prompt_tokens"],
                "pred": answers,
                "token_lengths": token_lengths,
                "gold": d["answer"],
                "options": d["options"],
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
