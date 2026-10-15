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
    final_inst = config["template"]["final"]
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
        reasoning_steps = []
        feedbacks = []
        final = False
        for r in range(0, config["rounds"]):
            if final:
                break
            # gen step
            reasoning_step_str = "\n".join(reasoning_steps)
            query = form.replace("[QUESTION]", d["question"]).replace(
                "[ANSWER]", reasoning_step_str
            )
            context = prompt + query
            messages = [
                {
                    "role": "user",
                    "content": context,
                },
            ]
            result = model(
                messages,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                stop=["\n"],
            )
            gen_result = result["messages"][-1]["content"]
            reasoning_steps.append(gen_result)
            if i == 0:
                print(messages_to_string(messages))
                print("=====================================")

            # feedback step
            while True:
                reasoning_step_str = "\n".join(reasoning_steps[:-1])
                query = (
                    feedback_inst.replace("[QUESTION]", d["question"])
                    .replace("[ANSWER]", reasoning_step_str)
                    .replace("[CURRENT_STEP]", reasoning_steps[-1])
                )
                context = query
                messages = [
                    {
                        "role": "user",
                        "content": context,
                    },
                ]
                # feedback
                feedback_result = model(
                    messages,
                    max_tokens=model_config["max_tokens"],
                    temperature=model_config["temperature"],
                )
                gen_result = feedback_result["messages"][-1]["content"]
                feedbacks.append(gen_result)
                if i == 0:
                    print(messages_to_string(messages))
                    print("=====================================")
                # generate next step if ACCURATE
                if "[ACCURATE]" in gen_result:
                    # check if final
                    reasoning_step_str = "\n".join(reasoning_steps)
                    query = final_inst.replace("[QUESTION]", d["question"]).replace(
                        "[ANSWER]", reasoning_step_str
                    )
                    context = query
                    messages = [
                        {
                            "role": "user",
                            "content": context,
                        },
                    ]
                    if_final = model(
                        messages,
                        max_tokens=model_config["max_tokens"],
                        temperature=0,
                        stop=["\n"],
                    )
                    gen_result = if_final["messages"][-1]["content"]
                    if i == 0:
                        print(messages_to_string(messages))
                        print("=====================================")
                    # break if final
                    if "[FINAL]" in gen_result:
                        final = True
                    break
                else:
                    # revise
                    reasoning_step_str = "\n".join(reasoning_steps[:-1])
                    revise_query = (
                        revise_inst.replace("QUESTION", d["question"])
                        .replace("[ANSWER]", reasoning_step_str)
                        .replace("[CURRENT_STEP]", reasoning_steps[-1])
                        .replace("[FEEDBACK]", gen_result)
                    )
                    context = revise_query
                    messages = [
                        {
                            "role": "user",
                            "content": context,
                        },
                    ]
                    revised_result = model(
                        messages,
                        max_tokens=model_config["max_tokens"],
                        temperature=model_config["temperature"],
                        stop=["\n"],
                    )
                    gen_result = revised_result["messages"][-1]["content"]
                    reasoning_steps[-1] = gen_result
                    if i == 0:
                        print(messages_to_string(messages))
                        print("=====================================")

        results.append(
            {
                "question": d["question"],
                "input": messages[0]["content"],
                "input_tokens": result["tokens"]["prompt_tokens"],
                "pred": reasoning_steps,
                "gold": d["answer"],
                "options": d["options"],
                "feedback": feedbacks,
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
