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

    while True:
        query = input("Enter the query: ")
        if query == "exit":
            break
        query = f"<|begin_of_text|>1<|start_header_id|>user<|end_header_id|>2{query}3<|eot_id|>4<|start_header_id|>assistant<|end_header_id|>5"
        messages = [
            {
                "role": "user",
                "content": query,
            },
        ]
        result = model(messages)
        print(result["messages"][-1]["content"])
        print("=====================================")


if __name__ == "__main__":
    main()
