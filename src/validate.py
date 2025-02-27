import json
import os
import time
import shutil

import gc
import click
import torch
import requests
import tempfile
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    file_utils,
)

from dotenv import load_dotenv
from pathlib import Path
from core.collator import SFTDataCollator
from core.dataset import UnifiedSFTDataset
from core.template import template_dict
from core.hf_utils import download_lora_config, download_lora_repo
from core.gpu_utils import get_gpu_type
from core.constant import SUPPORTED_BASE_MODELS
from core.exception import (
    handle_os_error,
    handle_runtime_error,
    handle_value_error,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from client.fed_ledger import FedLedger
from peft import PeftModel
import sys
import math
import modal
model_cache_volume = modal.Volume.from_name("flock-validator-models")

load_dotenv()
TIME_SLEEP = int(os.getenv("TIME_SLEEP", 60 * 3))
ASSIGNMENT_LOOKUP_INTERVAL = 60 * 3  # 3 minutes
FLOCK_API_KEY = os.getenv("FLOCK_API_KEY")
if FLOCK_API_KEY is None:
    raise ValueError("FLOCK_API_KEY is not set")
LOSS_FOR_MODEL_PARAMS_EXCEED = 999.0
HF_TOKEN = os.getenv("HF_TOKEN")
IS_DOCKER_CONTAINER = os.getenv("IS_DOCKER_CONTAINER", False)

if HF_TOKEN is None:
    raise ValueError(
        "You need to set HF_TOKEN to download some gated model from HuggingFace"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def download_file(url):
    try:
        # Send a GET request to the signed URL
        response = requests.get(url, stream=True)
        # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        # Create a temporary file to save the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write the content to the temp file in binary mode
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)

            # move the file pointer to the beginning of the file
            temp_file.flush()
            temp_file.seek(0)

            # get the file path
            file_path = temp_file.name
            logger.info(f"Downloaded the file to {file_path}")

            return file_path

    except requests.exceptions.RequestException as e:
        # Handle any exception that can be raised by the requests library
        logger.error(f"An error occurred while downloading the file: {e}")
        raise e


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if "gemma" in model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
        )

    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f"vocab_size of tokenizer: {tokenizer.vocab_size}")
    return tokenizer


def load_model(
    model_name_or_path: str, lora_only: bool, revision: str, val_args: TrainingArguments
) -> Trainer:
    logger.info(f"Loading model from base model: {model_name_or_path}")

    torch_dtype = torch.float16
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=None,
        cache_dir="/data"
    )
    # check whether it is a lora weight
    if download_lora_config(model_name_or_path, revision):
        logger.info("Repo is a lora weight, loading model with adapter weights")
        with open("lora/adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        base_model = adapter_config["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model, token=HF_TOKEN, **model_kwargs
        )
        # download the adapter weights
        download_lora_repo(model_name_or_path, revision)
        model = PeftModel.from_pretrained(
            model,
            "lora",
            device_map=None,
        )
        model = model.merge_and_unload()
        logger.info("Loaded model with adapter weights")
    # assuming full fine-tuned model
    else:
        if lora_only:
            logger.error(
                "Repo is not a lora weight, but lora_only flag is set to True. Will mark the assignment as failed"
            )
            return None
        logger.info("Repo is a full fine-tuned model, loading model directly")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, token=HF_TOKEN, **model_kwargs
        )

    if "output_router_logits" in model.config.to_dict():
        logger.info("set output_router_logits as True")
        model.config.output_router_logits = True
    logger.info(
        f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB"
    )

    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return model


def load_sft_dataset(
    eval_file: str, max_seq_length: int, template_name: str, tokenizer: AutoTokenizer
) -> UnifiedSFTDataset:
    if template_name not in template_dict.keys():
        raise ValueError(
            f"template_name doesn't exist, all template_name: {template_dict.keys()}"
        )
    template = template_dict[template_name]
    logger.info("Loading data with UnifiedSFTDataset")
    return UnifiedSFTDataset(eval_file, tokenizer, max_seq_length, template)


def clean_model_cache(
    auto_clean_cache: bool, cache_path: str = "/data"
):
    """
    Cleans up the local model cache directory by removing directories that are not
    listed in SUPPORTED_BASE_MODELS.

    Parameters:
    - auto_clean_cache (bool): A flag to determine whether to clean the cache.
    - cache_path (str): The path to the cache directory. Defaults to file_utils.default_cache_path.
    """
    if not auto_clean_cache:
        return

    try:
        cache_path = Path(cache_path)
        for item in cache_path.iterdir():
            if item.is_dir() and item.name.startswith("models"):
                if item.name not in {
                    f"models--{BASE_MODEL.replace('/', '--')}"
                    for BASE_MODEL in SUPPORTED_BASE_MODELS
                }:
                    shutil.rmtree(item)
                    logger.info(f"Removed directory: {item}")
        logger.info("Successfully cleaned up the local model cache")
    except (OSError, shutil.Error) as e:
        logger.error(f"Failed to clean up the local model cache: {e}")

def validate(
    model_name_or_path: str,
    base_model: str,
    eval_file: str,
    context_length: int,
    max_params: int,
    assignment_id: str = None,
    local_test: bool = False,
    lora_only: bool = True,
    revision: str = "main",
):
    if not local_test and assignment_id is None:
        raise ValueError(
            "assignment_id is required for submitting validation result to the server"
        )

    model = None
    eval_dataset = None

    try:
        fed_ledger = FedLedger(FLOCK_API_KEY)
        parser = HfArgumentParser(TrainingArguments)
        val_args = TrainingArguments(
            per_device_eval_batch_size=1,
            fp16=True,
            output_dir=".",
            remove_unused_columns=False,
        )
        gpu_type = get_gpu_type()

        tokenizer = load_tokenizer(model_name_or_path)
        eval_file= download_file(eval_file)
        eval_dataset = load_sft_dataset(
            eval_file, context_length, template_name=base_model, tokenizer=tokenizer
        )
        model = load_model(model_name_or_path, lora_only, revision, val_args)
        # if model is not loaded, mark the assignment as failed and return
        if model is None:
            fed_ledger.mark_assignment_as_failed(assignment_id)
            return
        # if the number of parameters exceeds the limit, submit a validation result with a large loss
        total = sum(p.numel() for p in model.parameters())
        if total > max_params:
            logger.error(
                f"Total model params: {total} exceeds the limit {max_params}, submitting validation result with a large loss"
            )
            if local_test:
                return
            resp = fed_ledger.submit_validation_result(
                assignment_id=assignment_id,
                loss=LOSS_FOR_MODEL_PARAMS_EXCEED,
                gpu_type=gpu_type,
            )
            # check response is 200
            if resp.status_code != 200:
                logger.error(f"Failed to submit validation result: {resp.content}")
            return
        data_collator = SFTDataCollator(tokenizer, max_seq_length=context_length)

        trainer = Trainer(
            model=model,
            args=val_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        eval_result = trainer.evaluate()
        eval_loss = eval_result["eval_loss"]
        logger.info("evaluate result is %s" % str(eval_result))
        if local_test:
            logger.info("The model can be correctly validated by validators.")
            return
        # sometimes the loss might not be a valid float
        if isinstance(eval_loss, float) and (
            math.isnan(eval_loss) or math.isinf(eval_loss)
        ):
            eval_loss = LOSS_FOR_MODEL_PARAMS_EXCEED
        resp = fed_ledger.submit_validation_result(
            assignment_id=assignment_id, loss=eval_loss, gpu_type=gpu_type
        )
        # check response is 200
        if resp.status_code != 200:
            logger.error(f"Failed to submit validation result: {resp.content}")
            if resp.json() == {
                "detail": "Validation assignment is not in validating status"
            }:
                logger.info(
                    "Validation assignment is not in validating status anymore, marking it as failed"
                )
                fed_ledger.mark_assignment_as_failed(assignment_id)
            return
        logger.info(
            f"Successfully submitted validation result for assignment {assignment_id}"
        )

    # raise for exceptions, will handle at `loop` level
    except Exception as e:
        raise e
    finally:
        # offload the model to save memory
        gc.collect()
        if model is not None:
            logger.debug("Offloading model to save memory")
            model.cpu()
            del model
        if eval_dataset is not None:
            logger.debug("Offloading eval_dataset to save memory")
            del eval_dataset
        torch.cuda.empty_cache()
        # remove lora folder
        if os.path.exists("lora"):
            logger.debug("Removing lora folder")
            os.system("rm -rf lora")
