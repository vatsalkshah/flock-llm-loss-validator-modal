import os
import modal
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import time
from client.fed_ledger import FedLedger
from validate import validate

# Load environment variables
FLOCK_API_KEY= ""
HF_TOKEN = ""
TASK_ID = ""
GPU = "A100-40GB"

# Define Modal stub and images
app = modal.App("llm-loss-validator")
model_cache_volume = modal.Volume.from_name("flock-validator-models")

# Base image with common dependencies
base_image = ( modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "requests==2.31.0",
        "loguru==0.7.0",
        "python-dotenv",
        "tenacity",
        "torch>=1.13.1",
        "huggingface-hub>=0.24.7,<0.25",
        "transformers>=4.43.0,<=4.45.0",
        "datasets>=2.14.3",
        "accelerate>=0.27.2",
        "peft>=0.10.0",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "einops",
        "transformers_stream_generator",
        "clock"
    )
    .env({
        "FLOCK_API_KEY" : FLOCK_API_KEY,
        "HF_TOKEN" : HF_TOKEN,
        "TASK_ID" : TASK_ID
    })
)
# Create a GPU-enabled container image
gpu_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch>=1.13.1",
        "huggingface-hub>=0.24.7,<0.25",
        "transformers>=4.43.0,<=4.45.0",
        "datasets>=2.14.3",
        "accelerate>=0.27.2",
        "peft>=0.10.0",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "einops",
        "transformers_stream_generator",
    )
    .pip_install(
        "loguru==0.7.0",
        "tqdm==4.62.3",
        "click",
        "tenacity",
        "python-dotenv",
    )
    .env({
        "FLOCK_API_KEY" : FLOCK_API_KEY,
        "HF_TOKEN" : HF_TOKEN,
        "TASK_ID" : TASK_ID
    })
)

@app.function(
    image=base_image,
    schedule=modal.Period(seconds=189),
    timeout=7200
)
def check_for_validation_task():
    """CPU function that checks for validation tasks"""
    from client.fed_ledger import FedLedger
    
    if not FLOCK_API_KEY:
        raise ValueError("FLOCK_API_KEY environment variable is not set")
    
    client = FedLedger(FLOCK_API_KEY)
    
    # Request a validation assignment
    response = client.request_validation_assignment(TASK_ID)
    
    if response.status_code == 200:
        assignment_data = response.json()
        if assignment_data.get("data"):
            # We have a model to validate - trigger GPU validation
            logger.info(f"Found model to validate: {assignment_data}")
            validate_model.remote(assignment_data)
        else:
            logger.info("No models to validate at this time")
    else:
        logger.error(f"Error requesting validation assignment: {response.text}")

@app.function(
    image=gpu_image,
    gpu=GPU,
    timeout=7200,
    volumes={"/data" : model_cache_volume}
)
def validate_model(assignment_data):
    """GPU function that performs the actual validation"""
    from validate import validate
    import json
    
    try:
        
        # Extract necessary parameters from assignment
        model_info = assignment_data["task_submission"]["data"]
        assignment_id = assignment_data["id"]

        # Run validation
        result = validate(
            model_name_or_path=model_info["hg_repo_id"],
            base_model=model_info["base_model"],
            eval_file=assignment_data["data"]["validation_set_url"],
            context_length=assignment_data["data"]["context_length"],
            max_params=assignment_data["data"]["max_params"],      
            assignment_id=assignment_id,
            local_test=False,
            lora_only=False
        )

        logger.info(f"Validation completed for assignment {assignment_id}")
        
        logger.info(f"Validation completed for assignment {assignment_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        fed_ledger = FedLedger(FLOCK_API_KEY)
        fed_ledger.mark_assignment_as_failed(assignment_id)

if __name__ == "__main__":
    app.run()