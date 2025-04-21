# %%
import os
import sys
import pandas as pd
import numpy as np
import torch
import random
import logging
import gc
import re
import json
import time
from pathlib import Path
from tqdm.notebook import tqdm # Use notebook tqdm for better display
from typing import Dict, List, Optional, Any # Added Any

from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig # If considering quantization
)
from peft import PeftModel


# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# %%
class InferenceConfig:
    """Configuration settings for the inference script."""
    # --- Model Identification ---
    base_model_name: str = "google/gemma-3-1b-it"
    # Dictionary mapping user-friendly names to model paths
    # Baseline uses the base model name, adapters use relative paths
    model_options: Dict[str, str] = {
        "baseline": base_model_name,
        "lora_r4": "models/gemma3_finetuned_20250421_163503_r4_lr2e-5/final_adapter",
        "lora_r8": "models/gemma3_finetuned_20250421_164347_r8_lr2e-5/final_adapter",
        "lora_r16": "models/gemma3_finetuned_20250421_163914_r16_lr2e-5/final_adapter",
    }

    # --- Dataset ---
    dataset_name: str = "gretelai/synthetic_text_to_sql"
    # Used only if selecting example by index
    seed: int = 42

    # --- Generation Parameters ---
    max_new_tokens: int = 256 # Max tokens for generated SQL + Explanation
    temperature: float = 0.1 # Low temperature for more deterministic output
    do_sample: bool = False # Use greedy decoding for consistency
    max_seq_length: int = 512 # Max overall sequence length (for input truncation)

    # --- Environment ---
    use_mps_fallback: bool = True # Enable MPS fallback if needed
    hf_token: Optional[str] = None # Loads from .env if None


# %%
def setup_inference_environment(config: InferenceConfig):
    """Set up seeds, device, and MPS fallback."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if config.use_mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("PYTORCH_ENABLE_MPS_FALLBACK enabled.")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")

    return device


# %%
def load_inference_tokenizer(config: InferenceConfig):
    """Loads the tokenizer."""
    logger.info(f"Loading tokenizer: {config.base_model_name}")
    try:
        tokenizer_load_params = {}
        if config.hf_token: tokenizer_load_params['token'] = config.hf_token
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, **tokenizer_load_params)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token.")
        return tokenizer
    except Exception as e:
        logger.error(f"Fatal: Failed to load tokenizer. Error: {e}")
        raise # Re-raise critical error


# %%
def load_inference_model(selected_model_path: str, config: InferenceConfig, device):
    """Loads the selected model (base or PEFT adapter)."""
    is_adapter = (selected_model_path != config.base_model_name)
    model_id_to_load = config.base_model_name # Always load base first

    logger.info(f"Loading base model: {model_id_to_load}")

    # Determine dtype
    model_dtype = torch.float16 if device.type != "cpu" else torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16

    model_load_params = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if config.hf_token:
        model_load_params["token"] = config.hf_token

    # Load base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_to_load,
            **model_load_params
        )
    except Exception as e:
        logger.error(f"Fatal: Failed to load base model {model_id_to_load}. Error: {e}")
        raise

    if is_adapter:
        logger.info(f"Loading adapter: {selected_model_path}")
        try:
            if not os.path.isdir(selected_model_path):
                 raise FileNotFoundError(f"Adapter directory not found: {selected_model_path}")
            # Load PEFT model by applying adapter to the base model
            model = PeftModel.from_pretrained(model, selected_model_path)
            logger.info(f"Successfully loaded adapter from {selected_model_path}")
            # Optional: Merge adapter if desired (faster inference, more memory)
            # logger.info("Merging adapter...")
            # model = model.merge_and_unload()
            # logger.info("Adapter merged.")
        except Exception as e:
            logger.error(f"Failed to load adapter from {selected_model_path}. Error: {e}")
            logger.warning("Proceeding with base model only.")
            # Model variable already holds the base model
    else:
        logger.info("Using base model directly.")

    model = model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info("Model loaded and set to evaluation mode.")
    return model


# %%
def get_example_from_dataset(config: InferenceConfig, index: int):
    """Loads a specific example from the test dataset."""
    try:
        dataset = load_dataset(config.dataset_name)
        split_name = 'test'
        if split_name not in dataset:
            split_name = 'train'
            logger.warning(f"'test' split not found, using '{split_name}' split.")

        data_split = dataset[split_name]
        if not 0 <= index < len(data_split):
            logger.error(f"Index {index} out of range for {split_name} split (size {len(data_split)}).")
            return None, None # Return None for example and reference

        example = data_split[index]
        reference_sql = example.get('sql', None)
        logger.info(f"Loaded example {index} from {split_name} split.")
        return example, reference_sql
    except Exception as e:
        logger.error(f"Error loading dataset example: {e}")
        return None, None


# %%
def format_inference_prompt(tokenizer, prompt: str, context: Optional[str] = None):
    """Formats the prompt using the chat template for generation."""
    user_message = f"Generate the SQL query for the following request"
    if context:
        user_message += f" based on the provided context.\n\nRequest: {prompt}\n\nDatabase Context:\n{context}"
    else:
        user_message += f".\n\nRequest: {prompt}"

    messages = [{"role": "user", "content": user_message}]
    try:
        # Apply chat template, ensuring it adds the prompt for the model to respond
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Crucial for inference
        )
    except Exception as e:
        logger.warning(f"tokenizer.apply_chat_template failed: {e}. Using manual format.")
        formatted_prompt = f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"

    return formatted_prompt


# %%
def generate_text(model, tokenizer, prompt_text: str, config: InferenceConfig, device):
    """Generates text using the specified model and configuration."""
    logger.info("Starting generation...")
    start_time = time.time()

    # Prepare generation config
    gen_config_params = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    if config.do_sample:
        gen_config_params["temperature"] = config.temperature
        gen_config_params["do_sample"] = True
    else:
        gen_config_params["do_sample"] = False
    generation_config = GenerationConfig(**gen_config_params)
    logger.info(f"Using GenerationConfig: {generation_config}")


    # Tokenize input
    max_input_length = config.max_seq_length - config.max_new_tokens
    if max_input_length <= 0:
        logger.error(f"max_seq_length ({config.max_seq_length}) too small for max_new_tokens ({config.max_new_tokens}).")
        return "Error: Configuration invalid (max_seq_length too small).", 0
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    ).to(device)

    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        inference_time = time.time() - start_time
        logger.info(f"Generation finished in {inference_time:.2f} seconds.")

        # Decode generated part
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text, inference_time

    except Exception as e:
        logger.error(f"Error during model.generate: {e}", exc_info=True)
        return f"Error during generation: {e}", time.time() - start_time



# %%
def extract_sql_from_output(generated_text: str):
    """Extracts the SQL query from the model's generated text."""
    logger.debug(f"Attempting to extract SQL from: {generated_text[:200]}...")
    # Try to find "SQL:" prefix first
    sql_match = re.search(r"SQL:\s*(.*?)(?:\nExplanation:|$)", generated_text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        extracted_sql = sql_match.group(1).strip()
        logger.debug("Found SQL using 'SQL:' prefix.")
    else:
        # Fallback logic if "SQL:" prefix is missing
        potential_sql = generated_text.strip()
        if "\nExplanation:" in potential_sql:
             extracted_sql = potential_sql.split("\nExplanation:", 1)[0].strip()
             logger.debug("Found SQL by splitting at 'Explanation:'.")
        elif "\n\n" in potential_sql: # Check for double newline as separator
             extracted_sql = potential_sql.split("\n\n", 1)[0].strip()
             logger.debug("Found SQL by splitting at double newline.")
        else: # Otherwise take the first line
             extracted_sql = potential_sql.split('\n')[0].strip()
             logger.debug("Using first line as SQL (fallback).")
    # Final cleanup (remove potential trailing markdown/code fences)
    extracted_sql = re.sub(r"```$", "", extracted_sql.strip()).strip()
    return extracted_sql


# %%
def run_interactive_inference():
    """Handles user interaction for model/input selection and runs inference."""
    config = InferenceConfig()
    logger.info(f"Inference config: {vars(config)}")

    if config.hf_token is None:
        load_dotenv()
        config.hf_token = os.getenv('HF_TOKEN')
        if config.hf_token: logger.info("Loaded HF Token from .env")

    device = setup_inference_environment(config)
    tokenizer = load_inference_tokenizer(config)

    # --- Model Selection ---
    print("\nAvailable models:")
    model_choices = list(config.model_options.keys())
    for i, name in enumerate(model_choices):
        print(f"{i+1}. {name}")

    while True:
        try:
            model_choice_idx = int(input(f"Select model (1-{len(model_choices)}): ")) - 1
            if 0 <= model_choice_idx < len(model_choices):
                selected_model_key = model_choices[model_choice_idx]
                selected_model_path = config.model_options[selected_model_key]
                logger.info(f"User selected model: {selected_model_key} ({selected_model_path})")
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # --- Load Selected Model ---
    model = None
    try:
        model = load_inference_model(selected_model_path, config, device)
    except Exception as e:
        logger.error(f"Fatal error loading selected model: {e}", exc_info=True)
        return # Exit if model loading fails

    # --- Input Selection ---
    print("\nSelect input method:")
    print("1. Custom prompt (and optional context)")
    print("2. Example from test dataset by index")

    input_text = None
    reference_sql = None
    example_details = {}

    while True:
        try:
            input_choice = input("Select input method (1-2): ").strip()
            if input_choice == '1':
                prompt = input("Enter the SQL prompt/request: ")
                context = input("Enter the SQL context (CREATE TABLE... statements) [Optional, press Enter to skip]: ")
                input_text = format_inference_prompt(tokenizer, prompt.strip(), context.strip() if context else None)
                example_details['prompt'] = prompt
                example_details['context'] = context if context else "None provided"
                break
            elif input_choice == '2':
                try:
                    idx = int(input(f"Enter test dataset index (0 to ~5850): "))
                    example, reference_sql = get_example_from_dataset(config, idx)
                    if example:
                        input_text = format_inference_prompt(tokenizer, example['sql_prompt'], example['sql_context'])
                        example_details['prompt'] = example['sql_prompt']
                        example_details['context'] = example['sql_context']
                        example_details['index'] = idx
                        break
                    else:
                        print("Failed to load example. Please try again.")
                        # Loop continues
                except ValueError:
                    print("Invalid index. Please enter a number.")
            else:
                print("Invalid choice.")
        except EOFError: # Handle cases where input stream ends unexpectedly
             logger.error("Input stream closed unexpectedly.")
             return


    # --- Run Inference ---
    if input_text and model:
        print("\n" + "="*80)
        print(f"Running inference with model: {selected_model_key}")
        print(f"Input Prompt:\n{example_details.get('prompt', 'N/A')}")
        print(f"Input Context:\n{example_details.get('context', 'N/A')}")
        print("="*80)

        generated_text, inference_time = generate_text(model, tokenizer, input_text, config, device)

        print(f"\n--- Raw Model Output (took {inference_time:.2f}s) ---")
        print(generated_text)
        print("-" * 80)

        extracted_sql = extract_sql_from_output(generated_text)
        print(f"\n--- Extracted SQL ---")
        print(extracted_sql)
        print("-" * 80)

        if reference_sql:
            print(f"\n--- Reference SQL (Example {example_details.get('index', 'N/A')}) ---")
            print(reference_sql)
            print("-" * 80)

    elif not model:
         logger.error("Model was not loaded successfully. Cannot run inference.")
    else:
         logger.error("Input text was not prepared successfully. Cannot run inference.")


    # --- Cleanup ---
    logger.info("Cleaning up...")
    del model
    gc.collect()
    if device.type == "mps": torch.mps.empty_cache()
    elif device.type == "cuda": torch.cuda.empty_cache()
    logger.info("Inference complete.")



# %%
if __name__ == "__main__":
     # Check if running in an interactive environment (like Jupyter)
    if 'get_ipython' in globals() or 'google.colab' in sys.modules or os.environ.get("IPYKERNEL_CELL_NAME"):
        run_interactive_inference()
    else:
        logger.info("Script appears to be running in a non-interactive environment.")
        logger.info("Running interactive inference...")
        run_interactive_inference()



