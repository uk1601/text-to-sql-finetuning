# %% Imports
import os
import sys
import pandas as pd
import numpy as np
import torch
import random
import logging
import shutil
import gc
import re
import json
from pathlib import Path
from tqdm.notebook import tqdm # Use notebook tqdm for better display
from typing import Dict, List, Optional, Any # Added Any

# --- Imports for ipywidgets ---
# Removed ipywidgets imports as they are not used in this script version
# --- End ipywidgets imports ---

from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig # If considering quantization
)
from peft import PeftModel
import evaluate # Hugging Face Evaluate library
import sqlparse # For SQL normalization

# %% Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Reduce logging spam from underlying libraries
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
# Keep our own logger at INFO level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# %% Configuration Class (Edit parameters here)
class EvalConfig:
    """Configuration settings for the evaluation script."""
    # --- Model Identification ---
    base_model_name: str = "google/gemma-3-1b-it"
    # List of adapter paths (relative to the script location)
    # Add the paths to the 'final_adapter' directories from your runs
    adapter_paths: Dict[str, str] = {
        "LoRA_r4_lr2e-5": "models/gemma3_finetuned_20250421_163503_r4_lr2e-5/final_adapter",
        "LoRA_r8_lr2e-5": "models/gemma3_finetuned_20250421_164347_r8_lr2e-5/final_adapter",
        "LoRA_r16_lr2e-5": "models/gemma3_finetuned_20250421_163914_r16_lr2e-5/final_adapter",
    }
    # Optionally add more adapters if you run more experiments

    # --- Dataset ---
    dataset_name: str = "gretelai/synthetic_text_to_sql"
    # Use the same test subset size as in training evaluation, or -1 for full test set
    test_subset_size: int = 3 # Reduced for quick testing, set back to 30 or -1
    seed: int = 42 # Seed for subset selection if used

    # --- Evaluation Parameters ---
    metrics_to_compute: List[str] = ["bleu", "rouge", "exact_match"]
    # Generation config
    max_new_tokens: int = 256 # Max tokens to generate for SQL + Explanation
    temperature: float = 0.1 # Set a default positive temperature (will be ignored if do_sample=False)
    do_sample: bool = False # Set to False for deterministic evaluation (greedy)
    # Max length for tokenizer input (prompt part). Should generally match training.
    # Input will be truncated to (max_seq_length - max_new_tokens)
    max_seq_length: int = 512

    # --- Environment ---
    use_mps_fallback: bool = True # Enable MPS fallback if needed
    hf_token: Optional[str] = None # Loads from .env if None

    # --- Output Directory ---
    evaluation_output_dir: str = "evaluation" # Name of the subdirectory for outputs


# %% Setup Environment (No changes needed here)
def setup_eval_environment(config: EvalConfig):
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

# %% Load Test Data (No changes needed here)
def load_test_data(config: EvalConfig):
    """Load and prepare the test dataset subset."""
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name)

    if 'test' not in dataset:
        logger.warning("No 'test' split found. Using 'train' split for evaluation.")
        test_dataset_full = dataset['train']
    else:
        test_dataset_full = dataset['test']

    logger.info(f"Full test set size: {len(test_dataset_full)}")

    # Select subset
    if config.test_subset_size == -1 or config.test_subset_size >= len(test_dataset_full):
        logger.info(f"Using full test dataset ({len(test_dataset_full)} examples).")
        test_dataset = test_dataset_full
    else:
        logger.info(f"Selecting subset of {config.test_subset_size} examples for test dataset.")
        actual_size = min(config.test_subset_size, len(test_dataset_full))
        test_dataset = test_dataset_full.shuffle(seed=config.seed).select(range(actual_size))

    logger.info(f"Using {len(test_dataset)} examples for evaluation.")
    return test_dataset

# %% Prepare Input Prompt (No changes needed here)
def format_input_for_generation(example, tokenizer):
    """Formats the prompt using the chat template for generation."""
    prompt = example['sql_prompt']
    context = example['sql_context']

    user_message = f"Generate the SQL query for the following request based on the provided context.\n\nRequest: {prompt}\n\nDatabase Context:\n{context}"

    # Use apply_chat_template for generation - it should add the prompt structure correctly
    # including the final turn marker for the assistant.
    messages = [
        {"role": "user", "content": user_message}
    ]
    try:
        # Set add_generation_prompt=True to correctly format for prompting the model
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback to manual formatting if template application fails
        logger.warning(f"tokenizer.apply_chat_template failed: {e}. Using manual format.")
        formatted_prompt = f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"

    return formatted_prompt

# %% Load Model Function (No changes needed here)
def load_evaluation_model(model_name_or_path, base_model_name, device, is_adapter=False, hf_token=None):
    """Loads either the base model or a PEFT adapter model."""
    logger.info(f"Loading model: {model_name_or_path}")

    # Determine dtype
    model_dtype = torch.float16 if device.type != "cpu" else torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16

    model_load_params = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager", # Keep consistent with training
    }
    if hf_token:
        model_load_params["token"] = hf_token

    # Load base model first
    # Use try-except for robustness
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_load_params
        )
    except Exception as e:
        logger.error(f"Fatal: Failed to load base model {base_model_name}. Error: {e}")
        raise # Re-raise as this is critical

    if is_adapter:
        # Load PEFT model by applying adapter to the base model
        try:
            # Ensure adapter path exists before attempting to load
            if not os.path.isdir(model_name_or_path):
                 raise FileNotFoundError(f"Adapter directory not found: {model_name_or_path}")
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            logger.info(f"Successfully loaded adapter from {model_name_or_path}")
        except Exception as e:
            logger.error(f"Failed to load adapter from {model_name_or_path}: {e}")
            logger.warning("Returning base model instead due to adapter load failure.")
            model = base_model # Fallback to base model if adapter fails
    else:
        # Using the base model directly
        model = base_model

    model = model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info("Model loaded and set to evaluation mode.")
    return model

# %% Generation Function (No changes needed here)
def generate_predictions(model, tokenizer, dataset, config, device):
    """Generate predictions for the entire dataset."""
    predictions = []
    references = []

    # --- Corrected GenerationConfig setup ---
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
    # --- End of Correction ---


    # Calculate max length for input tokens to leave space for generation
    max_input_length = config.max_seq_length - config.max_new_tokens
    if max_input_length <= 0:
        raise ValueError(f"max_seq_length ({config.max_seq_length}) must be greater than max_new_tokens ({config.max_new_tokens})")

    logger.info(f"Generating predictions for {len(dataset)} examples...")
    logger.info(f"Max input length for tokenizer: {max_input_length}")

    for example in tqdm(dataset, desc="Generating"):
        # 1. Prepare input
        formatted_prompt = format_input_for_generation(example, tokenizer)
        # Tokenize with truncation based on calculated max_input_length
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length # Use calculated max input length
        ).to(device)

        # 2. Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # 3. Decode and Extract SQL
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        sql_match = re.search(r"SQL:\s*(.*?)(?:\nExplanation:|$)", generated_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            extracted_sql = sql_match.group(1).strip()
        else:
            potential_sql = generated_text.strip()
            if "\nExplanation:" in potential_sql:
                 extracted_sql = potential_sql.split("\nExplanation:", 1)[0].strip()
            elif "\n\n" in potential_sql:
                 extracted_sql = potential_sql.split("\n\n", 1)[0].strip()
            else:
                 extracted_sql = potential_sql.split('\n')[0].strip()
            logger.debug(f"Could not find 'SQL:' prefix. Extracted: {extracted_sql[:100]}...")

        predictions.append(extracted_sql)
        references.append(example['sql']) # Store reference SQL

    logger.info("Generation complete.")
    return predictions, references

# %% Metrics Calculation (No changes needed here)
def normalize_sql(query):
    """Normalize SQL query for comparison."""
    try:
        normalized = sqlparse.format(
            str(query),
            keyword_case='lower',
            identifier_case='lower',
            reindent=True,
            strip_comments=True
        )
        normalized = normalized.strip().rstrip(';')
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    except Exception as e:
        logger.warning(f"SQL normalization failed for query: {str(query)[:100]}... Error: {e}")
        return str(query).strip().rstrip(';').lower()

def compute_metrics(predictions, references, metrics_to_compute):
    """Compute evaluation metrics."""
    results = {}
    logger.info(f"Computing metrics: {metrics_to_compute}")

    logger.info("Normalizing SQL queries for Exact Match...")
    norm_predictions = [normalize_sql(p) for p in tqdm(predictions, desc="Normalizing Preds")]
    norm_references = [normalize_sql(r) for r in tqdm(references, desc="Normalizing Refs")]
    logger.info("Normalization complete.")

    if "exact_match" in metrics_to_compute:
        exact_match_count = sum(1 for pred, ref in zip(norm_predictions, norm_references) if pred == ref)
        results["exact_match"] = exact_match_count / len(references) if references else 0
        logger.info(f"Exact Match (Normalized): {results['exact_match']:.4f}")

    if "bleu" in metrics_to_compute:
        try:
            logger.info("Loading BLEU metric...")
            bleu_metric = evaluate.load("bleu")
            bleu_references = [[ref] for ref in references]
            logger.info("Calculating BLEU score...")
            bleu_score = bleu_metric.compute(predictions=predictions, references=bleu_references)
            results["bleu"] = bleu_score['bleu']
            logger.info(f"BLEU Score: {results['bleu']:.4f}")
        except Exception as e:
            logger.error(f"Failed to compute BLEU: {e}")
            results["bleu"] = 0.0

    if "rouge" in metrics_to_compute:
        try:
            logger.info("Loading ROUGE metric...")
            rouge_metric = evaluate.load("rouge")
            logger.info("Calculating ROUGE scores...")
            rouge_score = rouge_metric.compute(predictions=predictions, references=references)
            results.update(rouge_score)
            logger.info(f"ROUGE Scores: R1={results.get('rouge1', 0.0):.4f}, R2={results.get('rouge2', 0.0):.4f}, RL={results.get('rougeL', 0.0):.4f}")
        except ModuleNotFoundError:
             logger.error("ROUGE calculation failed: Missing dependencies. Try `pip install rouge_score absl-py nltk`")
             results.update({"rouge1": "Error", "rouge2": "Error", "rougeL": "Error", "rougeLsum": "Error"})
        except Exception as e:
            logger.error(f"Failed to compute ROUGE: {e}")
            results.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0})

    return results

# %% Main Evaluation Loop
def run_evaluation():
    """Run the full evaluation pipeline."""
    config = EvalConfig()
    logger.info(f"Evaluation config: {vars(config)}")

    # --- Create output directory ---
    eval_output_dir = config.evaluation_output_dir
    os.makedirs(eval_output_dir, exist_ok=True)
    logger.info(f"Evaluation outputs will be saved to: {eval_output_dir}")
    # --- End directory creation ---

    if config.hf_token is None:
        load_dotenv()
        config.hf_token = os.getenv('HF_TOKEN')
        if config.hf_token: logger.info("Loaded HF Token from .env")

    device = setup_eval_environment(config)
    test_dataset = load_test_data(config)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.base_model_name}")
    try:
        tokenizer_load_params = {}
        if config.hf_token: tokenizer_load_params['token'] = config.hf_token
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, **tokenizer_load_params)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token.")
    except Exception as e:
        logger.error(f"Fatal: Failed to load tokenizer. Error: {e}")
        return

    all_results = {}
    models_to_evaluate = {"baseline": config.base_model_name}
    models_to_evaluate.update(config.adapter_paths)

    for model_key, model_path in models_to_evaluate.items():
        logger.info(f"\n--- Evaluating Model: {model_key} ---")
        logger.info(f"Path/Name: {model_path}")

        model = None
        gc.collect()
        if device.type == "mps": torch.mps.empty_cache()
        elif device.type == "cuda": torch.cuda.empty_cache()

        is_adapter = (model_key != "baseline")
        try:
            model = load_evaluation_model(
                model_path,
                config.base_model_name,
                device,
                is_adapter=is_adapter,
                hf_token=config.hf_token
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_key}. Skipping. Error: {e}", exc_info=True)
            all_results[model_key] = {"error": f"Model loading failed: {e}"}
            continue

        try:
            predictions, references = generate_predictions(model, tokenizer, test_dataset, config, device)

            # --- Save predictions to evaluation directory ---
            try:
                pred_df = pd.DataFrame({'reference': references, 'prediction': predictions})
                # Use os.path.join to save inside the evaluation directory
                pred_filename = os.path.join(eval_output_dir, f"predictions_{model_key}.csv")
                pred_df.to_csv(pred_filename, index=False)
                logger.info(f"Saved predictions for {model_key} to {pred_filename}")
            except Exception as e:
                logger.error(f"Failed to save predictions for {model_key}: {e}")
            # --- End save predictions ---

            metrics = compute_metrics(predictions, references, config.metrics_to_compute)
            all_results[model_key] = metrics

        except Exception as e:
            logger.error(f"Error during generation or metric calculation for {model_key}: {e}", exc_info=True)
            all_results[model_key] = {"error": f"Generation/Metrics failed: {e}"}

        del model
        gc.collect()
        if device.type == "mps": torch.mps.empty_cache()
        elif device.type == "cuda": torch.cuda.empty_cache()
        logger.info(f"--- Finished Evaluating Model: {model_key} ---")


    # --- Display Results ---
    logger.info("\n--- Evaluation Summary ---")
    results_df = pd.DataFrame.from_dict(all_results, orient='index')

    float_cols = results_df.select_dtypes(include=['number']).columns
    for col in float_cols:
         if col in results_df.columns:
              results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.number)) else x)

    print(results_df.to_markdown())

    # --- Save results to evaluation directory ---
    # Use os.path.join to save inside the evaluation directory
    results_file = os.path.join(eval_output_dir, "evaluation_results.json")
    try:
        serializable_results = {}
        for model, metrics in all_results.items():
            serializable_results[model] = {k: (float(v) if isinstance(v, (np.number, np.float32, np.float64)) else v) for k, v in metrics.items()}

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"Evaluation results saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}")
    # --- End save results ---

    logger.info("Evaluation script finished.")


# %% Run Evaluation
if __name__ == "__main__":
    # Check if running in an interactive environment (like Jupyter)
    if 'get_ipython' in globals() or 'google.colab' in sys.modules or os.environ.get("IPYKERNEL_CELL_NAME"):
        run_evaluation()
    else:
        logger.info("Script appears to be running in a non-interactive environment.")
        logger.info("Running evaluation...")
        run_evaluation()
