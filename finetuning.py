# %%
import os
import sys
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
import json
import logging
import shutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
# Removed argparse

from dotenv import load_dotenv
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig # If considering quantization later
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training # If using quantization
)


# %%
# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# %%
class Config:
    """Configuration settings for the fine-tuning script."""
    # --- Model and Tokenizer Arguments ---
    model_name: str = "google/gemma-3-1b-it" # Hugging Face model identifier
    hf_token: Optional[str] = None # Hugging Face API token (loads from .env if None)

    # --- Dataset Arguments ---
    dataset_name: str = "gretelai/synthetic_text_to_sql" # Hugging Face dataset identifier
    train_subset_size: int = 5000 # Number of training examples (-1 for all)
    val_subset_size: int = 750    # Number of validation examples (-1 for all)
    test_subset_size: int = 1000   # Number of test examples (-1 for all)
    validation_split_percentage: float = 0.15 # % of train for validation if needed

    # --- Training Hyperparameters ---
    learning_rate: float = 2e-5  # Learning rate
    num_train_epochs: int = 3      # Number of training epochs
    per_device_train_batch_size: int = 1 # Batch size per device (training)
    per_device_eval_batch_size: int = 1  # Batch size per device (evaluation)
    gradient_accumulation_steps: int = 8 # Accumulate gradients over N steps
    warmup_ratio: float = 0.1      # Warmup ratio for scheduler
    weight_decay: float = 0.01     # Weight decay
    max_grad_norm: float = 1.0     # Max gradient norm for clipping

    # --- LoRA Configuration ---
    lora_r: int = 4               # LoRA rank
    lora_alpha: int = 16          # LoRA alpha scaling
    lora_dropout: float = 0.05    # LoRA dropout
    # Modules to target with LoRA (common for Gemma)
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # --- Sequence Length Configuration ---
    max_seq_length: int = 512     # Max sequence length for tokenization

    # --- Environment and Reproducibility ---
    seed: int = 42                # Random seed
    use_mps_fallback: bool = True # Enable MPS fallback (set to False if not needed)

    # --- Output and Logging ---
    # Suffix for output dir (e.g., "_r8_lr2e-5"). Useful for hyperparameter tuning runs.
    output_dir_suffix: str = f"_r{lora_r}_lr2e-5"
    logging_steps: int = 10       # Log every N steps
    eval_steps: int = 25          # Evaluate every N steps
    save_steps: int = 50          # Save checkpoint every N steps
    save_total_limit: int = 1     # Max checkpoints to keep

    # --- Early Stopping ---
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01



# %%
def setup_environment(config: Config):
    """Set up seeds, device, and MPS fallback based on Config."""
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Set MPS fallback if requested
    if config.use_mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("PYTORCH_ENABLE_MPS_FALLBACK enabled.")
        # Optional: Lower high watermark ratio for potentially better memory management on MPS
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device.")
        # Test MPS device
        try:
            x = torch.ones(1, device=device)
            logger.info(f"MPS test successful: {x.cpu().item()}")
        except Exception as e:
            logger.error(f"MPS device test failed: {e}. Check PyTorch installation for MPS support.")
            device = torch.device("cpu") # Fallback to CPU
            logger.info("Falling back to CPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")

    return device


# %%
def load_and_prepare_dataset(config: Config):
    """Load dataset, create splits, and select subsets based on Config."""
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name)

    # Ensure 'train' split exists
    if 'train' not in dataset:
        logger.error("Fatal: 'train' split not found.")
        sys.exit(1)

    train_val_dataset = dataset['train']

    # Create validation split if it doesn't exist or if specified size requires it
    if 'validation' not in dataset or config.val_subset_size != -1:
        logger.info(f"Creating validation split ({config.validation_split_percentage * 100}%) from training data...")
        split_result = train_val_dataset.train_test_split(
            test_size=config.validation_split_percentage,
            seed=config.seed,
            shuffle=True
        )
        train_dataset_full = split_result['train']
        val_dataset_full = split_result['test']
        logger.info(f"Full train size: {len(train_dataset_full)}, Full validation size: {len(val_dataset_full)}")
    else:
        train_dataset_full = train_val_dataset
        val_dataset_full = dataset['validation']
        logger.info(f"Using existing validation split. Full train size: {len(train_dataset_full)}, Full validation size: {len(val_dataset_full)}")

    # Handle test split
    if 'test' in dataset:
        test_dataset_full = dataset['test']
        logger.info(f"Full test size: {len(test_dataset_full)}")
    else:
        test_dataset_full = None
        logger.warning("No test split found in the dataset.")

    # Select subsets
    def select_subset(dataset_full, subset_size, name):
        if dataset_full is None:
            return None
        # Use full dataset if subset_size is -1 or greater/equal to full size
        if subset_size == -1 or subset_size >= len(dataset_full):
            logger.info(f"Using full {name} dataset ({len(dataset_full)} examples).")
            return dataset_full
        else:
            logger.info(f"Selecting subset of {subset_size} examples for {name} dataset.")
            # Ensure subset_size is not larger than the dataset
            actual_size = min(subset_size, len(dataset_full))
            # Select requires indices, shuffle first for random subset
            return dataset_full.shuffle(seed=config.seed).select(range(actual_size))

    train_dataset = select_subset(train_dataset_full, config.train_subset_size, "train")
    val_dataset = select_subset(val_dataset_full, config.val_subset_size, "validation")
    test_dataset = select_subset(test_dataset_full, config.test_subset_size, "test")

    # Create a DatasetDict
    processed_datasets = DatasetDict()
    if train_dataset: processed_datasets['train'] = train_dataset
    if val_dataset: processed_datasets['validation'] = val_dataset
    if test_dataset: processed_datasets['test'] = test_dataset

    logger.info(f"Final dataset sizes: { {k: len(v) for k, v in processed_datasets.items()} }")

    # Clean up large unused datasets
    del dataset, train_val_dataset, train_dataset_full, val_dataset_full, test_dataset_full
    gc.collect()

    return processed_datasets


# %%
def preprocess_and_tokenize(dataset_dict, tokenizer, max_seq_length):
    """Apply chat template, tokenize, and mask labels."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(f"Tokenizer pad_token set to eos_token: {tokenizer.eos_token}")

    # Define the formatting function
    def format_and_prepare(example):
        # --- Construct the prompt using Gemma's chat template structure ---
        prompt = example['sql_prompt']
        context = example['sql_context']
        sql_output = example['sql']
        explanation = example.get('sql_explanation', '') # Use .get for safety
        target_response = f"SQL: {sql_output}"
        if explanation:
            target_response += f"\nExplanation: {explanation}"

        user_message = f"Generate the SQL query for the following request based on the provided context.\n\nRequest: {prompt}\n\nDatabase Context:\n{context}"

        # Apply template manually for precise label masking
        prompt_part = f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
        # Ensure EOS token marks the end of generation for the model to learn
        response_part = f"{target_response}{tokenizer.eos_token}"

        # Tokenize prompt and response parts separately
        # Not adding special tokens here as they are manually included in the template parts
        prompt_tokens = tokenizer(prompt_part, add_special_tokens=False)
        response_tokens = tokenizer(response_part, add_special_tokens=False)

        # Combine for input_ids and attention_mask
        input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids']
        attention_mask = [1] * len(input_ids) # Attention mask should be 1 for all actual tokens

        # Create labels: copy input_ids, then mask the prompt part
        labels = list(input_ids) # Make a mutable copy
        prompt_len = len(prompt_tokens['input_ids'])
        # Mask prompt tokens by setting their labels to -100
        labels[:prompt_len] = [-100] * prompt_len

        # --- Truncation ---
        # Truncate from the right if combined length exceeds max_seq_length
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            labels = labels[:max_seq_length]
            # Ensure the last token isn't a masked token if possible, although unlikely with right truncation
            # if labels[-1] == -100: logger.warning("Truncated sequence ends with a masked token.")


        # Note: Padding is handled dynamically by the DataCollatorForSeq2Seq

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Apply the function to all splits in the DatasetDict
    logger.info("Applying formatting and tokenization...")
    # Determine columns to remove dynamically from the first split found
    first_split_key = next(iter(dataset_dict))
    remove_cols = list(dataset_dict[first_split_key].column_names)
    tokenized_datasets = dataset_dict.map(
        format_and_prepare,
        # Consider batched=True for potential speedup if memory allows and logic is adapted
        # batched=True,
        # batch_size=100, # Adjust batch size if using batched=True
        remove_columns=remove_cols
    )
    logger.info("Tokenization complete.")

    # Log token length statistics after tokenization (optional but helpful)
    for split, dataset in tokenized_datasets.items():
        if len(dataset) > 0: # Check if dataset split is not empty
            lengths = [len(x) for x in dataset['input_ids']]
            logger.info(f"{split.capitalize()} split token lengths: Min={np.min(lengths)}, Mean={np.mean(lengths):.2f}, Max={np.max(lengths)}, Median={np.median(lengths)}")
            # Check if max length was hit often
            truncated_count = sum(1 for length in lengths if length >= max_seq_length)
            if truncated_count > 0:
                logger.warning(f"{truncated_count}/{len(lengths)} examples in {split} split potentially truncated to max_seq_length {max_seq_length}.")
        else:
            logger.info(f"{split.capitalize()} split is empty.")


    return tokenized_datasets


# %%
def load_model_and_lora(config: Config, device):
    """Load the base model and apply LoRA configuration based on Config."""

    # --- Model Justification ---
    # Choosing google/gemma-3-1b-it because:
    # 1. Size: 3.1B parameters is manageable for fine-tuning on consumer hardware (especially MPS/CPU with optimizations).
    # 2. Instruction-Tuned: '-it' indicates it's fine-tuned for following instructions, which is suitable for Text-to-SQL generation.
    # 3. Performance: Gemma models generally show strong performance for their size class.
    # 4. Openness: Relatively open model allows for easier experimentation.
    logger.info(f"Loading base model: {config.model_name}")

    # Determine dtype based on device
    model_dtype = torch.float16 if device.type == "mps" else torch.float32
    if device.type == "cuda":
        # Check CUDA capability for bfloat16
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            logger.info("Using bfloat16 for CUDA.")
        else:
            model_dtype = torch.float16
            logger.info("Using float16 for CUDA (bfloat16 not supported).")

    model_load_params = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True, # Helps when loading large models
        "attn_implementation": "eager", # Use eager attention for Gemma3 compatibility, esp. on MPS/CPU
        "use_cache": False, # Important: Disable use_cache for gradient checkpointing & training
    }
    if config.hf_token:
        model_load_params["token"] = config.hf_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_load_params
    )

    # --- LoRA Configuration ---
    logger.info(f"Configuring LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none", # Typically set to 'none' for LoRA
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules
    )

    # Apply LoRA to the model
    try:
        model = get_peft_model(model, lora_config)
    except ValueError as e:
        logger.error(f"Error applying LoRA. Check target modules: {config.lora_target_modules}. Error: {e}")
        # Attempt to find modules if error suggests mismatch
        logger.info("Available module names containing 'proj':")
        for name, module in model.named_modules():
            if 'proj' in name.lower():
                logger.info(f"- {name}")
        raise e # Re-raise after logging suggestions

    # Print trainable parameters
    model.print_trainable_parameters()

    # Enable gradient checkpointing *after* applying LoRA
    # Use reentrant=False for newer PyTorch versions if compatible, potentially saves more memory
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Move model to device (do this *after* PEFT application and checkpoint enabling)
    logger.info(f"Moving model to device: {device}")
    model = model.to(device)
    logger.info(f"Model loaded, LoRA applied, and moved to {device}.")

    return model


# %%
def train(config: Config, device, tokenized_datasets, tokenizer):
    """Configure and run the training process based on Config."""

    # Load Model with LoRA
    model = load_model_and_lora(config, device)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use a descriptive name including key hyperparams if suffix is provided
    output_dir_name = f"gemma3_finetuned_{timestamp}{config.output_dir_suffix}"
    model_output_dir = os.path.join("models", output_dir_name) # Relative path
    os.makedirs(model_output_dir, exist_ok=True)
    logger.info(f"Model output directory: {model_output_dir}")

    # Setup logging to file within the output directory
    log_file = os.path.join(model_output_dir, "training.log")
    # Remove existing file handler if script is re-run in same session (e.g., notebook)
    root_logger = logging.getLogger()
    # Check for existing file handlers to avoid duplicates
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in root_logger.handlers):
         logger.warning(f"Log file handler for {log_file} already exists. Skipping add.")
    else:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler) # Add handler to root logger
        logger.info(f"Logging detailed output to: {log_file}")


    # --- Training Arguments ---
    logger.info("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type="cosine", # Common scheduler
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm, # Gradient clipping

        logging_dir=os.path.join(model_output_dir, "logs"), # Logs subdirectory
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        load_best_model_at_end=True, # Load the best model based on eval loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Device and Performance settings
        # fp16=False, # Explicitly disable fp16/bf16 for MPS/CPU safety
        # bf16=False,
        gradient_checkpointing=True, # Already enabled on model, but set here too
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Match model setting
        # torch_compile=False, # Can cause issues on MPS/CPU
        dataloader_num_workers=0, # Safest for MPS/CPU
        dataloader_pin_memory=False, # Avoid pinning memory

        report_to="none", # Disable external reporting (wandb, tensorboard) unless configured
    )

    # --- Data Collator ---
    # Handles padding dynamically per batch to the longest sequence in the batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model, # Not strictly needed unless using model-specific padding logic
        label_pad_token_id=-100, # Ensure labels are padded with ignore index
        pad_to_multiple_of=8 # Optional: May improve performance on some hardware
    )

    # --- Callbacks ---
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    # --- Train ---
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("Training completed successfully.")

        # Log training metrics
        metrics = train_result.metrics
        # Calculate perplexity if loss is available
        try:
            perplexity = np.exp(metrics["train_loss"])
            metrics["train_perplexity"] = perplexity
        except KeyError:
            logger.warning("Could not calculate train perplexity: 'train_loss' not found in metrics.")
        except OverflowError:
            metrics["train_perplexity"] = float("inf")


        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() # Save final trainer state (optimizer, scheduler, etc.)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True) # Log traceback
        raise # Re-raise exception after logging

    # --- Save Final Model ---
    logger.info("Saving the best model...")
    # Trainer automatically saves the best checkpoint based on eval_loss
    # The final model state loaded corresponds to the best checkpoint due to load_best_model_at_end=True
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        logger.info(f"Best model checkpoint identified at: {best_model_path}")
        # You might want to save the final adapter separately for easier loading later
        final_adapter_dir = os.path.join(model_output_dir, "final_adapter")
        model.save_pretrained(final_adapter_dir) # Saves only the adapter weights
        tokenizer.save_pretrained(final_adapter_dir) # Save tokenizer with adapter
        logger.info(f"Final adapter saved to: {final_adapter_dir}")
        # Optionally copy the full best checkpoint if needed
        # final_model_dir = os.path.join(model_output_dir, "best_checkpoint_full")
        # if os.path.exists(best_model_path):
        #     shutil.copytree(best_model_path, final_model_dir, dirs_exist_ok=True)
        #     logger.info(f"Full best checkpoint copied to: {final_model_dir}")

    else:
        # Should not happen with load_best_model_at_end=True unless training stopped very early
        logger.warning("Best model checkpoint not found. Saving final model state as adapter.")
        final_adapter_dir = os.path.join(model_output_dir, "final_adapter")
        model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        logger.info(f"Final adapter saved to: {final_adapter_dir}")


    # --- Evaluate on Test Set (if available) ---
    if "test" in tokenized_datasets:
        logger.info("Evaluating on the test set using the best model...")
        try:
            test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
            # Calculate perplexity for test set
            try:
                test_perplexity = np.exp(test_results["eval_loss"])
                test_results["eval_perplexity"] = test_perplexity
            except KeyError:
                 logger.warning("Could not calculate test perplexity: 'eval_loss' not found.")
            except OverflowError:
                 test_results["eval_perplexity"] = float("inf")

            logger.info(f"Test Results: {test_results}")
            trainer.log_metrics("test", test_results)
            # Save test metrics in the main output directory
            with open(os.path.join(model_output_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=4)

        except Exception as e:
            logger.error(f"Error during final test evaluation: {e}", exc_info=True)
    else:
        logger.info("No test set provided for final evaluation.")

    # --- Cleanup ---
    logger.info("Cleaning up memory...")
    # Ensure model and trainer are deleted
    try:
        del model
    except NameError: pass
    try:
        del trainer
    except NameError: pass
    try:
        del data_collator
    except NameError: pass
    gc.collect() # Run garbage collection
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Fine-tuning process finished.")
    logger.info(f"Model adapter and logs saved in: {model_output_dir}")
    return model_output_dir



# %%
if __name__ == "__main__":
    # Create config instance
    config = Config()
    logger.info(f"Script configuration: {vars(config)}") # Log the config used

    # Load HF Token if not provided via config
    if config.hf_token is None:
        load_dotenv()
        config.hf_token = os.getenv('HF_TOKEN')
        if config.hf_token:
            logger.info("Loaded Hugging Face token from .env file.")

    # Setup device and seeds
    device = setup_environment(config)

    # Load and prepare dataset
    processed_datasets = load_and_prepare_dataset(config)

    # Load tokenizer (needed for preprocessing)
    logger.info(f"Loading tokenizer for {config.model_name}...")
    tokenizer_load_params = {}
    if config.hf_token: tokenizer_load_params['token'] = config.hf_token
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, **tokenizer_load_params)
    except Exception as e:
        logger.error(f"Fatal: Failed to load tokenizer '{config.model_name}'. Error: {e}")
        sys.exit(1)


    # Preprocess and tokenize data
    try:
        tokenized_datasets = preprocess_and_tokenize(processed_datasets, tokenizer, config.max_seq_length)
    except Exception as e:
        logger.error(f"Fatal: Failed during data preprocessing/tokenization. Error: {e}", exc_info=True)
        sys.exit(1)


    # Check if datasets are empty after processing (important!)
    if not tokenized_datasets or not tokenized_datasets.get("train"):
         logger.error("Fatal: Training dataset is empty after preprocessing. Check data loading and tokenization steps.")
         sys.exit(1)
    if not tokenized_datasets.get("validation"):
         logger.error("Fatal: Validation dataset is empty after preprocessing. Check data loading and tokenization steps.")
         sys.exit(1)


    # Run training
    try:
        trained_model_path = train(config, device, tokenized_datasets, tokenizer)
        logger.info(f"Training successful. Best model adapter saved in subdirectories within: {trained_model_path}")
    except Exception as e:
        logger.error("Training failed.", exc_info=True)
        sys.exit(1) # Exit with error status

# %%



