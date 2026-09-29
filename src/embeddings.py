import json
import os
import glob
import platform
import sys
import logging

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "bertin-project/bertin-base-gaussian-exp-512seqlen",
        data_set_path: str = "data/md/plain_text",
        output_model_dir: str = "./models/cima-bert-model",
        test_training_path: str = "data/splits/test.csv",
    ):
        self.model_name = model_name
        self.test_training_path = test_training_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.data_set_path = data_set_path
        self.output_model_dir = output_model_dir
        # Force CODIGO to string so leading-zero codes (e.g. 0119600) are
        # preserved instead of being silently coerced to int.
        self.test_data: pd.DataFrame = pd.read_csv(
            self.test_training_path, dtype={"CODIGO": str}
        )

    def load_data_set(self):
        # Get all files and filter them
        all_files = glob.glob(f"{self.data_set_path}/*")
        print(f"Total files found: {len(all_files)}")
        unique_codes: list[str] = (
            self.test_data["CODIGO"].astype(str).str.strip().unique().tolist()
        )
        files_to_exclude = [f"{code}.txt" for code in unique_codes]
        files_to_keep = [
            f for f in all_files if os.path.basename(f) not in files_to_exclude
        ]
        print(f"Total files to exclude from the test dataset: {len(files_to_exclude)}")
        print(f"Total files to keep for training: {len(files_to_keep)}")
        self.dataset = load_dataset("text", data_files={"train": files_to_keep})
        print("Type of dataset:", type(self.dataset))
        print("Type of first row:", type(self.dataset["train"][0]))
        print("First row content:", self.dataset["train"][0])

        def tokenize_function(examples):
            # The tokenizer will handle splitting into words and converting to IDs
            return self.tokenizer(
                examples["text"], truncation=True, max_length=128, padding=True
            )

        self.tokenized_dataset = self.dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

    def calculate_training_tokens(self) -> int:
        # Flatten the list of lists and sum all 1s (real tokens)
        all_attention_masks = self.tokenized_dataset["train"]["attention_mask"]
        total_tokens = np.sum([sum(mask) for mask in all_attention_masks])
        return total_tokens

    def train_model(self):
        # 4. Set up the Data Collator for MLM
        # This magic part automatically creates the [MASK] tokens for training!
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.18,  # Standard probability to mask tokens
        )

        # 5. Define Training Arguments
        self.training_args = TrainingArguments(
            output_dir=self.output_model_dir,
            learning_rate=2e-5,  # A common starting learning rate for fine-tuning
            num_train_epochs=3,  # Adjust based on your dataset size and desired training time
            per_device_train_batch_size=16,  # Lower if you run out of GPU memory
            save_strategy="no",  # Disable checkpoint saving during training
            prediction_loss_only=True,  # We only need the loss for training
            logging_strategy="no",  # Disable logging to avoid cluttering the output
        )

        # 6. Create the Trainer and Start Training!
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.tokenized_dataset["train"],
        )

        print("Starting training...")
        self.trainer.train()

        # 7. Save the final model and tokenizer
        self.trainer.save_model(self.output_model_dir)
        self.tokenizer.save_pretrained(self.output_model_dir)
        print(
            f"Training complete. Model and tokenizer saved to {self.output_model_dir}"
        )

    def save_model_information(self):
        artifacts_dir = self.output_model_dir
        os.makedirs(artifacts_dir, exist_ok=True)

        # 1) Save TrainingArguments and Trainer state/log history
        with open(os.path.join(artifacts_dir, "training_args.json"), "w") as f:
            f.write(self.training_args.to_json_string())

        # trainer_state.json (includes log_history) and rng states
        self.trainer.save_state()

        # Also write logs separately for convenience
        with open(os.path.join(artifacts_dir, "log_history.json"), "w") as f:
            json.dump(self.trainer.state.log_history, f, indent=2)

        # 2) Save optimizer and scheduler to allow exact resume without a checkpoint dir
        if getattr(self.trainer, "optimizer", None) is not None:
            torch.save(
                self.trainer.optimizer.state_dict(),
                os.path.join(artifacts_dir, "optimizer.pt"),
            )
        if getattr(self.trainer, "lr_scheduler", None) is not None:
            torch.save(
                self.trainer.lr_scheduler.state_dict(),
                os.path.join(artifacts_dir, "scheduler.pt"),
            )

        # 3) Save preprocessing/tokenization params used
        preproc_cfg = {
            "collator": type(self.data_collator).__name__,
            "mlm": getattr(self.data_collator, "mlm", None),
            "mlm_probability": getattr(self.data_collator, "mlm_probability", None),
            # Mirrors the tokenize_function used above
            "tokenize": {
                "max_length": 128,
                "truncation": True,
                "padding": True,
            },
        }
        with open(os.path.join(artifacts_dir, "preprocessing_config.json"), "w") as f:
            json.dump(preproc_cfg, f, indent=2)

        # 4) Save environment information for reproducibility
        env_info = {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "model_checkpoint": self.model_name,
            "seed": self.training_args.seed,
        }
        with open(os.path.join(artifacts_dir, "environment.json"), "w") as f:
            json.dump(env_info, f, indent=2)

        print(f"Extra artifacts saved to {artifacts_dir}")
