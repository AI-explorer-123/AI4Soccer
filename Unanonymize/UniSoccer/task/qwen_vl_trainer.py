import sys
sys.path.append('/remote-home/jiayuanrao/haokai/UniSoccer')
from traitlets import default
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
from datetime import datetime
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,  # Corrected class name
    set_seed,
    BitsAndBytesConfig
)
from transformers.trainer_utils import EvalPrediction
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from dataset.QwenVL_commentary_dataset import QwenVLCommentaryDataset

from utils.score_helper import calculate_metrics_of_set


# Setup basic logging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def compute_metrics(eval_pred: EvalPrediction):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                                   trust_remote_code=True,
                                                   use_fast=True)
    # 解码预测和标签
    predictions, label_ids = eval_pred
    # 处理为字符串
    pred_texts = processor.tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True)

    # 组装为字典形式
    # 这里假设每个样本一个句子
    references = {i: [label_texts[i].strip()]
                  for i in range(len(label_texts))}
    hypotheses = {i: [pred_texts[i].strip()]
                  for i in range(len(pred_texts))}

    # 计算指标
    metrics = calculate_metrics_of_set(references, hypotheses)
    # SFTTrainer会自动加上eval_前缀
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL for football commentary generation")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--processor_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Processor name or path (defaults to model_name_or_path if not specified)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to use LoRA")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float,
                        default=0.05, help="Dropout probability for LoRA")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj",  # Common modules
                        help="Comma-separated list of module names to apply LoRA to (e.g., 'q_proj,v_proj'). Use 'all' for all linear layers.")

    parser.add_argument("--use_4bit", action="store_true",
                        help="Whether to use 4-bit quantization (QLoRA)")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        help="Compute dtype for 4-bit BNB (bfloat16, float16, float32)")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                        help="Quantization type for 4-bit BNB (fp4 or nf4)")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true",
                        help="Whether to use double quantization for 4-bit BNB")

    # Data arguments
    parser.add_argument("--train_json_files", type=str, default='../train_data/json/MatchTime/classification_train.json',
                        help="Path(s) to training data JSON file(s), comma-separated")
    parser.add_argument("--val_json_files", type=str, default='../train_data/json/MatchTime/classification_valid.json',
                        help="Path(s) to validation data JSON file(s), comma-separated")
    parser.add_argument("--video_base_dirs", type=str, default='../train_data/video_clips/',
                        help="Base directory/ies for video files, comma-separated for multiple train_json_files")
    parser.add_argument("--match_info_base_dir", type=str, default='../../Dataset/MatchTime/MatchTime/train/',
                        help="Base directory for match information JSON files (e.g., .../MatchTime/dataset/MatchTime/train)")
    parser.add_argument("--max_token_length", type=int, default=1024,
                        help="Maximum token length for text")  # Increased default

    # Training arguments from SFTConfig
    parser.add_argument("--output_dir", type=str, default="../results/",
                        help="Output directory for model and checkpoints")
    parser.add_argument("--num_train_epochs", type=float,
                        default=3.0, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per GPU for training")  # Adjusted for VLM
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size per GPU for evaluation")  # Adjusted for VLM
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate for AdamW")  # Adjusted for VLM
    parser.add_argument("--warmup_ratio", type=float,
                        default=0.03, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str,
                        default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="Directory for storing logs")  
    parser.add_argument("--logging_steps", type=int,
                        default=10, help="Log every X updates steps")
    parser.add_argument("--save_strategy", type=str,
                        default="epoch", help="Save strategy (steps or epoch)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps (if save_strategy='steps')")
    parser.add_argument("--save_total_limit", type=int,
                        default=2, help="Limit the total amount of checkpoints")
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                        help="Evaluation strategy (steps or epoch or no)")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Run evaluation every X steps (if evaluation_strategy='steps')")

    parser.add_argument("--bf16", action="store_true",
                        help="Whether to use bf16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use fp16 precision (use bf16 if on Ampere or newer)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str,
                        default="adamw_torch", help="Optimizer to use")

    # Evaluation arguments
    parser.add_argument("--max_eval_new_tokens", type=int, default=128,
                        help="Max new tokens for generation during evaluation")
    parser.add_argument("--eval_temperature", type=float, default=0.7,
                        help="Temperature for generation during evaluation")
    parser.add_argument("--eval_top_p", type=float, default=0.9,
                        help="Top-p for generation during evaluation")
    parser.add_argument("--eval_do_sample", action="store_true",
                        help="Whether to use sampling during evaluation generation")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name for sub-folder in output_dir")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Report results to (tensorboard, wandb, none)")
    parser.add_argument("--max_grad_norm", type=float,
                        default=1.0, help="Max gradient norm for clipping")

    args = parser.parse_args()
    if args.processor_name is None:
        args.processor_name = args.model_name_or_path
    if args.use_4bit and not args.use_lora:
        logger.warning(
            "4-bit quantization (use_4bit) is typically used with LoRA (QLoRA). Training full 4-bit model is experimental.")
    if args.bf16 and args.fp16:
        raise ValueError("Cannot use both bf16 and fp16. Choose one.")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_name and args.exp_name != "test":
        output_dir = os.path.join(args.output_dir, f"{args.exp_name}_{timestamp}")
    else:
        output_dir = os.path.join(args.output_dir, "test")
    
    if args.exp_name == "test":
        args.report_to = None

    # Update args to reflect the full output path for TrainingArguments
    args.output_dir = output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load processor (includes tokenizer for Qwen-VL)
    logger.info(f"Loading processor from {args.processor_name}")
    processor = AutoProcessor.from_pretrained(args.processor_name,
                                              trust_remote_code=True,
                                              use_fast=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Common practice
    # processor.tokenizer.padding_side = "left"  # Common practice

    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model_dtype = torch.bfloat16 if args.bf16 else (
        torch.float16 if args.fp16 else torch.float32)

    quantization_config = None
    if args.use_4bit:
        logger.info("Using 4-bit quantization (QLoRA setup)")
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype if not args.use_4bit else None,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        if args.use_4bit:  # If QLoRA
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing)

        target_modules = [tm.strip()
                          for tm in args.lora_target_modules.split(",")]
        if "all" in target_modules:  # A helper to target all linear layers
            target_modules = [name for name, module in model.named_modules(
            ) if isinstance(module, torch.nn.Linear)]
            logger.info(
                f"LoRA target modules 'all' expanded to: {target_modules[:5]}... and more")

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load datasets
    logger.info("Loading datasets...")
    train_json_list = args.train_json_files.split(',')
    train_video_dirs = args.video_base_dirs.split(',') if args.video_base_dirs else [
        None] * len(train_json_list)

    train_dataset = QwenVLCommentaryDataset(
        json_file=train_json_list,
        video_base_dir=train_video_dirs,
        match_info_base_dir=args.match_info_base_dir,
        processor_name=args.processor_name,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")

    val_dataset = None
    if args.val_json_files:
        val_json_list = args.val_json_files.split(',')
        # Assume val shares the same base dir structure or provide specific ones
        val_video_dirs = train_video_dirs if len(val_json_list) == len(train_json_list) else (args.video_base_dirs.split(
            ',') if args.video_base_dirs else [None] * len(val_json_list))  # Simplified logic, adjust if needed

        val_dataset = QwenVLCommentaryDataset(
            json_file=val_json_list,
            video_base_dir=val_video_dirs,
            match_info_base_dir=args.match_info_base_dir,
            processor_name=args.processor_name
        )
        logger.info(f"Validation dataset size: {len(val_dataset)}")

    # TrainingArguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # metric_for_best_model="eval_CIDER",
        # greater_is_better=True,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        report_to=args.report_to,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=16
    )
    training_args.remove_unused_columns = False

    if args.report_to == "wandb":
        import wandb
        wandb.init(
            project="qwen-fine-tuning",
            name=f"{args.exp_name}_{timestamp}",  # change this
            config=training_args
        )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
        peft_config=peft_config if args.use_lora else None,
        processing_class=processor.tokenizer,
        # compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    logger.info(f"Script finished. Outputs are in {args.output_dir}")


if __name__ == "__main__":
    main()
