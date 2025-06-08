import sys
sys.path.append('/remote-home/jiayuanrao/haokai/UniSoccer')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    LogitsProcessorList
)
from transformers.generation.logits_process import LogitsProcessor
from typing import List
import pickle as pkl
from dataset import QwenVLCommentaryDataset
from utils.score_helper import calculate_metrics_of_set
from tqdm import tqdm
import logging
import json
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RestrictTokenGenerationLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_id_list: List[int]):
        super().__init__()
        self.allowed_token_id_list = allowed_token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float('inf'))
        for allowed_id in self.allowed_token_id_list:
            mask[:, allowed_id] = scores[:, allowed_id]
        return mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for Qwen-VL model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path to the saved model")
    parser.add_argument("--adapter_path", type=str,
                        default=None, help="Path to the adapter model")
    parser.add_argument("--val_json_files", type=str, default="../train_data/json/MatchTime/classification_valid.json",
                        help="Path to validation JSON file(s), comma-separated")
    parser.add_argument("--video_base_dirs", type=str, default="../train_data/video_clips/",
                        help="Base directory for video files")
    parser.add_argument("--match_info_base_dir", type=str, default="../../Dataset/MatchTime/MatchTime/valid/",
                        help="Base directory for match information")
    parser.add_argument("--inference_name", type=str, default="qwen",
                        help="Name for the inference run")
    parser.add_argument("--restricted_token_list_path", type=str, default="../words_world/qwen/unanonymized_matchtime.pkl",
                        help="Path to pickle file containing allowed tokens")
    parser.add_argument("--output_dir", type=str, default="./inference_results/",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=96,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and processor
    logger.info(f"Loading model from {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    if args.adapter_path:
        model.load_adapter(args.adapter_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True
    )

    # Load restricted token list
    logger.info(
        f"Loading restricted token list from {args.restricted_token_list_path}")
    with open(args.restricted_token_list_path, 'rb') as f:
        token_ids_list = pkl.load(f)

    # Setup logits processor
    logits_processor = RestrictTokenGenerationLogitsProcessor(token_ids_list)
    logits_processors = LogitsProcessorList([logits_processor])

    # Load validation dataset
    val_dataset = QwenVLCommentaryDataset(
        json_file=args.val_json_files.split(','),
        video_base_dir=args.video_base_dirs.split(','),
        match_info_base_dir=args.match_info_base_dir,
        processor_name=args.model_path,
        train=False
    )

    # Setup data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    # Inference
    model.eval()
    predictions = []
    references = []
    extra_infos = []

    logger.info("Starting inference...")
    with torch.no_grad():
        pbar = tqdm(val_loader)
        for index, (batch, caption_text, batch_extra_infos) in enumerate(pbar):
            batch = batch.to('cuda')
            generated_ids = model.generate(
                **batch,
                logits_processor=logits_processors,
                renormalize_logits=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                num_beams=5,
                min_length=5,
                repetition_penalty=1.0,
                length_penalty=1.0
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
            ]

            pred_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            ref_texts = caption_text

            predictions.extend(pred_texts)
            references.extend(ref_texts)
            extra_infos.extend(batch_extra_infos)

            paired_results = [
                {
                    "prediction": pred.strip(), 
                    "reference": ref.strip(),
                    "video_path": info["original_video_path"],
                    "comments_text_anonymized": info["comments_text_anonymized"]
                }
                for pred, ref, info in zip(predictions, references, extra_infos)
            ]

            # 实时计算metrics并打印
            ref_dict = {i: [ref.strip()] for i, ref in enumerate(references)}
            pred_dict = {i: [pred.strip()]
                         for i, pred in enumerate(predictions)}
            metrics = calculate_metrics_of_set(ref_dict, pred_dict)
            # 只显示主要指标
            pbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.items()})

            # 保存为 json 文件
            output_pair_path = os.path.join(
                args.output_dir, f'{args.inference_name}.json')
            if index % 100 == 0:
                with open(output_pair_path, 'w') as f:
                    results = {
                        'metrics': metrics,
                        'pairs': paired_results
                    }
                    json.dump(results, f, indent=2)

    # Format for metric calculation
    ref_dict = {i: [ref.strip()] for i, ref in enumerate(references)}
    pred_dict = {i: [pred.strip()] for i, pred in enumerate(predictions)}

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics_of_set(ref_dict, pred_dict)

    # # 最终保存结果
    # paired_results = [
    #     {
    #         "prediction": pred.strip(), 
    #         "reference": ref.strip(),
    #         "video_path": info["original_video_path"],
    #         "comments_text_anonymized": info["comments_text_anonymized"]
    #     }
    #     for pred, ref, info in zip(predictions, references, extra_infos)
    # ]

    # # Save results
    # results = {
    #     'metrics': metrics,
    #     'pairs': paired_results
    # }

    # output_path = os.path.join(
    #     args.output_dir, f'{args.inference_name}.json')
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=2)

    # logger.info(f"Results saved to {output_path}")
    # logger.info("Metrics:")
    # for metric_name, score in metrics.items():
    #     logger.info(f"{metric_name}: {score:.2f}")


if __name__ == "__main__":
    main()
