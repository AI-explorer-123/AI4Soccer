import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import sys
import json
sys.path.append('/remote-home/jiayuanrao/haokai/UniSoccer')
from dataset.MatchVision_commentary_new_benchmark_from_npy import MatchVisionCommentary_new_benchmark_from_npy_Dataset
from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
from torch.nn import DataParallel
import numpy as np
import random
import os
from pycocoevalcap.cider.cider import Cider
import wandb
from utils.score_helper import calculate_metrics_of_set
from optimizer.optimizer_utls import optimizer_commentary_new_benchmark
import csv


def inference(args):
    dataset_type = MatchVisionCommentary_new_benchmark_from_npy_Dataset
    commentary_model_type = matchvoice_model_all_blocks
    device_ids = args.device_ids
    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    valid_json = []
    valid_video_base_dir = []
    if args.valid_matchtime:
        valid_json.append(args.matchtime_json)
        valid_video_base_dir.append(args.matchtime_video_base)
    if args.valid_soccerreplay:
        valid_json.append(args.soccerreplay1988_json)
        valid_video_base_dir.append(args.soccerreplay1988_video_base)

    val_dataset = dataset_type(json_file=valid_json,
                               video_base_dir=valid_video_base_dir,
                               tokenizer_name=args.tokenizer_name)

    torch.cuda.init()
    torch.manual_seed(42)

    val_data_loader = DataLoader(val_dataset, batch_size=args.valid_batch_size, num_workers=args.valid_num_workers,
                                 drop_last=False, shuffle=True, pin_memory=True, collate_fn=val_dataset.collater)

    predictions = []
    references = []
    extra_infos = []

    print("===== Video features data loaded! =====")
    model = commentary_model_type(num_features=args.num_features, need_temporal=args.need_temporal, open_visual_encoder=args.open_visual_encoder, open_llm_decoder=args.open_llm_decoder, tokenizer_ckpt=args.tokenizer_name,
                                  llm_ckpt=args.tokenizer_name, restricted_token_list_path='../words_world/llama3/unanonymized_matchtime_llama3.pkl', visual_encoder_checkpoint=args.visual_encoder_checkpoint)

    model = model.to(devices[0])
    model = DataParallel(model, device_ids=device_ids)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    checkpoint = {'module.'+k: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("===== Model loaded! =====")

    val_pbar = tqdm(val_data_loader)
    with torch.no_grad():
        model = model.module if isinstance(
            model, torch.nn.DataParallel) else model
        model = model.to(devices[0])
        torch.cuda.empty_cache()
        for samples in val_pbar:
            samples["frames"] = samples["frames"].to(devices[0])
            pred_texts, ref_texts, video_paths = model(samples, True)
            anonymized_texts = samples["anonymous_caption_text"]

            predictions.extend(pred_texts)
            references.extend(ref_texts)

            # 收集额外信息
            for pred, ref, anon, path in zip(pred_texts, ref_texts, anonymized_texts, video_paths):
                extra_infos.append({
                    "video_path": path,
                    "comments_text_anonymized": anon
                })

            # 实时计算并显示指标
            if len(predictions) > 0:
                ref_dict = {i: [ref.strip()]
                            for i, ref in enumerate(references)}
                pred_dict = {i: [pred.strip()]
                             for i, pred in enumerate(predictions)}
                metrics = calculate_metrics_of_set(ref_dict, pred_dict)
                val_pbar.set_postfix(
                    {k: f"{v:.2f}" for k, v in metrics.items()})

    # 计算最终指标
    ref_dict = {i: [ref.strip()] for i, ref in enumerate(references)}
    pred_dict = {i: [pred.strip()] for i, pred in enumerate(predictions)}
    final_metrics = calculate_metrics_of_set(ref_dict, pred_dict)

    # 准备结果数据
    paired_results = [
        {
            "prediction": pred.strip(),
            "reference": ref.strip(),
            "video_path": info["video_path"],
            "comments_text_anonymized": info["comments_text_anonymized"]
        }
        for pred, ref, info in zip(predictions, references, extra_infos)
    ]

    # 保存结果
    results = {
        'metrics': final_metrics,
        'pairs': paired_results
    }

    # 保存为 JSON 文件
    output_json_path = f'./{args.exp_name}.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\nFinal Metrics:")
    for metric_name, score in final_metrics.items():
        print(f"{metric_name}: {score:.2f}")
    print(f"\nResults saved to {output_json_path}")


if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(
        description="Train a model with FRANZ dataset.")
    parser.add_argument("--need_temporal", type=str, default="yes")
    parser.add_argument("--tokenizer_name", type=str,
                        default="/home/jiayuanrao/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct")
    parser.add_argument("--valid_batch_size", type=int, default=40)
    parser.add_argument("--valid_num_workers", type=int, default=20)

    parser.add_argument("--valid_matchtime", type=bool, default=True)
    parser.add_argument("--valid_soccerreplay", type=bool, default=False)

    parser.add_argument("--num_features", type=int, default=768)
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--open_visual_encoder", type=bool, default=False)
    parser.add_argument("--open_llm_decoder", type=bool, default=True)

    parser.add_argument("--ckpt_path", type=str,
                        default="../results/unanonymized_unisoccer_classification_pretrained/checkpoints/model_save_best_val_CIDEr.pth")
    parser.add_argument("--visual_encoder_checkpoint", type=str,
                        default="../pretrained_weights/pretrained_classification.pth")

    parser.add_argument("--matchtime_json", type=str,
                        default="../train_data/json/MatchTime/classification_valid.json")
    parser.add_argument("--matchtime_video_base", type=str,
                        default="../train_data/video_clips/")
    parser.add_argument("--soccerreplay1988_json", type=str,
                        default="train_data/json/SoccerReplay-1988/classification_test.json")
    parser.add_argument("--soccerreplay1988_video_base", type=str,
                        default="FOLDER_OF_VIDEO_CLIPS_OF_SOCCERREPLAY_1988")
    
    parser.add_argument("--exp_name", type=str, default="pretrained_classification")

    args = parser.parse_args()
    inference(args)
