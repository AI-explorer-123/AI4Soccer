import sys
sys.path.append('/remote-home/jiayuanrao/haokai/UniSoccer')
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import matplotlib.pyplot as plt
import json
from dataset.MatchVision_commentary_new_benchmark_from_npy import MatchVisionCommentary_new_benchmark_from_npy_Dataset
from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim import AdamW
import torch
from torch.nn import DataParallel
import numpy as np
import random
from pycocoevalcap.cider.cider import Cider
import wandb
from utils.score_helper import calculate_metrics_of_set
from optimizer.optimizer_utls import optimizer_commentary_new_benchmark


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def eval_cider(predicted_captions, gt_captions):
    cider_evaluator = Cider()
    predicted_captions_dict = {i: [caption]
                               for i, caption in enumerate(predicted_captions)}
    gt_captions_dict = {i: [caption] for i, caption in enumerate(gt_captions)}
    _, cider_scores = cider_evaluator.compute_score(
        predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()


def plot_curves(metrics, save_path):
    plt.figure(figsize=(12, 5))
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.subplot(1, len(metrics), i + 1)
        plt.plot(values, label=metric_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(metric_name)
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(args):
    if args.exp_name:
        results_dir = os.path.join(
            "/remote-home/jiayuanrao/haokai/UniSoccer/results", args.exp_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{timestamp}"
        results_dir = os.path.join("results", exp_name)

    os.makedirs(results_dir, exist_ok=True)
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 初始化记录列表
    train_losses = []
    metrics_history = {"CIDEr": [], "BLEU-1": [],
                       "BLEU-4": [], "METEOR": [], "ROUGE-L": []}

    dataset_type = MatchVisionCommentary_new_benchmark_from_npy_Dataset
    commentary_model_type = matchvoice_model_all_blocks
    device_ids = args.device_ids
    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    if args.use_wandb:
        wandb.init(project=args.wandb_name, entity="jy_rao")
        wandb.config = {"remark": args.wandb_name}

    train_json = []
    valid_json = []
    train_video_base_dir = []
    valid_video_base_dir = []
    if args.train_matchtime:
        train_json.append(args.matchtime_train_json)
        train_video_base_dir.append(args.matchtime_video_folder)
        valid_json.append(args.matchtime_valid_json)
        valid_video_base_dir.append(args.matchtime_video_folder)
    if args.train_soccerreplay:
        train_json.append(args.soccerreplay1988_train_json)
        train_video_base_dir.append(args.soccerreplay1988_video_folder)
        valid_json.append(args.soccerreplay1988_valid_json)
        valid_video_base_dir.append(args.soccerreplay1988_video_folder)

    train_dataset = dataset_type(json_file=train_json, 
                                 video_base_dir=train_video_base_dir,
                                 tokenizer_name=args.tokenizer_name)
    val_dataset = dataset_type(json_file=valid_json,
                               video_base_dir=valid_video_base_dir,
                               tokenizer_name=args.tokenizer_name)

    torch.cuda.init()
    torch.manual_seed(42)

    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers,
                                   drop_last=True, shuffle=True, pin_memory=True, collate_fn=train_dataset.collater)
    val_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers,
                                 drop_last=False, shuffle=True, pin_memory=True, collate_fn=val_dataset.collater)

    print("===== Video features data loaded! =====")
    model = commentary_model_type(visual_encoder_checkpoint=args.visual_encoder_checkpoint,
                                  llm_ckpt=args.tokenizer_name, 
                                  tokenizer_ckpt=args.tokenizer_name,
                                  restricted_token_list_path=args.restricted_token_list_path,
                                  num_features=args.num_features, 
                                  need_temporal=args.need_temporal,
                                  open_visual_encoder=args.open_visual_encoder, 
                                  open_llm_decoder=args.open_llm_decoder,
                                  llm_lora_rank=args.llm_lora_rank)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params / 1e6:.2f}M")
    
    model = model.to(devices[0])
    model = DataParallel(model, device_ids=device_ids)

    if args.continue_train:
        checkpoint = torch.load(args.load_ckpt, map_location="cpu")[
            "state_dict"]
        checkpoint = {'module.'+k: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    optimizer = optimizer_commentary_new_benchmark(
        model, open_visual=args.open_visual_encoder, open_text=args.open_llm_decoder)
    print("===== Model and Checkpoints loaded! =====")

    max_val_CIDEr = max(float(0), args.pre_max_CIDEr)
    for epoch in range(args.pre_epoch, args.num_epoch):
        torch.cuda.empty_cache()
        model.train()
        train_loss_accum = 0.0
        train_pbar = tqdm(train_data_loader,
                          desc=f'Epoch {epoch+1}/{args.num_epoch} Training')
        for samples in train_pbar:
            samples["frames"] = samples["frames"].to(devices[0])
            samples["input_ids"] = samples["input_ids"].to(devices[0])
            samples["attention_mask"] = samples["attention_mask"].to(
                devices[0])
            samples["labels"] = samples["labels"].to(devices[0])
            optimizer.zero_grad()
            loss = model(samples)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_train_loss = train_loss_accum / len(train_data_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_CIDEr = 0.0
        reference_list = []
        hypothesis_list = []

        val_pbar = tqdm(val_data_loader,
                        desc=f'Epoch {epoch+1}/{args.num_epoch} Validation')
        with torch.no_grad():
            model = model.module if isinstance(
                model, torch.nn.DataParallel) else model
            model = model.to(devices[0])
            torch.cuda.empty_cache()
            for samples in val_pbar:
                samples["frames"] = samples["frames"].to(devices[0])
                temp_res_text, anonymized, video_path = model(samples, True)
                cur_CIDEr_score = eval_cider(temp_res_text, anonymized)
                val_CIDEr += sum(cur_CIDEr_score) / len(cur_CIDEr_score)
                val_pbar.set_postfix(
                    {"Scores": f"|C:{sum(cur_CIDEr_score)/len(cur_CIDEr_score):.4f}"})
                reference_list.extend(anonymized)
                hypothesis_list.extend(temp_res_text)

        avg_val_CIDEr = val_CIDEr / len(val_data_loader)
        reference_dict = {i: [s] for i, s in enumerate(reference_list)}
        hypothesis_dict = {i: [s] for i, s in enumerate(hypothesis_list)}
        scores_this_epoch = calculate_metrics_of_set(
            reference_dict, hypothesis_dict)

        # 记录 metrics
        metrics_history["CIDEr"].append(scores_this_epoch["CIDER"])
        metrics_history["BLEU-1"].append(scores_this_epoch["BLEU-1"])
        metrics_history["BLEU-4"].append(scores_this_epoch["BLEU-4"])
        metrics_history["METEOR"].append(scores_this_epoch["METEOR"])
        metrics_history["ROUGE-L"].append(scores_this_epoch["ROUGE-L"])

        print(
            f"Epoch {epoch+1} Summary: Average Training Loss: {avg_train_loss:.3f}, Average Validation scores: B1:{scores_this_epoch['BLEU-1']:.3f}|B4:{scores_this_epoch['BLEU-4']:.3f}|M:{scores_this_epoch['METEOR']:.3f}|R:{scores_this_epoch['ROUGE-L']:.3f}|C:{scores_this_epoch['CIDER']:.3f}")

        # 保存模型
        if args.model_save and epoch % args.model_save_every == 0:
            file_path = f"{ckpt_dir}/model_save_{epoch+1}.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            }
            torch.save(checkpoint, file_path)
            print(f'Checkpoint saved at epoch {epoch+1}')

        if avg_val_CIDEr > max_val_CIDEr:
            max_val_CIDEr = avg_val_CIDEr
            file_path = f"{ckpt_dir}/model_save_best_val_CIDEr.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            }
            torch.save(checkpoint, file_path)
            print(f'Best model saved at epoch {epoch+1}')

        # 绘制并保存曲线
        plot_curves(metrics_history, os.path.join(
            results_dir, 'training_curves.png'))

        # 保存日志
        log_data = {
            'epoch': list(range(len(train_losses))),
            'train_loss': train_losses,
            **metrics_history
        }
        with open(os.path.join(results_dir, 'training_log.json'), 'w') as f:
            json.dump(log_data, f, indent=4)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(
        description="Train a model with soccer dataset.")
    parser.add_argument("--open_visual_encoder", type=str2bool, default=False)
    parser.add_argument("--open_llm_decoder", type=str2bool, default=True)
    parser.add_argument("--need_temporal", type=str, default="yes")
    parser.add_argument("--tokenizer_name", type=str,
                        default="/home/jiayuanrao/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct")
    parser.add_argument("--restricted_token_list_path", type=str,
                        default="../words_world/llama3/unanonymized_matchtime_llama3.pkl")
    parser.add_argument("--visual_encoder_checkpoint", type=str,
                        default="../pretrained_weights/pretrained_both.pth")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_num_workers", type=int, default=32)
    parser.add_argument("--train_matchtime", type=str2bool, default=True)
    parser.add_argument("--train_soccerreplay", type=str2bool, default=False)
    parser.add_argument("--valid_matchtime", type=str2bool, default=True)
    parser.add_argument("--valid_soccerreplay", type=str2bool, default=False)

    parser.add_argument("--matchtime_train_json", type=str,
                        default="../train_data/json/MatchTime/classification_train.json")
    parser.add_argument("--matchtime_valid_json", type=str,
                        default="../train_data/json/MatchTime/classification_valid.json")
    parser.add_argument("--matchtime_video_folder", type=str,
                        default="../train_data/video_clips/")
    parser.add_argument("--soccerreplay1988_train_json", type=str,
                        default="../train_data/json/SoccerReplay-1988/classification_train.json")
    parser.add_argument("--soccerreplay1988_valid_json", type=str,
                        default="../train_data/json/SoccerReplay-1988/classification_valid.json")
    parser.add_argument("--soccerreplay1988_video_folder", type=str,
                        default="FOLDER_OF_VIDEO_CLIPS_OF_SOCCERREPLAY_1988")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--num_features", type=int, default=768)
    parser.add_argument("--llm_lora_rank", type=int, default=16)
    parser.add_argument("--model_save", type=bool, default=True)
    parser.add_argument("--model_save_every", type=int, default=1)
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_name", type=str, default="YOUR_PROJECT_NAME")
    parser.add_argument("--continue_train", type=str2bool, default=False)
    parser.add_argument("--pre_max_CIDEr", type=float, default=0.0)
    parser.add_argument("--pre_epoch", type=int, default=0)
    parser.add_argument("--load_ckpt", type=str,
                        default="TO_CONTINUE_TRAINING_FROM_THIS_CHECKPOINT")
    parser.add_argument("--exp_name", type=str,
                        default="unanonymized_unisoccer_both_pretrained_2", help="Experiment name (optional)")

    args = parser.parse_args()
    train(args)
