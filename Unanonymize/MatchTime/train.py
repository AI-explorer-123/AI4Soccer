import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from matchvoice_dataset import MatchVoice_Dataset
from models.matchvoice_model import matchvoice_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
import torch
import numpy as np
import random
from pycocoevalcap.cider.cider import Cider
import matplotlib.pyplot as plt
from datetime import datetime
from utils.score_helper import calculate_metrics_of_set
import json

# Use CIDEr score to do validation


def eval_cider(predicted_captions, gt_captions):
    cider_evaluator = Cider()
    predicted_captions_dict = dict()
    gt_captions_dict = dict()
    for i, caption in enumerate(predicted_captions):
        predicted_captions_dict[i] = [caption]
    for i, caption in enumerate(gt_captions):
        gt_captions_dict[i] = [caption]
    _, cider_scores = cider_evaluator.compute_score(
        predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()


def plot_curves(train_losses, val_ciders, save_path):
    plt.figure(figsize=(12, 5))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制验证CIDEr分数
    plt.subplot(1, 2, 2)
    plt.plot(val_ciders, 'r-', label='Validation CIDEr')
    plt.xlabel('Epoch')
    plt.ylabel('CIDEr Score')
    plt.title('Validation CIDEr')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(args):
    if args.exp_name:
        results_dir = os.path.join("results", args.exp_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{timestamp}"
        results_dir = os.path.join("results", exp_name)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 初始化记录列表
    train_losses = []
    val_ciders = []

    train_dataset = MatchVoice_Dataset(feature_root=args.feature_root, ann_root=args.train_ann_root,
                                       window=args.window, fps=args.fps, tokenizer_name=args.tokenizer_name, timestamp_key=args.train_timestamp_key)
    val_dataset = MatchVoice_Dataset(feature_root=args.feature_root, ann_root=args.val_ann_root,
                                     window=args.window, fps=args.fps, tokenizer_name=args.tokenizer_name, timestamp_key=args.val_timestamp_key)

    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers,
                                   drop_last=False, shuffle=True, pin_memory=True, collate_fn=train_dataset.collater)
    val_data_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.val_num_workers,
                                 drop_last=True, shuffle=False, pin_memory=True, collate_fn=train_dataset.collater)
    print("===== Video features data loaded! =====")
    model = matchvoice_model(llm_ckpt=args.tokenizer_name, tokenizer_ckpt=args.tokenizer_name, restricted_token_list_path=args.restricted_token_list_path, window=args.window, num_query_tokens=args.num_query_tokens,
                             num_video_query_token=args.num_video_query_token, num_features=args.num_features, device=args.device).to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print(f"===== Model loaded! =====\nTotal trainable parameters: {num_params:.2f}M")

    if args.continue_train:
        model.load_state_dict(torch.load(args.load_ckpt))
    optimizer = AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.model_output_dir, exist_ok=True)
    print("===== Model and Checkpoints loaded! =====")

    # max_val_CIDEr = max(float(0), args.pre_max_CIDEr)
    # for epoch in range(args.pre_epoch, args.num_epoch):
    #     model.train()
    #     train_loss_accum = 0.0
    #     train_pbar = tqdm(train_data_loader,
    #                       desc=f'Epoch {epoch+1}/{args.num_epoch} Training')
    #     for samples in train_pbar:
    #         optimizer.zero_grad()
    #         loss = model(samples)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss_accum += loss.item()
    #         train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    #         avg_train_loss = train_loss_accum / len(train_data_loader)

    #     model.eval()
    #     val_CIDEr = 0.0
    #     val_pbar = tqdm(val_data_loader,
    #                     desc=f'Epoch {epoch+1}/{args.num_epoch} Validation')
    #     with torch.no_grad():
    #         for samples in val_pbar:
    #             temp_res_text, ground_truth_comment, anonymous_comment, file_path = model(
    #                 samples, True)
    #             cur_CIDEr_score = eval_cider(
    #                 temp_res_text, ground_truth_comment)
    #             val_CIDEr += sum(cur_CIDEr_score) / len(cur_CIDEr_score)
    #             val_pbar.set_postfix(
    #                 {"Scores": f"|C:{sum(cur_CIDEr_score)/len(cur_CIDEr_score):.4f}"})

    #     avg_val_CIDEr = val_CIDEr / len(val_data_loader)
    #     print(f"Epoch {epoch+1} Summary: Average Training Loss: {avg_train_loss:.3f}, Average Validation scores: C:{avg_val_CIDEr*100:.3f}")

    #     # 记录数据
    #     train_losses.append(avg_train_loss)
    #     val_ciders.append(avg_val_CIDEr)

    #     # 绘制并保存曲线
    #     plot_curves(train_losses, val_ciders,
    #                 os.path.join(results_dir, 'training_curves.png'))

    #     # 保存日志数据
    #     log_data = {
    #         'epoch': list(range(len(train_losses))),
    #         'train_loss': train_losses,
    #         'val_cider': val_ciders
    #     }
    #     with open(os.path.join(results_dir, 'training_log.txt'), 'w') as f:
    #         for i in range(len(train_losses)):
    #             f.write(
    #                 f"Epoch {log_data['epoch'][i]+1}: Loss={log_data['train_loss'][i]:.4f}, CIDEr={log_data['val_cider'][i]:.4f}\n")

    #     # 修改检查点保存路径
    #     if epoch % 5 == 0:
    #         file_path = os.path.join(ckpt_dir, f"model_save_{epoch+1}.pth")
    #         save_matchvoice_model(model, file_path)

    #     if avg_val_CIDEr > max_val_CIDEr:
    #         max_val_CIDEr = avg_val_CIDEr
    #         file_path = os.path.join(ckpt_dir, "model_save_best_val_CIDEr.pth")
    #         save_matchvoice_model(model, file_path)

    # 加载最佳模型进行最终验证
    print("Loading best model for final validation...")
    best_model = matchvoice_model(llm_ckpt=args.tokenizer_name, 
                                  tokenizer_ckpt=args.tokenizer_name, 
                                  restricted_token_list_path=args.restricted_token_list_path, 
                                  window=args.window, 
                                  num_query_tokens=args.num_query_tokens,
                                  num_video_query_token=args.num_video_query_token, 
                                  num_features=args.num_features, 
                                  device=args.device).to(args.device)
    # best_model_path = os.path.join(ckpt_dir, "model_save_best_val_CIDEr.pth")
    best_model_path = './results/unanonymized_matchtime_1fps_aligned_time/checkpoints/model_save_best_val_CIDEr.pth'
    # best_model_path = os.path.join(ckpt_dir, "model_save_best_val_CIDEr.pth")
    other_parts_state_dict = torch.load(best_model_path)
    new_model_state_dict = best_model.state_dict()
    for key, value in other_parts_state_dict.items():
        if key in new_model_state_dict:
            new_model_state_dict[key] = value
    best_model.load_state_dict(new_model_state_dict)
    
    print("Running final validation...")
    final_metrics = final_validation(best_model, val_data_loader, results_dir)
    
    print("\nFinal Validation Metrics:")
    for metric_name, score in final_metrics.items():
        print(f"{metric_name}: {score:.2f}")


def save_matchvoice_model(model, file_path):
    state_dict = model.cpu().state_dict()
    state_dict_without_llama = {}
    # 遍历原始模型的 state_dict，并排除 llama_model 相关的权重
    for key, value in state_dict.items():
        if "llama_model.model.layers" not in key:
            state_dict_without_llama[key] = value
    torch.save(state_dict_without_llama, file_path)
    model.to(model.device)


def final_validation(model, val_data_loader, results_dir):
    model.eval()
    predictions = []
    references = []
    anonymous_comments = []
    file_paths = []
    with torch.no_grad():
        val_pbar = tqdm(val_data_loader, desc='Final Validation')
        for samples in val_pbar:
            pred_texts, ref_texts, anonymous_comment, file_path = model(
                samples, True)
            predictions.extend(pred_texts)
            references.extend(ref_texts)
            anonymous_comments.extend(anonymous_comment)
            file_paths.extend(file_path)

    # 格式化用于计算指标
    ref_dict = {i: [ref.strip()] for i, ref in enumerate(references)}
    pred_dict = {i: [pred.strip()] for i, pred in enumerate(predictions)}

    # 计算指标
    metrics = calculate_metrics_of_set(ref_dict, pred_dict)

    # 准备结果数据
    paired_results = [
        {
            "prediction": pred.strip(),
            "reference": ref.strip(),
            "anonymous_comment": anon.strip(),
            "file_path": file_path
        }
        for pred, ref, anon, file_path in zip(predictions, references, anonymous_comments, file_paths)
    ]

    # 保存结果
    results = {
        'metrics': metrics,
        'pairs': paired_results
    }

    output_path = os.path.join(results_dir, 'final_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return metrics


if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(
        description="Train a model with FRANZ dataset.")
    parser.add_argument("--feature_root", type=str,
                        default="./features/features_CLIP")
    parser.add_argument("--window", type=float, default=15)
    parser.add_argument("--tokenizer_name", type=str,
                        default="/home/jiayuanrao/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct")
    parser.add_argument("--restricted_token_list_path", type=str,
                        default="./Restricted_Token_List/unanonymized_llama3.pkl")

    parser.add_argument("--train_ann_root", type=str,
                        default="./dataset/MatchTime/train")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--train_num_workers", type=int, default=32)
    parser.add_argument("--train_timestamp_key", type=str, default="contrastive_aligned_gameTime")

    parser.add_argument("--val_ann_root", type=str,
                        default="./dataset/MatchTime/valid")
    parser.add_argument("--val_batch_size", type=int, default=20)
    parser.add_argument("--val_num_workers", type=int, default=32)
    parser.add_argument("--val_timestamp_key", type=str,
                        default="contrastive_aligned_gameTime")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--num_video_query_token", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=512)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--model_output_dir", type=str, default="./ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")

    # If continue training from any epoch
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--pre_max_CIDEr", type=float, default=0.0)
    parser.add_argument("--pre_epoch", type=int, default=0)
    parser.add_argument("--load_ckpt", type=str,
                        default="./ckpt/model_save_best_val_CIDEr.pth")
    parser.add_argument("--exp_name", type=str, default="unanonymized_matchtime_1fps_aligned_time",
                        help="Experiment name (optional)")

    args = parser.parse_args()

    train(args)
