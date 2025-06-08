import os
import json
import shutil
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse
from pathlib import Path


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        return duration
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def analyze_and_sample_videos(input_dir, output_dir, num_samples=200):
    """分析视频长度分布并采样"""
    print("开始扫描视频文件...")

    # 收集所有视频及其时长
    video_info = defaultdict(list)  # 按比赛名组织视频信息
    for match_dir in os.listdir(input_dir):
        match_path = os.path.join(input_dir, match_dir)
        if os.path.isdir(match_path):
            for file in os.listdir(match_path):
                if file.endswith('.mp4'):
                    video_path = os.path.join(match_path, file)
                    duration = get_video_duration(video_path)
                    if duration is not None:
                        video_info[match_dir].append({
                            'path': video_path,
                            'duration': duration,
                            'filename': file
                        })

    if not video_info:
        print("未找到有效视频文件！")
        return

    # 计算总视频数量和时长
    all_durations = []
    all_videos = []
    for match_videos in video_info.values():
        for video in match_videos:
            all_durations.append(video['duration'])
            all_videos.append(video)

    durations = np.array(all_durations)
    print(f"\n总视频数量: {len(durations)}")
    print(f"平均时长: {np.mean(durations):.2f}秒")
    print(f"最短时长: {np.min(durations):.2f}秒")
    print(f"最长时长: {np.max(durations):.2f}秒")

    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, alpha=0.7, color='blue', label='原始分布')
    plt.title('duratioon distribution')
    plt.xlabel('时长(秒)')
    plt.ylabel('频数')
    plt.savefig(os.path.join(os.path.dirname(output_dir), 'duration_distribution.png'))
    plt.close()

    # 确保总采样数量为num_samples
    selected_videos = []
    num_bins = 10
    percentile_edges = np.percentile(
        durations, np.linspace(0, 100, num_bins + 1))

    # 计算每个bin的目标采样数量
    bin_counts = []
    for i in range(num_bins):
        bin_mask = (durations >= percentile_edges[i]) & (
            durations <= percentile_edges[i + 1])
        bin_size = np.sum(bin_mask)
        target_size = int(np.round(bin_size / len(durations) * num_samples))
        bin_counts.append(target_size)

    # 调整bin_counts确保总和等于num_samples
    while sum(bin_counts) != num_samples:
        if sum(bin_counts) > num_samples:
            max_idx = np.argmax(bin_counts)
            if bin_counts[max_idx] > 0:
                bin_counts[max_idx] -= 1
        else:
            max_idx = np.argmax([np.sum((durations >= percentile_edges[i]) &
                                        (durations <= percentile_edges[i + 1]))
                                 for i in range(num_bins)])
            bin_counts[max_idx] += 1

    # 对每个bin进行采样
    for i in range(num_bins):
        bin_mask = (durations >= percentile_edges[i]) & (
            durations <= percentile_edges[i + 1])
        bin_videos = [v for idx, v in enumerate(all_videos) if bin_mask[idx]]
        if bin_videos and bin_counts[i] > 0:
            selected = np.random.choice(
                len(bin_videos), bin_counts[i], replace=False)
            selected_videos.extend([bin_videos[idx] for idx in selected])

    # 创建输出目录结构并复制文件
    print("\n开始复制选中的视频样本...")
    copied_count = 0
    for video in tqdm(selected_videos):
        match_name = os.path.basename(os.path.dirname(video['path']))
        dst_match_dir = os.path.join(output_dir, match_name)
        os.makedirs(dst_match_dir, exist_ok=True)

        dst_path = os.path.join(dst_match_dir, os.path.basename(video['path']))
        shutil.copy2(video['path'], dst_path)
        copied_count += 1

    # 保存采样信息
    sample_info = {
        'total_videos': len(durations),
        'sampled_videos': copied_count,
        'duration_stats': {
            'mean': float(np.mean(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations)),
            'median': float(np.median(durations))
        },
        'sampled_files': [os.path.basename(v['path']) for v in selected_videos]
    }

    with open(os.path.join(output_dir, 'sampling_info.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_info, f, indent=2, ensure_ascii=False)

    print(f"\n采样完成！选取了 {copied_count} 个视频样本")
    print(f"采样结果保存在: {os.path.join(output_dir, 'sampled_clips')}")
    print(f"采样信息保存在: {os.path.join(output_dir, 'sampling_info.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对视频样本进行分布式采样')
    parser.add_argument('--input_dir', type=str,
                        default='../clips_resized/',
                        help='原始clips视频目录路径')
    parser.add_argument('--output_dir', type=str,
                        default='../tiny_clips',
                        help='采样结果输出目录')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='需要采样的视频数量')

    args = parser.parse_args()
    analyze_and_sample_videos(
        args.input_dir, args.output_dir, args.num_samples)
