import os
import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time


def resize_video(video_path, output_path, target_size=(224, 224)):
    """
    将视频调整为指定尺寸

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        target_size: 目标尺寸，默认(224, 224)
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return False

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 调整帧大小
            resized_frame = cv2.resize(frame, target_size)

            # 写入新帧
            out.write(resized_frame)

        # 释放资源
        cap.release()
        out.release()
        return True

    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {e}")
        return False


def get_all_mp4_files(input_dir):
    """
    递归获取目录及其子目录下的所有MP4文件

    Args:
        input_dir: 输入目录路径

    Returns:
        列表，包含所有MP4文件的完整路径和相对路径
    """
    mp4_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir)
                mp4_files.append((full_path, rel_path))
    return mp4_files


def process_videos(input_dir, output_dir, num_workers=8):
    """
    处理目录下所有MP4视频（包括子目录）

    Args:
        input_dir: 输入视频目录
        output_dir: 输出视频目录
        num_workers: 工作线程数，默认为8
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有MP4文件
    video_files = get_all_mp4_files(input_dir)

    if not video_files:
        print(f"在 {input_dir} 及其子目录中未找到MP4文件")
        return

    # 使用多线程处理视频
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for full_path, rel_path in video_files:
            # 创建对应的输出目录结构
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            futures[executor.submit(
                resize_video, full_path, output_path)] = rel_path

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理视频"):
            video = futures[future]



def main():
    """主函数"""
    # 设置输入和输出目录
    input_dir = r'/DB/data/jiayuanrao-1/sports/haokai_intern/Dataset/Long-Video-QA/Tiny_QA/clips'
    output_dir = r'/DB/data/jiayuanrao-1/sports/haokai_intern/Dataset/Long-Video-QA/Tiny_QA/clips_resized'

    # 记录开始时间
    start_time = time.time()

    print(f"开始处理视频文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 处理视频
    process_videos(input_dir, output_dir, num_workers=max(1, os.cpu_count() // 2 if os.cpu_count() else 1))

    # 计算总用时
    end_time = time.time()
    print(f"处理完成! 总用时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
