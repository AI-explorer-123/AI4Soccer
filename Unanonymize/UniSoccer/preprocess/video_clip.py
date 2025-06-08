import json
import os
import ffmpeg
from pathlib import Path
import re
import glob
from tqdm import tqdm


def extract_time_from_path(video_path):
    """从视频路径中提取时间 (例如: 43_52.mp4)"""
    time_pattern = r'(\d+)_(\d+)_(\d+)\.mp4$'
    match = re.search(time_pattern, video_path)
    if match:
        half, minutes, seconds = map(int, match.groups())
        return half, minutes * 60 + seconds
    return None


def clip_video(input_path, output_path, center_time, window_size=30):
    """
    在中心时间点周围裁剪指定时长的视频
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        center_time: 中心时间点(秒)
        window_size: 窗口大小(秒)
    """
    start_time = max(0, center_time - window_size / 2)

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        stream = ffmpeg.input(input_path, ss=start_time, t=window_size)
        stream = ffmpeg.filter(stream, 'scale', 224, 224)  # 调整尺寸到224x224
        stream = ffmpeg.output(stream, output_path,
                               vcodec='libx264',  # 使用H.264编码
                               preset='medium',   # 编码速度和质量的平衡
                               crf=23,            # 控制视频质量
                               an=None)           # 删除音频部分
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        print(
            f"处理视频时出错 {input_path}: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        print(f"处理视频时发生意外错误 {input_path}: {str(e)}")


def process_json_file(json_path, base_video_path, output_base_path):
    """处理单个JSON文件"""
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        print(f"正在处理文件: {os.path.basename(json_path)}")

        for i, item in enumerate(tqdm(json_data)):
            if "video" not in item:
                continue

            video_path = item["video"]
            if not video_path:
                continue

            match_path = os.path.dirname(video_path)
            # 获取分钟和秒数
            time_str = os.path.basename(video_path)
            half, center_time = extract_time_from_path(time_str)

            if center_time is None:
                continue

            input_video = os.path.join(
                base_video_path, match_path, f"{half}_224p.mkv")
            output_video = os.path.join(output_base_path, video_path)

            if not os.path.exists(input_video):
                print(f"未找到输入视频: {input_video}")
                continue

            clip_video(input_video, output_video, center_time)

    except json.JSONDecodeError as e:
        print(f"JSON文件解析错误 {json_path}: {str(e)}")
    except Exception as e:
        print(f"处理JSON文件时发生错误 {json_path}: {str(e)}")


def main():
    # 定义路径
    json_dir = "/remote-home/jiayuanrao/haokai/UniSoccer/train_data/json/MatchTime"
    base_video_path = "/remote-home/jiayuanrao/haokai/Dataset/SoccerNet/Raw_match"
    output_base_path = "/remote-home/jiayuanrao/haokai/UniSoccer/train_data/video_clips"

    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    print(f"找到 {len(json_files)} 个JSON文件需要处理")

    # 处理每个JSON文件
    for json_file in json_files:
        process_json_file(json_file, base_video_path, output_base_path)


if __name__ == "__main__":
    main()
