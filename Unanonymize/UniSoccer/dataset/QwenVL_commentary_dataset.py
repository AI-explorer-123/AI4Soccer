import os
import torch
import json
import copy
import random
import numpy as np
import re
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
from dataset.video_utils_siglip import read_frames_decord
from qwen_vl_utils import process_vision_info
from transformers.trainer_utils import EvalPrediction
from utils.score_helper import calculate_metrics_of_set

IGNORE_INDEX = -100


class QwenVLCommentaryDataset(Dataset):
    def __init__(self,
                 json_file,
                 video_base_dir=None,
                 match_info_base_dir=None,
                 processor_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Dataset for Qwen2.5-VL commentary generation.
        Can be loaded from pre-extracted npy feature files or raw video files.

        Args:
            json_file: Path or list of paths to JSON files.
            video_base_dir: Base directory or list of directories for video files.
            match_info_base_dir: Base directory for match information JSON files.
            num_frames: Number of frames to extract from each video.
            processor_name: Qwen2.5-VL processor name from Hugging Face.
        """
        self.processor = AutoProcessor.from_pretrained(processor_name,
                                                       trust_remote_code=True,
                                                       use_fast=True)
        # Tokenizer is part of the processor for Qwen-VL
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "left"

        self.video_base_dir = video_base_dir
        self.match_info_base_dir = match_info_base_dir

        self.multiple_json = isinstance(json_file, list)
        self.data = []

        json_files_to_load = json_file if isinstance(
            json_file, list) else [json_file]
        video_base_dirs_list = video_base_dir if isinstance(video_base_dir, list) else [
            video_base_dir] * len(json_files_to_load)

        for i, current_json_file in enumerate(json_files_to_load):
            with open(current_json_file, 'r') as file:
                current_data = json.load(file)
                for item in current_data:
                    # Store original relative video path for match info extraction
                    item["original_video_path"] = item["video"]
                    # item["video"] path becomes the full path to the video file
                    item["video"] = os.path.join(
                        video_base_dirs_list[i] if self.multiple_json else self.video_base_dir, item["video"])
                self.data.extend(current_data)
                print(
                    f"File loaded: {current_json_file}, {len(current_data)} items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_info = self.data[idx]

        # 1. load match_info
        match_info_path = self.extract_match_info_path(
            video_info["original_video_path"])
        match_info = self.load_match_info(match_info_path)

        # 2. construct user prompt
        user_prompt = self.build_prompt(match_info)

        # 3. construct system prompt
        system_prompt = "You are a football commentary expert. Please generate a professional and concise English commentary for the given video clip, using the provided team and player names if relevant. ONLY return the commentary text. Do not include any other information or explanations."

        # 4. constuct full chat
        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_info["video"],
                        "fps": 1.0,
                        "resized_height": 224,
                        "resized_width": 224
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": video_info.get("comments_text", "")
                    }
                ]
            }
        ]
        return chat

    def collate_fn(self, examples):
        # examples: list of chat dicts
        # 1. 用processor.apply_chat_template生成文本
        texts = [self.processor.apply_chat_template(
            example, tokenize=False) for example in examples]

        # 2. 处理视觉输入（支持图片和视频）
        video_inputs = []
        for example in examples:
            _, vid_inp = process_vision_info(example)
            video_inputs.append(vid_inp)

        # 3. 编码文本和视觉输入
        batch = self.processor(
            text=texts,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        )

        # 4. 构造labels，屏蔽pad和image token
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        video_token_id = [151652, 151653, 151656]
        for video_token_id in video_token_id:
            labels[labels == video_token_id] = -100

        # assistant_token_position = torch.where(
        #     batch["input_ids"] == self.processor.tokenizer.convert_tokens_to_ids("<|assistant|>"))[0]
        # if len(assistant_token_position) > 0:
        #     labels[:assistant_token_position+1] = -100
        

        # 找到每个样本中 <|assistant|> 的位置，并将该位置之前的内容设置为 0
        for i in range(batch["input_ids"].shape[0]):
            assistant_positions = torch.where(
                batch["input_ids"][i] == self.processor.tokenizer.convert_tokens_to_ids(
                    "assistant")
            )[0]
            if len(assistant_positions) > 0:
                labels[i, :assistant_positions + 2] = -100

        batch["labels"] = labels
        return batch

    def extract_match_info_path(self, original_video_path):
        """
        Extract match information path from the original relative video path.
        Example original_video_path: "england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/video.mp4"
        """
        # Assuming original_video_path is like "league_season/match_name/video_file.mp4"
        # We want "league_season/match_name"
        match_identifier_parts = original_video_path.split(
            '/')[:-1]  # Remove filename
        if not match_identifier_parts:
            return None
        match_identifier = "/".join(match_identifier_parts)

        match_info_path = os.path.join(
            self.match_info_base_dir, match_identifier, "Labels-caption.json")
        return match_info_path

    @staticmethod
    def load_match_info(match_info_path):
        if not match_info_path or not os.path.exists(match_info_path):
            return None
        try:
            with open(match_info_path, 'r') as file:
                match_info = json.load(file)
            return match_info
        except Exception as e:
            print(f"Error loading match info from {match_info_path}: {e}")
            return None

    @staticmethod
    def build_prompt(match_info):
        """
        构建user prompt，包含队伍名和两队所有球员名
        """
        if not match_info:
            return "No match info provided."

        home_team = match_info.get("gameHomeTeam", "Home Team")
        away_team = match_info.get("gameAwayTeam", "Away Team")

        # home队球员
        if "lineup" in match_info:
            if "home" in match_info["lineup"]:
                home_players = [
                    f"Number {player['shirt_number']}:{player['long_name']}"
                    for player in match_info["lineup"]["home"]["players"]
                ]

            # away队球员
            if "away" in match_info["lineup"]:
                away_players = [
                    f"Number {player['shirt_number']}: {player['long_name']}"
                    for player in match_info["lineup"]["home"]["players"]
                ]

        prompt = f"Match: {home_team} (Home Team) vs {away_team} (Away Team).\n"
        # prompt += f"{home_team} (Home Team) Players: " + ", ".join(
        #     home_players) + f"\n{away_team} (Away Team) Players: " + ", ".join(away_players) + "\n"
        return prompt

    def compute_metrics(self, eval_pred: EvalPrediction):
        # 解码预测和标签
        predictions, label_ids = eval_pred
        # 处理为字符串
        pred_texts = self.processor.tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        label_texts = self.processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)

        # 组装为字典形式
        # 这里假设每个样本一个句子
        references = {i: [label_texts[i].strip()] for i in range(len(label_texts))}
        hypotheses = {i: [pred_texts[i].strip()] for i in range(len(pred_texts))}

        # 计算指标
        metrics = calculate_metrics_of_set(references, hypotheses)
        # SFTTrainer会自动加上eval_前缀
        return metrics
