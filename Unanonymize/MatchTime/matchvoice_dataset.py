import os
import random
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from transformers import AutoTokenizer
import copy

IGNORE_INDEX = -100


class MatchVoice_Dataset(Dataset):
    def __init__(self, feature_root, ann_root, window=15, fps=2, timestamp_key="gameTime",
                 tokenizer_name='meta-llama/Meta-Llama-3-8B', max_token_length=128):

        self.caption = traverse_and_parse(ann_root, timestamp_key, feature_root)
        self.feature_root = feature_root
        self.window = window
        self.fps = fps
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token_id = 128001
        # self.tokenizer.add_tokens(
        #     ["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])"], special_tokens=True)
        self.max_token_length = max_token_length

    def __getitem__(self, index):
        num_retries = 50
        fetched_features = None
        for _ in range(num_retries):
            half, timestamp, type, ground_truth_comment, league, game, _, _ = self.caption[index]
            feature_folder = os.path.join(self.feature_root, league, game)
            file_paths = [os.path.join(feature_folder, file) for file in os.listdir(
                feature_folder) if file.startswith(str(half)) and file.endswith(".npy")]
            # if len(file_paths) == 0:
            #     print(feature_folder)
            features = load_adjusted_features(
                file_paths[0], timestamp, self.window, self.fps)
            fetched_features = torch.from_numpy(features)

            ground_truth_comment_tokens = self.tokenizer(
                ground_truth_comment,
                return_tensors="pt",
                max_length=self.max_token_length,
                truncation=True
            ).input_ids[0]
            break
        else:
            raise RuntimeError(
                f"Failed to fetch video after {num_retries} retries.")
        return {
            "features": fetched_features,
            "tokens_input_ids": ground_truth_comment_tokens,
            "caption_info": self.caption[index]
        }

    def __len__(self):
        return len(self.caption)

    def collater(self, instances):
        input_ids = [
            torch.cat((
                       instance["tokens_input_ids"],
                       torch.tensor([self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")]))) 
            for instance in instances]  # add end token
        labels = copy.deepcopy(input_ids)
        caption_info = [instance["caption_info"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(
                self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")),
            labels=labels,
            caption_info=caption_info
        )

        if 'features' in instances[0]:
            features = [instance['features'] for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['features'] = torch.stack(features)
            else:
                batch['features'] = features
        return batch


def load_adjusted_features(feature_path, timestamp, window, fps=2):
    """
    Load and adjust video features based on the given timestamp and window.

    Args:
    - feature_path (str): The path to the .npy file containing video features.
    - timestamp (int): The target timestamp in seconds.
    - window (float): The window size in seconds.

    Returns:
    - np.array: The adjusted array of video features.
    """
    features = np.load(feature_path)[::2, :]
    total_frames = int(window * 2 * fps)  # Total frames to extract
    if timestamp * fps > len(features):
        print(f"Timestamp {timestamp} is out of range for feature file {feature_path}")
        return None

    start_frame = int(max(0, timestamp - window) * fps + 1)
    end_frame = int((timestamp + window) * fps + 1)
    if end_frame > len(features):
        # Adjust to get the last total_frames
        start_frame = int(max(0, len(features) - total_frames))
    ad = features[start_frame:start_frame + total_frames]
    return ad


def parse_labels_caption(file_path, league, game, timestamp_key):
    """
    Parses a Labels-caption.json file and extracts the required data.
    Parameters:
        file_path (str): The path to the Labels-caption.json file.
        league (str): The league name.
        game (str): The game name.
    Returns:
        list: A list of tuples containing (half, timestamp, type, ground_truth_comment, league, game).
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    result = []
    for annotation in data.get('annotations', []):
        try:
            gameTime, _ = annotation.get(timestamp_key, ' - ').split(' - ')
            half = int(gameTime.split(' ')[0])
            if half not in [1, 2]:
                continue
            minutes, seconds = map(int, _.split(':'))
            timestamp = minutes * 60 + seconds
            if timestamp > 2685:
                continue
            label = annotation.get('label', '')
            ground_truth_comment = annotation.get('description', '')
            anonymous_comment = annotation.get("anonymized", '')
            result.append(
                (half, timestamp, label, ground_truth_comment, league, game, anonymous_comment, file_path))
        except ValueError:
            continue
    # print(len(result))
    return result


def traverse_and_parse(root_dir, timestamp_key, feature_root_dir, fps=2):
    """
    Traverses a directory and its subdirectories to find and parse all Labels-caption.json files.
    Parameters:
        root_dir (str): The root directory to start traversal.
    Returns:
        list: A combined list of tuples from all Labels-caption.json files found.
    """
    all_data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'Labels-caption.json' or file == "Labels-caption_with_gt.json" or file == "Labels-caption_event_aligned_with_contrastive.json":
                league = os.path.basename(os.path.dirname(subdir))
                game = os.path.basename(subdir)
                file_path = os.path.join(subdir, file)
                infos = parse_labels_caption(
                    file_path, league, game, timestamp_key)
                half, timestamp, type, ground_truth_comment, league, game, _, _ = infos[0]
                feature_folder = os.path.join(feature_root_dir, league, game)
                file_paths = [os.path.join(feature_folder, file) for file in os.listdir(
                feature_folder) if file.startswith(str(half)) and file.endswith(".npy")]
                if len(file_paths) == 0:
                    continue
                features = np.load(file_paths[0])
                infos = [info for info in infos if info[1] * fps < len(features)]
                all_data.extend(infos)
    return all_data


