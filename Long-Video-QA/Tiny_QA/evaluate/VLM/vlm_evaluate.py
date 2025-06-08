import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from qwen_vl_utils import process_vision_info
import sys
import math
import argparse
import json
import time
import torch
import numpy as np
# Import necessary classes based on potential models
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    pipeline,
    Qwen2_5_VLForConditionalGeneration,  # Specific import for Qwen
    LlavaOnevisionForConditionalGeneration  # Specific import for LlavaOnevision
)
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import warnings
import decord  # Import directly, will raise ImportError if not found
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(sys.path)
sys.path.insert(0, parent_dir)  # Insert parent dir into sys.path
print(sys.path)
from evaluate_qa import BaseEvaluator  # Now this should work

# Suppress specific warnings if needed (e.g., from transformers)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the parent directory (evaluate) to the Python path to allow relative import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Using decord directly
decord.bridge.set_bridge('torch')  # Use PyTorch tensors
print("Using decord for video loading.")

# ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VLMEvaluator(BaseEvaluator):
    """
    Base evaluator class, providing common functionalities for different vision-language models.
    """

    def __init__(self, qa_dir, match_data_folder, output_base_dir, test_name,
                 video_clip_dir, model_name_or_path, processor_name_or_path=None,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 attn_implementation=None,
                 model_load_kwargs=None,
                 max_num_frames_to_sample=16,
                 shuffle_choices=False, shuffle_times=4):
        super().__init__(qa_dir, match_data_folder, output_base_dir, test_name,
                         shuffle_choices=shuffle_choices, shuffle_times=shuffle_times)
        self.video_clip_dir = video_clip_dir
        self.model_name_or_path = model_name_or_path
        self.processor_name_or_path = processor_name_or_path if processor_name_or_path else model_name_or_path
        self.device = device  # Store device preference
        self.max_num_frames_to_sample = max_num_frames_to_sample
        self.attn_implementation = attn_implementation
        self.model_load_kwargs = model_load_kwargs or {}
        self.model_load_kwargs['trust_remote_code'] = True

        if attn_implementation:
            self.model_load_kwargs['attn_implementation'] = attn_implementation

        # Base class model and processor/tokenizer
        self.model = None
        self.processor = None
        self.tokenizer = None

        # Output configuration information
        print(f"--- VLM Evaluator Configuration ---")
        print(f"Model Path: {self.model_name_or_path}")
        print(f"Processor/Tokenizer Path: {self.processor_name_or_path}")
        print(f"Target Device: {self.device}")
        print(f"Max samples frames: {self.max_num_frames_to_sample}")
        if attn_implementation:
            print(f"Attention Implementation: {attn_implementation}")

    def _construct_video_path(self, match_id, qa_item):
        """Construct the expected path for a video clip."""
        qa_index = qa_item.get('index', 'N/A')
        event_type = qa_item.get('event_type', 'unknown')
        match_name_base = match_id  # Ensure this variable is defined in actual code
        video_filename = f"{match_name_base}_{event_type}_{qa_index}.mp4"
        video_path = os.path.join(
            self.video_clip_dir, match_id, video_filename)
        return video_path

    @staticmethod
    def load_video_frames(video_path, num_frames=16):
        """Load and sample frames using decord."""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            print(f"Warning: Video {video_path} has 0 frames.")
            return []

        # Sample frame indices uniformly
        indices = torch.linspace(
            0, total_frames - 1, num_frames, dtype=torch.long)
        # Ensure indices are valid
        indices = torch.clamp(indices, 0, total_frames - 1)

        # Return tensor (num_frames, H, W, C)
        frames = vr.get_batch(indices).float()
        return frames

    def get_model_answer(self, match_id, qa):
        """Get model's answer - to be implemented by subclasses"""
        raise NotImplementedError(
            "This method should be implemented by subclasses")

    def _validate_answer(self, model_answer_raw, choices, match_id, qa_index, full_response=""):
        """Validate the format of the model's answer and extract the final answer"""
        if model_answer_raw is None:
            print(
                f"Warning: Inference failed. Match: {match_id}, Q-Index: {qa_index}")
            return None

        valid_choice_prefixes = [choice.split(
            '.')[0].upper() for choice in choices if '.' in choice]
        model_answer_clean = model_answer_raw.strip('"` ')

        if len(model_answer_clean) == 1 and model_answer_clean in valid_choice_prefixes:
            return model_answer_clean
        else:
            if model_answer_clean and model_answer_clean[0] in valid_choice_prefixes:
                print(
                    f"Warning: Extracted first char '{model_answer_clean[0]}' from response: '{model_answer_raw}'. Match: {match_id}, Q-Index: {qa_index}")
                return model_answer_clean[0]

            print(
                f"Warning: Received invalid response format: '{model_answer_raw}' (Cleaned: '{model_answer_clean}'). Expected one of {valid_choice_prefixes}. Match: {match_id}, Q-Index: {qa_index}, Full Response Snippet: {full_response[:100]}...")
            return None


class QwenEvaluator(VLMEvaluator):
    """Evaluator for Qwen vision-language models"""

    def __init__(self, qa_dir, match_data_folder, output_base_dir, test_name,
                 video_clip_dir, model_name_or_path, processor_name_or_path=None,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 attn_implementation=None,
                 model_load_kwargs=None,
                 max_num_frames_to_sample=16,
                 shuffle_choices=False, shuffle_times=4):
        super().__init__(qa_dir, match_data_folder, output_base_dir, test_name,
                         video_clip_dir, model_name_or_path, processor_name_or_path,
                         device, attn_implementation, model_load_kwargs, max_num_frames_to_sample,
                         shuffle_choices=shuffle_choices, shuffle_times=shuffle_times)

        print("Loading Qwen model and processor...")

        # Qwen-specific model loading configuration
        self.model_load_kwargs['torch_dtype'] = torch.bfloat16
        self.model_load_kwargs['device_map'] = self.device if self.device != 'auto' else "auto"

        # Load model and processor
        print(self.model_load_kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **self.model_load_kwargs
        ).eval()
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        self.processor = AutoProcessor.from_pretrained(
            self.processor_name_or_path, trust_remote_code=True, use_fast=True)

        print("Qwen model and processor loaded successfully.")
        if hasattr(self.model, 'device'):
            print(f"Model loaded on device: {self.model.device}")

    @torch.inference_mode()
    def get_model_answer(self, match_id, qa):
        """Get answer using the Qwen model"""
        self.model.eval()

        question = qa['question']
        choices = qa['choices']
        qa_index = qa.get('index', 'N/A')

        # 1. Construct video path and check existence
        video_path = self._construct_video_path(match_id, qa)
        if not os.path.exists(video_path):
            print(
                f"Warning: Video clip not found for match {match_id}, Q-Index {qa_index} at {video_path}. Skipping.")
            return None

        # 2. Prepare prompts and inputs
        formatted_choices = "\n".join(choices)
        prompt_text = f"Watch the video clip and answer the multiple-choice question based ONLY on the video content.\nRespond with a single letter (A, B, C, or D) corresponding to the best answer.\n\nQuestion: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"

        # Qwen uses a message format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 1.0,
                        "resized_height": 224,
                        "resized_width": 224
                    },
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Process input
        target_device = self.model.device if hasattr(
            self.model, 'device') else self.device
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(target_device)

        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=10, do_sample=False)

        # Process output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        full_response = output_text[0] if output_text else ""
        model_answer_raw = full_response.strip().upper()

        # Validate and return the answer
        return self._validate_answer(model_answer_raw, choices, match_id, qa_index, full_response)


class InternVLEvaluator(VLMEvaluator):
    """Evaluator for InternVL vision-language models"""

    def __init__(self, qa_dir, match_data_folder, output_base_dir, test_name,
                 video_clip_dir, model_name_or_path, processor_name_or_path=None,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 attn_implementation=None,
                 model_load_kwargs=None,
                 max_num_frames_to_sample=16,
                 shuffle_choices=False, shuffle_times=4):
        super().__init__(qa_dir, match_data_folder, output_base_dir, test_name,
                         video_clip_dir, model_name_or_path, processor_name_or_path,
                         device, attn_implementation, model_load_kwargs, max_num_frames_to_sample,
                         shuffle_choices=shuffle_choices, shuffle_times=shuffle_times)

        print("Loading InternVL model and tokenizer...")

        # InternVL-specific model loading configuration
        self.model_load_kwargs['torch_dtype'] = torch.bfloat16
        self.model_load_kwargs['low_cpu_mem_usage'] = True
        self.model_load_kwargs['device_map'] = self.device if self.device != 'auto' else "auto"

        # Flash Attention support
        if attn_implementation and 'flash' in attn_implementation:
            self.model_load_kwargs['use_flash_attn'] = True
            if 'attn_implementation' in self.model_load_kwargs:
                del self.model_load_kwargs['attn_implementation']

        # Load model and tokenizer
        print(self.model_load_kwargs)
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path, **self.model_load_kwargs
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.processor_name_or_path, trust_remote_code=True, use_fast=False)

        print("InternVL model and tokenizer loaded successfully.")

    @staticmethod
    def build_transform(input_size=448):
        """Create image transformation function for InternVL"""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """InternVL3's dynamic preprocessing to handle different aspect ratios"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = InternVLEvaluator.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        """Find closest aspect ratio from target_ratios to the input aspect_ratio"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def get_video_indices(total_frames, num_segments=32):
        """Sample video frames uniformly across segments"""
        seg_size = float(total_frames) / num_segments
        indices = np.array([
            int((seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        # Clip to valid range
        indices = np.clip(indices, 0, total_frames - 1)
        return indices

    @staticmethod
    def load_video_frames_internvl(video_path, num_frames=16, input_size=448, max_num=1):
        """InternVL3-specific video frame loading and preprocessing function with LlavaOnevision-style frame sampling"""
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return None, None

        # Read video using decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

        if total_frames == 0:
            print(f"Warning: Video {video_path} has 0 frames.")
            return None, None

        # Use smart_nframes logic from LlavaOnevision
        def ceil_by_factor(n, f): return math.ceil(n / f) * f
        def floor_by_factor(n, f): return math.floor(n / f) * f

        # Calculate number of frames using LlavaOnevision parameters
        fps = 1.0
        frame_factor = 2
        min_frames = 4
        max_frames = 576  # Using LlavaOnevision's default max

        # Calculate ideal frame count based on target fps
        nframes = total_frames / video_fps * fps

        # Ensure frame count is within limits
        min_frames = ceil_by_factor(min_frames, frame_factor)
        max_frames = min(floor_by_factor(
            max_frames, frame_factor), total_frames)
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, frame_factor)

        # Ensure result is within valid range
        if not (frame_factor <= nframes and nframes <= total_frames):
            nframes = max(frame_factor, min(total_frames, nframes))
            nframes = floor_by_factor(nframes, frame_factor)

        num_frames = int(nframes)

        # Sample frame indices using segmentation approach
        frame_indices = InternVLEvaluator.get_video_indices(
            total_frames, num_segments=num_frames)

        # Create transformation function
        transform = InternVLEvaluator.build_transform(input_size)

        # Process each frame with dynamic preprocessing
        pixel_values_list = []
        num_patches_list = []

        for frame_index in frame_indices:
            # Get frame and convert to PIL image
            frame = vr[frame_index].asnumpy()
            img = Image.fromarray(frame).convert('RGB')

            # Apply InternVL's dynamic preprocessing
            processed_images = InternVLEvaluator.dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num
            )

            # Transform each processed image
            for processed_img in processed_images:
                pixel_values = transform(processed_img)
                pixel_values_list.append(pixel_values)

            # Record number of patches for this frame
            num_patches_list.append(len(processed_images))

        # Stack all processed frames
        pixel_values = torch.stack(pixel_values_list)

        return pixel_values, num_patches_list

    @torch.inference_mode()
    def get_model_answer(self, match_id, qa):
        """Get answer using the InternVL model"""
        self.model.eval()
        question = qa['question']
        choices = qa['choices']
        qa_index = qa.get('index', 'N/A')

        # 1. Construct video path and check existence
        video_path = self._construct_video_path(match_id, qa)
        if not os.path.exists(video_path):
            print(
                f"Warning: Video clip not found for match {match_id}, Q-Index {qa_index} at {video_path}. Skipping.")
            return None

        # 2. Use InternVL's specific method to load and preprocess frames
        video_frames, num_patches_list = self.load_video_frames_internvl(
            video_path, num_frames=self.max_num_frames_to_sample)

        if video_frames is None:
            print(
                f"Warning: Failed to load video frames for {video_path}. Skipping.")
            return None

        # 3. Prepare prompt text and model input
        formatted_choices = "\n".join(choices)
        prompt_text = f"Watch the video clip and answer the multiple-choice question based ONLY on the video content.\nRespond with a single letter (A, B, C, or D) corresponding to the best answer.\n\nQuestion: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"

        # Move video frames to appropriate device and convert to correct data type
        target_device = self.model.device if hasattr(
            self.model, 'device') else self.device
        video_frames = video_frames.to(target_device).to(torch.bfloat16)

        # Add prefix for video frames
        video_prefix = ''.join(
            [f'Frame{i+1}: <image>\n' for i in range(len(video_frames))])
        question_text = video_prefix + prompt_text

        # Set generation configuration
        generation_config = {'max_new_tokens': 10, 'do_sample': False}

        # Call InternVL's chat method to generate answer
        response = self.model.chat(
            self.tokenizer,
            video_frames,
            question_text,
            generation_config,
            num_patches_list=num_patches_list
        )

        # Process response
        full_response = response
        model_answer_raw = response.strip().upper()

        # Validate and return the answer
        return self._validate_answer(model_answer_raw, choices, match_id, qa_index, full_response)


class LlavaOnevisionEvaluator(VLMEvaluator):
    """Evaluator for LlavaOnevision vision-language models"""

    def __init__(self, qa_dir, match_data_folder, output_base_dir, test_name,
                 video_clip_dir, model_name_or_path, processor_name_or_path=None,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 attn_implementation=None,
                 model_load_kwargs=None,
                 max_num_frames_to_sample=16,
                 shuffle_choices=False, shuffle_times=4):
        super().__init__(qa_dir, match_data_folder, output_base_dir, test_name,
                         video_clip_dir, model_name_or_path, processor_name_or_path,
                         device, attn_implementation, model_load_kwargs, max_num_frames_to_sample,
                         shuffle_choices=shuffle_choices, shuffle_times=shuffle_times)

        print("Loading LlavaOnevision model and processor...")

        # LlavaOnevision-specific model loading configuration
        self.model_load_kwargs['torch_dtype'] = torch.float16
        self.model_load_kwargs['low_cpu_mem_usage'] = True
        self.model_load_kwargs['device_map'] = self.device if self.device != 'auto' else "auto"

        print(self.model_load_kwargs)
        # Load model and processor
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **self.model_load_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            self.processor_name_or_path, trust_remote_code=True, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        print("LlavaOnevision model and processor loaded successfully.")
        if hasattr(self.model, 'device'):
            print(f"Model loaded on device: {self.model.device}")

    @staticmethod
    def smart_nframes(total_frames, video_fps, fps=1.0, min_frames=4, max_frames=576, frame_factor=2):
        """Calculate the number of frames that should be extracted from the video

        Args:
            total_frames: Total number of frames in the video
            video_fps: Original video frame rate
            fps: Target frame rate for extraction, default 1.0
            min_frames: Minimum number of frames, default 4
            max_frames: Maximum number of frames, default 576
            frame_factor: Frame factor to make frame count a multiple of this value, default 2

        Returns:
            int: Calculated number of frames to extract
        """
        # Round up to multiple of frame_factor
        def ceil_by_factor(n, f): return math.ceil(n / f) * f
        # Round down to multiple of frame_factor
        def floor_by_factor(n, f): return math.floor(n / f) * f
        # Round to nearest multiple of frame_factor
        def round_by_factor(n, f): return round(n / f) * f

        # Calculate ideal frame count based on target fps
        nframes = total_frames / video_fps * fps

        # Ensure frame count is within limits
        min_frames = ceil_by_factor(min_frames, frame_factor)
        max_frames = min(floor_by_factor(
            max_frames, frame_factor), total_frames)
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, frame_factor)

        # Ensure result is within valid range
        if not (frame_factor <= nframes and nframes <= total_frames):
            nframes = max(frame_factor, min(total_frames, nframes))
            nframes = floor_by_factor(nframes, frame_factor)

        return int(nframes)

    @torch.inference_mode()
    def get_model_answer(self, match_id, qa):
        """Get answer using the LlavaOnevision model"""
        self.model.eval()
        question = qa['question']
        choices = qa['choices']
        qa_index = qa.get('index', 'N/A')

        # 1. Construct video path and check existence
        video_path = self._construct_video_path(match_id, qa)
        if not os.path.exists(video_path):
            print(
                f"Warning: Video clip not found for match {match_id}, Q-Index {qa_index} at {video_path}. Skipping.")
            return None

        # 2. Load video frames and calculate proper num_frames
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

        num_frames = self.smart_nframes(
            total_frames,
            video_fps,
            fps=1.0,
            min_frames=4,
            max_frames=self.max_num_frames_to_sample,
            frame_factor=2
        )

        # 3. Prepare prompts and model inputs
        formatted_choices = "\n".join(choices)
        prompt_text = f"Watch the video clip and answer the multiple-choice question based ONLY on the video content.\nRespond with a single letter (A, B, C, or D) corresponding to the best answer.\n\nQuestion: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        # 处理输入，使用计算得出的num_frames
        target_device = self.model.device if hasattr(
            self.model, 'device') else self.device

        # 使用新的API格式处理输入
        inputs = self.processor.apply_chat_template(
            conversation,
            num_frames=num_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(target_device)

        # Generate answer
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=10, do_sample=False)

        # Process output
        output_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        full_response = output_text[0] if output_text else ""

        # 提取回答部分（去掉对话历史）
        if "assistant" in full_response:
            full_response = full_response.split("assistant")[-1].strip()

        model_answer_raw = full_response.strip().upper()

        # Validate and return the answer
        return self._validate_answer(model_answer_raw, choices, match_id, qa_index, full_response)


class VideoLLaMA3Evaluator(VLMEvaluator):
    """Evaluator for VideoLLaMA3 vision-language models"""

    def __init__(self, qa_dir, match_data_folder, output_base_dir, test_name,
                 video_clip_dir, model_name_or_path, processor_name_or_path=None,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 attn_implementation=None,
                 model_load_kwargs=None,
                 max_num_frames_to_sample=16,
                 shuffle_choices=False, shuffle_times=4):
        super().__init__(qa_dir, match_data_folder, output_base_dir, test_name,
                         video_clip_dir, model_name_or_path, processor_name_or_path,
                         device, attn_implementation, model_load_kwargs, max_num_frames_to_sample,
                         shuffle_choices=shuffle_choices, shuffle_times=shuffle_times)

        print("Loading VideoLLaMA3 model and processor...")

        # VideoLLaMA3-specific model loading configuration
        self.model_load_kwargs['torch_dtype'] = torch.bfloat16
        self.model_load_kwargs['device_map'] = self.device if self.device != 'auto' else "auto"

        print(self.model_load_kwargs)
        # Load model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **self.model_load_kwargs
        )
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        self.processor = AutoProcessor.from_pretrained(
            self.processor_name_or_path, trust_remote_code=True)

        print("VideoLLaMA3 model and processor loaded successfully.")
        if hasattr(self.model, 'device'):
            print(f"Model loaded on device: {self.model.device}")

    @torch.inference_mode()
    def get_model_answer(self, match_id, qa):
        """Get answer using the VideoLLaMA3 model"""
        self.model.eval()
        question = qa['question']
        choices = qa['choices']
        qa_index = qa.get('index', 'N/A')

        # 1. Construct video path and check existence
        video_path = self._construct_video_path(match_id, qa)
        if not os.path.exists(video_path):
            print(
                f"Warning: Video clip not found for match {match_id}, Q-Index {qa_index} at {video_path}. Skipping.")
            return None

        # 2. Prepare prompts and inputs
        formatted_choices = "\n".join(choices)
        prompt_text = f"Watch the video clip and answer the multiple-choice question based ONLY on the video content.\nRespond with a single letter (A, B, C, or D) corresponding to the best answer.\n\nQuestion: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"

        # VideoLLaMA3 uses a conversation format
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video",
                     "video": {"video_path": video_path,
                               "fps": 1,
                               "max_frames": self.max_num_frames_to_sample}},
                    {"type": "text", "text": prompt_text},
                ]
            },
        ]

        # Process input
        target_device = self.model.device if hasattr(
            self.model, 'device') else self.device

        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.to(target_device) if isinstance(
            v, torch.Tensor) else v for k, v in inputs.items()}

        # Convert pixel values to bfloat16 if present
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Generate answer
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=10, do_sample=False)

        # Process output
        full_response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        model_answer_raw = full_response.strip().upper()

        # Validate and return the answer
        return self._validate_answer(model_answer_raw, choices, match_id, qa_index, full_response)


def get_evaluator_class(model_type):
    """Get appropriate evaluator class based on model type"""
    model_type = model_type.lower() if model_type else ""

    if model_type == 'qwen':
        return QwenEvaluator
    elif model_type == 'internvl':
        return InternVLEvaluator
    elif model_type == 'llava_onevision':
        return LlavaOnevisionEvaluator
    elif model_type == 'videollama3':
        return VideoLLaMA3Evaluator
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM-based evaluation for Tiny QA using video clips, supporting multiple models.")

    # --- Paths ---
    parser.add_argument("--qa_dir", type=str, default="../../First_Half_QA/",
                        help="Directory containing QA JSON files.")
    parser.add_argument("--match_data_folder", type=str, default="../../Subject_Data",
                        help="Directory containing match metadata JSON files.")
    parser.add_argument("--output_base_dir", type=str, default="evaluation_output",
                        help="Base directory to save evaluation results.")
    parser.add_argument("--test_name", type=str, required=True,
                        help="Unique name for this test run.")
    parser.add_argument("--video_clip_dir", type=str, default="../../clips/clips_resized",
                        help="Directory containing the pre-generated MP4 video clips.")

    # --- VLM Config ---
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Hugging Face model ID or local path.")
    parser.add_argument("--processor_name_or_path", type=str, default=None,
                        help="HF processor/tokenizer ID or path. Defaults to model_name_or_path.")
    parser.add_argument("--model_type", type=str, default=None, choices=['qwen', 'internvl', 'llava_onevision', 'llava',
                        'generic', 'videollama3'], help="Explicitly specify model type.")
    parser.add_argument("--device", type=str, default="auto" if torch.cuda.is_available()
                        else "cpu", help="Device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Attention implementation (e.g., 'flash_attention_2').")
    parser.add_argument("--max_frames", type=int, default=576,
                        help="Number of frames to sample from each video clip.")
    parser.add_argument("--shuffle_choices", action="store_true",
                        help="Whether to shuffle choices and use voting evaluation.")
    parser.add_argument("--shuffle_times", type=int, default=4,
                        help="Number of times to shuffle choices for each QA pair.")

    args = parser.parse_args()

    # --- Validate Paths ---
    if not os.path.isdir(args.qa_dir):
        print(f"Error: QA directory not found: {args.qa_dir}")
        sys.exit(1)
    if not os.path.isdir(args.video_clip_dir):
        print(f"Error: Video clip directory not found: {args.video_clip_dir}")
        sys.exit(1)

    # Determine model type
    if args.model_type is None:
        name_lower = args.model_name_or_path.lower()
        if 'qwen' in name_lower and 'llava-onevision' not in name_lower:
            model_type = 'qwen'
        elif 'internvl' in name_lower:
            model_type = 'internvl'
        elif 'llava-onevision' in name_lower:
            model_type = 'llava_onevision'
        elif 'llava' in name_lower:
            model_type = 'llava'
        elif 'videollama3' in name_lower:
            model_type = 'videollama3'
        else:
            print(
                f"Warning: Could not infer model type from '{args.model_name_or_path}'. Assuming generic VLM.")
            model_type = 'generic'
    else:
        model_type = args.model_type

    # Get appropriate evaluator class
    EvaluatorClass = get_evaluator_class(model_type)

    print(f"--- Initializing {model_type.capitalize()} Evaluator ---")
    evaluator = EvaluatorClass(
        qa_dir=args.qa_dir,
        match_data_folder=args.match_data_folder,
        output_base_dir=args.output_base_dir,
        test_name=args.test_name,
        video_clip_dir=args.video_clip_dir,
        model_name_or_path=args.model_name_or_path,
        processor_name_or_path=args.model_name_or_path,
        device=args.device,
        attn_implementation=args.attn_implementation,
        max_num_frames_to_sample=args.max_frames,
        shuffle_choices=args.shuffle_choices,
        shuffle_times=args.shuffle_times
    )

    print("\n--- Starting Evaluation Run ---")
    evaluator.run_evaluation()
    print("\n--- Evaluation Script Finished ---")
