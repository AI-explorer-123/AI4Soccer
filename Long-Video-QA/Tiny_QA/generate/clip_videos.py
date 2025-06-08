import os
import json
import subprocess
import argparse
import datetime
from tqdm import tqdm
import concurrent.futures
import time  # For potential delays if needed

# --- Optional Decord Import for Validation ---
try:
    import decord
    # decord.bridge.set_bridge('torch') # Not needed for basic validation
    decord_available = True
    print("Info: 'decord' library found. Will use it for validating existing clips.")
except ImportError:
    decord_available = False
    print("Warning: 'decord' library not found. Existing clip validation will only check file size (>0). Install 'decord' for more robust checks.")
# --- End Optional Import ---


def load_json(filepath):
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file - {filepath}")
        return None


def parse_timestamp_to_seconds(timestamp_str):
    """Converts MM:SS timestamp string to total seconds."""
    try:
        minutes, seconds = map(int, timestamp_str.split(':'))
        return minutes * 60 + seconds
    except ValueError:
        print(
            f"Error: Invalid timestamp format - {timestamp_str}. Expected MM:SS.")
        return None


def get_event_timestamp(match_data, event_index):
    """Retrieves the timestamp (in seconds) for a given event index."""
    if match_data and 0 <= event_index < len(match_data.get('comments', [])):
        timestamp_str = match_data['comments'][event_index].get('time_stamp')
        if timestamp_str:
            return parse_timestamp_to_seconds(timestamp_str)
    return None


def find_previous_event_timestamp(match_data, current_index, event_type):
    """Finds the timestamp (in seconds) of the previous event of the same type."""
    if not match_data or not match_data.get('comments'):
        return None
    previous_timestamp_sec = None
    comments = match_data.get('comments', [])
    for i in range(current_index - 1, -1, -1):
        if i < len(comments) and comments[i].get('comments_type') == event_type:
            timestamp_str = comments[i].get("time_stamp")
            if timestamp_str:
                previous_timestamp_sec = parse_timestamp_to_seconds(
                    timestamp_str)
                if previous_timestamp_sec is not None:
                    break  # Found a valid previous timestamp
            else:
                print(
                    f"Warning: Missing 'time_stamp' for event index {i} while searching for previous event.")

    return previous_timestamp_sec


def format_time(seconds):
    """Converts seconds into HH:MM:SS.ms format for ffmpeg."""
    if seconds < 0:
        seconds = 0
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# --- New Validation Function ---
def is_video_valid(filepath, min_duration=1.0):
    """
    Checks if a video file is likely valid using decord (if available).
    Checks if it can be opened and has a minimum duration.
    """
    if not decord_available:
        return True  # Fallback: rely on size check if decord is not installed

    try:
        # Use cpu context for validation to avoid GPU memory issues
        vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
        num_frames = len(vr)
        if num_frames == 0:
            print(f"Validation failed: Video has 0 frames - {filepath}")
            return False

        # Attempt to get average fps, handle potential division by zero if vr.get_avg_fps() is 0
        avg_fps = vr.get_avg_fps()
        if avg_fps <= 0:
            # If FPS is invalid, check if frame count is reasonable (e.g., > 1)
            print(
                f"Validation warning: Invalid average FPS ({avg_fps}) for {filepath}. Checking frame count > 1.")
            return num_frames > 1  # Consider valid if it has more than one frame

        duration = num_frames / avg_fps
        if duration < min_duration:
            print(
                f"Validation failed: Video duration ({duration:.2f}s) is less than minimum ({min_duration}s) - {filepath}")
            return False

        return True  # Looks valid
    except Exception as e:
        print(
            f"Validation failed: Error opening or reading video with decord - {filepath}\nError: {e}")
        return False
# --- End Validation Function ---


def clip_video(input_path, output_path, start_time_sec, end_time_sec):
    """Clips the video using ffmpeg."""
    start_time_str = format_time(start_time_sec)
    end_time_str = format_time(end_time_sec)
    duration_sec = end_time_sec - start_time_sec
    if duration_sec <= 0:
        print(
            f"Warning: Invalid time range for {output_path}. Start: {start_time_sec}, End: {end_time_sec}. Skipping.")
        return False  # Indicate failure due to invalid range

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        'ffmpeg',
        '-i', input_path,
        '-ss', start_time_str,
        '-to', end_time_str,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',
        '-loglevel', 'error',
        output_path
    ]
    try:
        result = subprocess.run(command, check=True, text=True,
                                capture_output=True)
        # Verify output file integrity *after* clipping attempt
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            print(
                f"\nError: ffmpeg command seemed to succeed but {output_path} is missing or empty.")
            print(f"Command: {' '.join(command)}")
            if result.stderr:
                print(f"ffmpeg stderr: {result.stderr.strip()}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\nError clipping video {input_path} to {output_path}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            print(f"Stderr: {e.stderr.strip()}")
        if os.path.exists(output_path):
            try:
                if os.path.getsize(output_path) == 0:
                    os.remove(output_path)
                    print(f"Removed empty output file: {output_path}")
            except OSError:
                pass
        return False
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Make sure ffmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"\nUnexpected error during clip_video for {output_path}: {e}")
        return False


def process_qa_files(qa_dir, match_dir, raw_match_dir, output_dir, num_workers):
    """Processes all QA files and clips corresponding videos using multiple threads."""
    qa_files = [f for f in os.listdir(qa_dir) if f.endswith('.json')]
    if not qa_files:
        print(f"No JSON files found in {qa_dir}")
        return

    tasks = []
    skipped_qa_count = 0
    already_exist_count = 0
    error_clip_count = 0
    print("Preparing clipping tasks...")

    # --- Prepare all clipping tasks ---
    for qa_filename in tqdm(qa_files, desc="Reading QA files & Preparing Tasks"):
        qa_filepath = os.path.join(qa_dir, qa_filename)
        qa_data = load_json(qa_filepath)
        if not qa_data:
            print(f"Warning: Could not load or empty QA file: {qa_filename}")
            continue

        match_name, _ = os.path.splitext(qa_filename)
        match_subdir_path = os.path.join(match_dir, match_name)

        if not os.path.isdir(match_subdir_path):
            print(
                f"Skipping {match_name}: Match data directory not found - {match_subdir_path}")
            skipped_qa_count += len(qa_data)
            continue

        match_json_files = [f for f in os.listdir(
            match_subdir_path) if f.endswith('.json')]
        if not match_json_files:
            print(
                f"Skipping {match_name}: No JSON file found in {match_subdir_path}")
            skipped_qa_count += len(qa_data)
            continue
        match_filename = match_json_files[0]
        match_filepath = os.path.join(match_subdir_path, match_filename)
        match_data = load_json(match_filepath)
        if not match_data:
            print(f"Skipping {match_name}: match data not found or invalid.")
            skipped_qa_count += len(qa_data)
            continue

        raw_match_video_dir = os.path.join(raw_match_dir, match_name)
        raw_mkv_files = [f for f in os.listdir(raw_match_video_dir) if f.endswith(
            '_1.mkv')] if os.path.isdir(raw_match_video_dir) else []
        if not raw_mkv_files:
            print(
                f"Skipping {match_name}: Raw match MKV file ending in '_1.mkv' not found in {raw_match_video_dir}")
            skipped_qa_count += len(qa_data)
            continue
        raw_match_filename = raw_mkv_files[0]
        raw_match_filepath = os.path.join(
            raw_match_video_dir, raw_match_filename)

        if not os.path.exists(raw_match_filepath):
            print(
                f"Skipping {match_name}: Raw match file path constructed but not found - {raw_match_filepath}")
            skipped_qa_count += len(qa_data)
            continue

        match_output_dir = os.path.join(output_dir, match_name)
        os.makedirs(match_output_dir, exist_ok=True)

        for i, qa_pair in enumerate(qa_data):
            task_added = False
            event_type = None
            qa_index_str = qa_pair.get('index', f'idx{i}')
            try:
                event_type = qa_pair.get('event_type')
                question_type = qa_pair.get('type')
                current_event_index = qa_pair.get('current_event_index')
                related_event_indices = qa_pair.get('related_event_index', [])

                if not event_type:
                    print(
                        f"Warning: Skipping QA pair {qa_index_str} in {qa_filename} due to missing 'event_type'.")
                    continue

                start_time_sec = None
                end_time_sec = None

                if question_type == 'half':
                    if not related_event_indices:
                        print(
                            f"Warning: Skipping 'half' QA pair {qa_index_str} in {qa_filename} due to empty 'related_event_index'.")
                        continue
                    min_related_index = min(related_event_indices)
                    max_related_index = max(related_event_indices)
                    min_related_event_timestamp = get_event_timestamp(
                        match_data, min_related_index)
                    max_related_event_timestamp = get_event_timestamp(
                        match_data, max_related_index)
                    if min_related_event_timestamp is None or max_related_event_timestamp is None:
                        print(
                            f"Warning: Skipping 'half' QA pair {qa_index_str} in {qa_filename}. Could not find timestamp for min/max related index ({min_related_index}/{max_related_index}).")
                        continue
                    start_time_sec = min_related_event_timestamp - 15
                    end_time_sec = max_related_event_timestamp + 15
                elif question_type in ['past', 'future']:
                    if current_event_index is None:
                        print(
                            f"Warning: Skipping '{question_type}' QA pair {qa_index_str} in {qa_filename} due to missing 'current_event_index'.")
                        continue
                    current_event_timestamp = get_event_timestamp(
                        match_data, current_event_index)
                    if current_event_timestamp is None:
                        print(
                            f"Warning: Skipping '{question_type}' QA pair {qa_index_str} in {qa_filename}. Could not find timestamp for current_event_index {current_event_index}.")
                        continue
                    end_time_sec = current_event_timestamp + 15
                    if question_type == 'past':
                        if not related_event_indices:
                            start_time_sec = current_event_timestamp - 300
                        else:
                            min_related_index = min(related_event_indices)
                            min_related_event_timestamp = get_event_timestamp(
                                match_data, min_related_index)
                            if min_related_event_timestamp is None:
                                print(
                                    f"Warning: Skipping 'past' QA pair {qa_index_str} in {qa_filename}. Could not find timestamp for min related index {min_related_index}.")
                                continue
                            start_time_sec = min_related_event_timestamp - 15
                    elif question_type == 'future':
                        prev_event_timestamp = find_previous_event_timestamp(
                            match_data, current_event_index, event_type)
                        if prev_event_timestamp is None:
                            start_time_sec = current_event_timestamp - 300
                        else:
                            start_time_sec = prev_event_timestamp - 15
                else:
                    print(
                        f"Warning: Skipping QA pair {qa_index_str} in {qa_filename}. Unknown question_type: {question_type}")
                    continue

                if start_time_sec is not None and end_time_sec is not None:
                    start_time_sec = max(0, start_time_sec)
                    duration_sec = end_time_sec - start_time_sec

                    if duration_sec > 0:
                        output_filename = f"{match_name}_{event_type}_{qa_index_str}.mp4"
                        output_filepath = os.path.join(
                            match_output_dir, output_filename)

                        clip_needed = True
                        if os.path.exists(output_filepath):
                            try:
                                file_size = os.path.getsize(output_filepath)
                                if file_size > 0:
                                    if is_video_valid(output_filepath):
                                        already_exist_count += 1
                                        clip_needed = False
                                    else:
                                        print(
                                            f"Info: Existing clip found but failed validation, will re-clip: {output_filepath}")
                                else:
                                    print(
                                        f"Warning: Existing clip found but is empty, will re-clip: {output_filepath}")
                            except OSError as e:
                                print(
                                    f"Warning: Could not get size/validate existing file {output_filepath}, assuming re-clip needed. Error: {e}")

                        if clip_needed:
                            tasks.append(
                                (raw_match_filepath, output_filepath, start_time_sec, end_time_sec))
                            task_added = True
                    else:
                        print(
                            f"Warning: Skipping QA pair {qa_index_str} in {qa_filename}. Invalid Fduration ({duration_sec}s) calculated: Start={start_time_sec}, End={end_time_sec}.")

            except Exception as e:
                print(
                    f"\nError processing QA pair {qa_index_str} in {qa_filename}: {e}")
                import traceback
                traceback.print_exc()

            if not task_added:
                should_have_existed_and_be_valid = False
                if event_type:
                    output_filename_check = f"{match_name}_{event_type}_{qa_index_str}.mp4"
                    output_filepath_check = os.path.join(
                        match_output_dir, output_filename_check)
                    if os.path.exists(output_filepath_check):
                        try:
                            if os.path.getsize(output_filepath_check) > 0:
                                if is_video_valid(output_filepath_check):
                                    should_have_existed_and_be_valid = True
                        except OSError:
                            pass

                if not should_have_existed_and_be_valid:
                    skipped_qa_count += 1

    if not tasks:
        print(
            f"No new clipping tasks found to process. Skipped {skipped_qa_count} QA pairs due to missing data, errors, or invalid calculated times. Found {already_exist_count} already existing valid clips.")
        return

    print(f"Found {len(tasks)} new clipping tasks to process. Skipped {skipped_qa_count} QA pairs. Found {already_exist_count} already existing valid clips.")

    print(f"Starting clipping process with {num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(
            clip_video, *task): task for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Clipping All Videos"):
            task_args = future_to_task[future]
            input_path, output_path, _, _ = task_args
            try:
                success = future.result()
                if not success:
                    error_clip_count += 1
            except Exception as exc:
                error_clip_count += 1
                print(
                    f'\nTask for {output_path} generated an unexpected exception during execution: {exc}')
                import traceback
                traceback.print_exc()

    print(
        f"\nClipping finished. {len(tasks) - error_clip_count} clips processed successfully.")
    if error_clip_count > 0:
        print(
            f"Encountered errors during {error_clip_count} clipping tasks (check logs above for details).")


if __name__ == "__main__":
    default_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)

    parser = argparse.ArgumentParser(
        description="Clip soccer match videos based on QA data using multiple threads.")
    parser.add_argument('--qa_dir', type=str, required=False,
                        default='/DB/data/jiayuanrao-1/sports/haokai_intern/Dataset/Long-Video-QA/Tiny_QA/First_Half_QA',
                        help='Directory containing the First_Half_QA JSON files.')
    parser.add_argument('--match_dir', type=str, required=False,
                        default='/DB/data/jiayuanrao-1/sports/haokai_intern/Dataset/Long-Video-QA/Tiny_QA/Subject_Data',
                        help='Directory containing the match_Data JSON files.')
    parser.add_argument('--raw_match_dir', type=str, required=False,
                        default='/DB/data/jiayuanrao-1/sports/haokai_intern/Dataset/SoccerReplay-1988/Raw_match/england_epl_2017-2018',
                        help='Directory containing league/season folders, which in turn contain match folders with MKV files (e.g., .../england_epl_2017-2018/match_name/match_name_1.mkv).')
    parser.add_argument('--output_dir', type=str, required=False,
                        default='/DB/data/jiayuanrao-1/sports/haokai_intern/Dataset/Long-Video-QA/Tiny_QA/clips',
                        help='Directory where the output MP4 clips will be saved inside match-specific subfolders.')
    parser.add_argument('--num_workers', type=int, default=default_workers,
                        help=f'Number of worker threads for clipping (default: {default_workers}).')

    args = parser.parse_args()

    if not os.path.isdir(args.qa_dir):
        print(f"Error: QA directory not found: {args.qa_dir}")
        exit(1)
    if not os.path.isdir(args.match_dir):
        print(f"Error: Match data directory not found: {args.match_dir}")
        exit(1)
    if not os.path.isdir(args.raw_match_dir):
        print(f"Error: Raw match directory not found: {args.raw_match_dir}")
        exit(1)
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(
                f"Error: Output directory does not exist and could not be created: {args.output_dir}\n{e}")
            exit(1)

    print("Starting video clipping process...")
    process_qa_files(args.qa_dir, args.match_dir,
                     args.raw_match_dir, args.output_dir, args.num_workers)
    print("Video clipping process finished.")
