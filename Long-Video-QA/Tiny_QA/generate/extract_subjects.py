import json
import os
from openai import OpenAI
from tqdm import tqdm


def get_subject_extraction_prompt(text):
    """Generate prompt based on event type"""
    prompt = f"""Extract ONLY the subject name(s) from the following football event commentary. THINK CAREFULLY and extract the CORRECT player name.
    
    Rules for different event types:
    - For goals: Extract ONLY the scoring player (NOT ASSISTS!)
    - For corners: Extract ONLY the player taking the corner
    - For cards: Extract ONLY the player receiving the card
    - For penalties: Extract ONLY the player taking the penalty
    - For substitutions: Extract ONLY the two players (incoming and outgoing) separated by '|'
    
    Examples:
    1. Goal example:
    Commentary: "Jose Holebas takes over corner duties and expertly finds Stefano Okaka (Watford), who jumps highest and coolly plants a bullet header into the back of the net. The score is now 1:0."
    Answer: Stefano Okaka

    2. Corner example:
    Commentary: "Corner taken by Kevin De Bruyne from the right by-line."
    Answer: Kevin De Bruyne

    3. Card example:
    Commentary: "Yellow card shown to Paul Pogba for a bad foul."
    Answer: Paul Pogba
    
    4. Penalty example:
    Commentary: "Goal! Penalty scored by Mohamed Salah into the bottom right corner."
    Answer: Mohamed Salah

    5. Substitution example:
    Commentary: "Substitution for Arsenal. Gabriel Jesus replaces Eddie Nketiah."
    Answer: Gabriel Jesus|Eddie Nketiah

    Commentary: {text}
    Answer:"""
    return prompt


def extract_subject(text, event_type, api_key="sk-FOvSp53sdWa96yE71d74F52a4d074d7f9128E7Bb4f113cE7"):
    client = OpenAI(
        api_key=api_key,
        base_url="https://az.gptplus5.com/v1",
        timeout=2400.0
    )

    system_prompt = "You are a football event commentary analysis assistant. Extract ONLY player names, nothing else."
    prompt = get_subject_extraction_prompt(text)

    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    try:
        result = completion.choices[0].message.content.strip()
        # print(f"Extracted subject: {result}")
        # 处理不同类型的结果
        if event_type == "substitution":
            players = result.split("|")
            if len(players) == 2:
                return {"subject": players}
            return {"subject": "Unknown"}
        else:
            return {"subject": result if result else "Unknown"}
    except:
        return {"subject": "Unknown"}


def extract_subject_with_retry(text, event_type, max_retries=3, pbar=None):
    """Extract subject with retry logic for handling LLM response format errors"""
    for attempt in range(max_retries):
        try:
            subject_info = extract_subject(text, event_type)
            # Validate the response format
            if isinstance(subject_info, dict) and "subject" in subject_info:
                subject = subject_info["subject"]
                # Validate substitution format
                if event_type == "substitution" and not isinstance(subject, list):
                    raise ValueError(
                        f"Substitution subject should be a list, but get {subject_info}")
                # Validate other events format
                elif event_type != "substitution" and not isinstance(subject, str):
                    raise ValueError("Subject should be a string")
                return subject_info
            raise ValueError("Invalid response format")
        except Exception as e:
            if pbar:
                pbar.write(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return {"subject": "Unknown"}
            continue


def process_file(input_file, output_dir, pbar=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each comment
    comments = data.get("comments", [])
    event_comments = [(i, c) for i, c in enumerate(comments)
                      if c.get("comments_type") in ["goal", "corner", "yellow card", "red card", "penalty", "substitution"]]

    for i, comment in tqdm(event_comments, desc=f"Processing {os.path.basename(input_file)}",
                           leave=False, position=1):
        event_type = comment.get("comments_type")
        subject_info = extract_subject_with_retry(
            comment["comments_text"], event_type, pbar=pbar)
        comments[i]["subject"] = subject_info["subject"]

    # Save processed file
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # Define paths
    data_dir = "./Data"
    output_dir = "./Subject_Data"

    # Get all JSON files
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                all_files.append((root, file))

    # Process all files with progress bar
    with tqdm(total=len(all_files), desc="Processing files", position=0) as pbar:
        for root, file in all_files:
            input_file = os.path.join(root, file)
            relative_path = os.path.relpath(root, data_dir)
            current_output_dir = os.path.join(output_dir, relative_path)
            process_file(input_file, current_output_dir, pbar)
            pbar.update(1)


if __name__ == "__main__":
    main()
