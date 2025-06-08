import json
from openai import OpenAI
import os
from tqdm import tqdm


def workflow(input_text, Instruction, follow_up_prompt=None, api_key="sk-FOvSp53sdWa96yE71d74F52a4d074d7f9128E7Bb4f113cE7"):
    client = OpenAI(api_key=api_key,
                    base_url="https://az.gptplus5.com/v1", timeout=2400.0)
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
        timeout=2400.0,
        stream=False
    )
    return completion.choices[0].message.content


def clean_json_block(text):
    if text.strip().startswith("```"):
        lines = text.strip().splitlines()
        return "\n".join(lines[1:-1])
    return text


def extract_event_comments(json_file_path, event_type):
    """Extract event comments with subject information"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    comments_list = data.get("comments", [])
    result = []

    for comment in comments_list:
        if comment.get("comments_type") != event_type:
            continue

        half = comment.get("half")
        if half not in [1, 2]:
            continue

        timestamp = comment.get("time_stamp", "")
        if not timestamp:
            continue

        comments_text = comment.get("comments_text", "")
        if not comments_text:
            continue

        subject = comment.get("subject", "Unknown")
        half_str = "1st half" if half == 1 else "2nd half"

        # Format event information with subject
        commentary_line = (f"{half_str} - {timestamp} "
                           f"Subject: {subject} "
                           f"Commentary: \"{comments_text}\" "
                           f"(Index: {comment.get('index', -1)})")
        result.append(commentary_line)

    return "\n".join(result)


def extract_first_half_event_comments(json_file_path, event_type):
    """Extract first half event comments with subject information"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    comments_list = data.get("comments", [])
    result = []

    for comment in comments_list:
        # Only get first half (half=1) comments
        if comment.get("half") != 1:
            continue

        if comment.get("comments_type") != event_type:
            continue

        timestamp = comment.get("time_stamp", "")
        if not timestamp:
            continue

        comments_text = comment.get("comments_text", "")
        if not comments_text:
            continue

        subject = comment.get("subject", "Unknown")

        # Format event information with subject
        commentary_line = (f"1st half - {timestamp} "
                           f"Subject: {subject} "
                           f"Commentary: \"{comments_text}\" "
                           f"(Index: {comment.get('index', -1)})")
        result.append(commentary_line)

    return "\n".join(result)


def generate_event_specific_prompt(event_type):
    """Generate event-specific prompt with subject-focused questions"""
    event_prompts = {
        "goal": """
- Subject is the scoring player
- Use verbs like 'scored', 'netted', 'put the ball in the net'
- Example questions:
  * "Who was the first player to score in the match?"
  * "Has [subject] scored multiple goals?"
  * "Which team's players have scored more goals?"
""",
        "corner": """
- Subject is the player taking the corner kick
- Use verbs like 'took', 'delivered', 'played in'
- Example questions:
  * "Which player has taken the most corners?"
  * "How many corners has [subject] taken?"
  * "Did [subject] take any more corners after this?"
""",
        "yellow card": """
- Subject is the player receiving the card
- Use verbs like 'received', 'was shown', 'was booked'
- Example questions:
  * "Who was the first player to receive a yellow card?"
  * "How many Liverpool players were booked?"
  * "Did [subject] receive another card later?"
"""
    }
    return event_prompts.get(event_type, "")


def retry_workflow(input_text, instruction, max_retries=3):
    """Retry workflow with error handling"""
    for attempt in range(max_retries):
        try:
            qa_pairs = workflow(input_text=input_text, Instruction=instruction)
            qa_pairs = clean_json_block(qa_pairs)
            # 验证 JSON 格式
            qa_list = json.loads(qa_pairs)

            # 验证每个 QA 对的格式
            for qa in qa_list:
                if not all(key in qa for key in ["question", "choices", "answer", "type",
                                                 "question_level", "related_event_index",
                                                 "current_event_index", "index"]):
                    raise ValueError("Missing required fields in QA pair")
                if qa["type"] not in ["past", "full", "future", "half"]:
                    raise ValueError("Invalid type value")
                if qa["question_level"] not in ["match", "team", "player"]:
                    raise ValueError("Invalid question_level value")
                if not isinstance(qa["related_event_index"], list):
                    raise ValueError("related_event_index must be a list")
                if not isinstance(qa["choices"], list) or len(qa["choices"]) < 2:
                    raise ValueError(
                        "choices must be a list with at least 2 options")
                if not qa["answer"] in ["A", "B", "C", "D"]:
                    raise ValueError("answer must be A, B, C, or D")

            return qa_list

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"尝试 {attempt + 1} 失败: {str(e)}")
                print("重试中...")
                continue
            else:
                print(f"所有尝试都失败了: {str(e)}")
                return None


def generate_qa_pairs(match_path, save_path, event_type):
    """Generate QA pairs for specific event type with subject focus"""
    event_comments = extract_event_comments(match_path, event_type)
    if not event_comments:
        print(f"No {event_type} events found in the match.")
        return

    instruction = f"""You are a football analyst creating QA pairs about {event_type} events.
Focus on the subjects (main actors) of each event when generating questions.

{generate_event_specific_prompt(event_type)}

Create questions that:
1. Directly reference the subjects of events
2. Compare different subjects' actions
3. Track subjects' involvement throughout the match
4. Consider team-level patterns of subjects' actions"""

    input_text = f"""Follow these requirements to create QA pairs about {event_type} events:
    
1. CONTEXT:
The provided text contains {event_type}-related commentary from a football match. For each event, you HAVE TO Generate three types of QA pairs at three different levels.

2. RESPONSE FORMAT:
Return ONLY a JSON array of QA pairs with this structure:
{{
    "question": "The question text",
    "choices": ["A. Choice A", "B. Choice B", "C. Choice C", "D. Choice D"],  # 2-4 choices
    "answer": "The correct choice letter (A/B/C/D)", # Note that the answer should only CONTAIN A SINGLE LETTER, without any other text
    "type": "past/full/future",
    "question_level": "match/team/player",
    "related_event_index": [0, 1, 2],  # List of relevant event indices in ascending order, and the question should be answered with all the related_events information
    "current_event_index": 5,  # Index for 'past' and 'future' types, None for 'full'
    "index": 0  # Question index
}}
You MUST NOT include any other text or explanation.

3. QA TYPES AND REQUIREMENTS:
A. Past Video QA (Generate for each {event_type} event)
- Questions about events up to the current event
- Must consider all events up to and including current_event_index
- related_event_index must all be <= current_event_index
- Examples for each level:
  Match Level:
    * "How many {event_type}s have occurred so far?"
    * "When did the most recent {event_type} happen?"
  Team Level:
    Event-specific examples:
    For goals:
    * "Which team has scored more goals up to now?"
    * "Has team X managed to score any goals yet?"
    
    For corners:
    * "Which team has taken more corners up to now?"
    * "Has team X won any corners yet?"
    
    For cards:
    * "Which team has received more yellow/red cards up to now?"
    * "Have any players from team X been booked yet?"

  Player Level:
    Event-specific examples:
    For goals:
    * "Who scored the last goal?"
    * "How many goals has player X scored so far?"
    
    For corners:
    * "Who took the last corner kick?"
    * "How many corners has player X taken so far?"
    
    For cards:
    * "Which player received the last card?"
    * "How many cards has player X received so far?"

B. Full Video QA (Generate for entire match)
- Questions about all {event_type} events in the match
- Must consider all events in the match
- related_event_index should include all relevant event indices
- Examples for each level:
  Match Level:
    * "How many total {event_type}s occurred in the match?"
    * "In which half were more {event_type}s awarded?"
  Team Level:
    * "Which team had the most {event_type}s in the match?"
    * "Did both teams have {event_type}s in both halves?"
  Player Level:
    * "Which player was involved in the most {event_type}s?"
    * "Who were all the players involved in {event_type}s?"

C. Future Prediction QA (Generate at different match points)
- Predictions about future events from current point
- current_event_index represents when the question is asked
- For correct predictions that happened, related_event_index must be > current_event_index
- For predictions that didn't happen, related_event_index should be empty list []
- Examples for each level:
  Match Level:
    * "Will there be more {event_type}s in this match?"
    * "How many more {event_type}s will occur?"
  Team Level:
    * "Which team is more likely to get the next {event_type}?"
    * "Will team X get another {event_type}?"
  Player Level:
    * "Which player is most likely to be involved in the next {event_type}?"
    * "Will player X be involved in another {event_type}?"

4. REQUIREMENTS:
- Adjust the number of QA pairs based on the frequency of the event_type in the match: generate fewer QA pairs if the event_type occurs rarely, and more QA pairs if it occurs frequently。
- Generate past questions for each {event_type} event
- Generate at least 3 full questions for each level
- Generate future questions at different match points
- Every question must be a multiple choice question with 2-4 choices
- All questions must require the provided event data to answer
- related_event_index must satisfy:
  * For 'past' questions: all indices <= current_event_index
  * For 'future' questions: indices > current_event_index for events that occurred, [] for predictions that didn't happen
  * For 'full' questions: all relevant event indices
- Include specific details from the commentary
- Ensure answers are clear and accurate

5. COMMENTARY:
{event_comments}

Generate the QA pairs in the specified JSON format only."""

    # 使用重试机制生成 QA 对
    qa_pairs = retry_workflow(input_text, instruction)

    if qa_pairs is None:
        print(f"Failed to generate valid QA pairs for {event_type}")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"Successfully generated QA pairs for {event_type}")
    except Exception as e:
        print(f"Error saving QA pairs for {event_type}: {e}")


def generate_first_half_qa_pairs(match_path, save_path, event_type):
    """Generate QA pairs for specific event type with subject focus (first half only)"""
    event_comments = extract_first_half_event_comments(match_path, event_type)
    if not event_comments:
        print(f"No {event_type} events found in the first half.")
        return

    instruction = f"""You are a football analyst creating QA pairs about {event_type} events in the first half.
Focus on the subjects (main actors) of each event when generating questions.

{generate_event_specific_prompt(event_type)}

Create questions that:
1. Directly reference the subjects of events
2. Compare different subjects' actions
3. Track subjects' involvement throughout the first half
4. Consider team-level patterns of subjects' actions"""

    input_text = f"""Follow these requirements to create QA pairs about {event_type} events in the first half:
    
1. CONTEXT:
The provided text contains {event_type}-related commentary from the first half of a football match. For each event, you HAVE TO Generate three types of QA pairs at three different levels.

2. RESPONSE FORMAT:
Return ONLY a JSON array of QA pairs with this structure:
{{
    "question": "The question text",
    "choices": ["A. Choice A", "B. Choice B", "C. Choice C", "D. Choice D"],  # 2-4 choices
    "answer": "The correct choice letter (A/B/C/D)", # Note that the answer should only CONTAIN A SINGLE LETTER, without any other text
    "type": "past/half/future",
    "question_level": "match/team/player",
    "related_event_index": [0, 1, 2],  # List of relevant event indices in ascending order
    "current_event_index": 5,  # Index for 'past' and 'future' types, None for 'half'
    "index": 0  # Question index
}}
You MUST NOT include any other text or explanation.

3. QA TYPES AND REQUIREMENTS:
A. Past Video QA (Generate for each {event_type} event in the first half)
- Questions about events up to the current event
- Must consider all events up to and including current_event_index
- related_event_index must all be <= current_event_index
- Examples for each level:
  Match Level:
    * "How many {event_type}s have occurred so far in the first half?"
    * "When did the most recent {event_type} happen?"
  Team Level:
    Event-specific examples:
    For goals:
    * "Which team has scored more goals in the first half up to now?"
    * "Has team X managed to score any goals yet?"
    
    For corners:
    * "Which team has taken more corners in the first half up to now?"
    * "Has team X won any corners yet?"
    
    For cards:
    * "Which team has received more cards in the first half up to now?"
    * "Have any players from team X been booked yet?"

  Player Level:
    Event-specific examples as before, but focused on first half

B. Half Video QA (Generate for entire first half)
- Questions about all {event_type} events in the first half
- Must consider all first half events
- related_event_index should include all relevant first half event indices
- Examples for each level:
  Match Level:
    * "How many total {event_type}s occurred in the first half?"
    * "At what point in the first half were most {event_type}s awarded?"
  Team Level:
    * "Which team had the most {event_type}s in the first half?"
    * "Did both teams have {event_type}s in the first half?"
  Player Level:
    * "Which player was involved in the most {event_type}s in the first half?"
    * "Who were all the players involved in first half {event_type}s?"

C. Future Prediction QA (Generate at different first half points)
- Predictions about future events from current point within the first half
- current_event_index represents when the question is asked
- For correct predictions that happened in first half, related_event_index must be > current_event_index
- For predictions that didn't happen, related_event_index should be empty list []
- Examples for each level:
  Match Level:
    * "Will there be more {event_type}s in the first half?"
    * "How many more {event_type}s will occur before half-time?"
  Team Level:
    * "Which team is more likely to get the next {event_type}?"
    * "Will team X get another {event_type} before half-time?"
  Player Level:
    * "Which player is most likely to be involved in the next {event_type}?"
    * "Will player X be involved in another {event_type} before half-time?"

4. REQUIREMENTS:
- Same as before but focused on first half events only
- Generate past questions for each first half {event_type} event
- Generate at least 3 half questions for each level
- Generate future questions at different first half points
- Every question must be a multiple choice question with 2-4 choices
- All questions must require the provided event data to answer
- related_event_index must satisfy first half event indices only

5. COMMENTARY:
{event_comments}

Generate the QA pairs in the specified JSON format only."""

    # Use retry mechanism to generate QA pairs
    qa_pairs = retry_workflow(input_text, instruction)

    if qa_pairs is None:
        print(f"Failed to generate valid QA pairs for {event_type}")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"Successfully generated first half QA pairs for {event_type}")
    except Exception as e:
        print(f"Error saving QA pairs for {event_type}: {e}")


def main():
    match_path = "../Subject_Data/2017-08-12_watford-fc-liverpool-fc/2017-08-12_Watford_3-3_Liverpool_AaZvBO5T.json"
    event_types = ["goal", "yellow card", "corner"]

    for event_type in tqdm(event_types, desc="Generating QA pairs"):
        save_path = f"./QA/{event_type.replace(' ', '_')}.json"
        generate_qa_pairs(match_path, save_path, event_type)

    for event_type in tqdm(event_types, desc="Generating first half QA pairs"):
        save_path = f"./First_Half_QA/{event_type.replace(' ', '_')}.json"
        generate_first_half_qa_pairs(match_path, save_path, event_type)


if __name__ == "__main__":
    main()
