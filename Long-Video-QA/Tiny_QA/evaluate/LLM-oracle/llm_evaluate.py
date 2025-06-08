import os
import sys  # Import sys
from openai import OpenAI
import time
import argparse  # Import argparse
import json      # Import json

# Add the parent directory (evaluate) to the Python path to allow relative import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Insert parent dir into sys.path

try:
    from evaluate_qa import BaseEvaluator  # Now this should work
except ImportError:
    # Fallback or more specific error handling if needed
    print("Error: Could not import BaseEvaluator from evaluate_qa.py. Ensure it's in the parent directory.")
    sys.exit(1)


class LLMEvaluator(BaseEvaluator):
    """
    Evaluator using LLM-based methods (like OpenAI API) for evaluation.
    This evaluator requires match commentary.
    """

    def __init__(self, qa_dir, match_data_folder, output_base_dir, test_name,
                 model_name="deepseek-chat", api_type="az", api_key=None, base_url=None, timeout=60.0):
        """
        Initialize the LLM evaluator.

        Args:
            qa_dir (str): Directory for QA files.
            match_data_folder (str): Directory for match data (used for comment extraction).
            output_base_dir (str): Base directory for output.
            test_name (str): Name of the test.
            model_name (str): Name of the LLM model to use.
            api_type (str): API type ('az' or 'mine') to determine default API key/URL source (env vars).
            api_key (str, optional): API key. Overrides env vars if provided.
            base_url (str, optional): API base URL. Overrides env vars if provided.
            timeout (float): API request timeout in seconds.
        """
        super().__init__(qa_dir, match_data_folder, output_base_dir, test_name)
        self.model_name = model_name
        self.timeout = timeout
        # Store match_data_folder specifically for comment extraction
        self.match_data_folder = match_data_folder
        self._match_comments_cache = {}  # Cache for comments

        # Configure API key and base URL based on api_type if not provided directly
        # Prioritize direct arguments, then environment variables
        resolved_api_key = api_key
        resolved_base_url = base_url

        if not resolved_api_key:
            if api_type == "az":
                resolved_api_key = "sk-FOvSp53sdWa96yE71d74F52a4d074d7f9128E7Bb4f113cE7"
                resolved_base_url = "https://az.gptplus5.com/v1"
            elif api_type == "mine":
                resolved_api_key = "sk-3db847df235b46d894d71b425c57c0f0"
                resolved_base_url = "https://api.deepseek.com"
            else:
                raise ValueError(
                    f"Unsupported api_type: {api_type}. Choose 'az' or 'mine'.")

        if not resolved_api_key:
            raise ValueError(
                "API Key is not set. Provide via --api_key argument or environment variable (e.g., AZURE_API_KEY or DEEPSEEK_API_KEY).")
        if not resolved_base_url:
            raise ValueError(
                "API Base URL is not set. Provide via --base_url argument or environment variable (e.g., AZURE_BASE_URL or DEEPSEEK_BASE_URL).")

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url

        print(f"Using LLM model: {self.model_name}")
        print(f"API Base URL: {self.base_url}")
        # Don't print the key itself for security
        print(f"API Key Source: {'Argument/Env Var'}")

    def _extract_all_comments(self, match_id):
        """Extract commentary for the first half (half=1) of the specified match ID, using caching."""
        if match_id in self._match_comments_cache:
            return self._match_comments_cache[match_id]

        match_folder_path = os.path.join(self.match_data_folder, match_id)
        if not os.path.isdir(match_folder_path):
            print(
                f"Error: Match folder not found for match_id {match_id} at {match_folder_path}")
            return ""  # Return empty string if folder not found

        # Find the first JSON file in the match directory
        json_file_path = None
        try:
            for filename in os.listdir(match_folder_path):
                if filename.endswith(".json"):
                    json_file_path = os.path.join(match_folder_path, filename)
                    break  # Use the first JSON file found
        except FileNotFoundError:
            print(
                f"Error: Match folder not found for match_id {match_id} at {match_folder_path}")
            return ""

        if not json_file_path or not os.path.exists(json_file_path):
            print(
                f"Error: Could not find match JSON file in folder: {match_folder_path}")
            return ""  # Return empty string if no JSON file found

        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            print(f"Error reading or parsing JSON file {json_file_path}: {e}")
            return ""

        comments_list = data.get("comments", [])
        result = []

        for comment in comments_list:
            half = comment.get("half")
            # Only process comments from the first half
            if half == 1:
                timestamp = comment.get("time_stamp", "")
                comments_text = comment.get("comments_text", "")
                index = comment.get("index", -1)
                # half_str is always "1st half" now, but keep for consistency if format changes
                half_str = "1st half"
                commentary_line = f"{half_str} - {timestamp} \"{comments_text}\" (Index: {index})"
                result.append(commentary_line)

        all_comments_text = "\n".join(result)
        # Cache the result (now only first half comments)
        self._match_comments_cache[match_id] = all_comments_text
        return all_comments_text

    def get_model_answer(self, match_id, qa):
        """
        Get the model's answer using an OpenAI compatible API, extracting comments as needed.

        Args:
            match_id (str): The match id indicating the specific match.
            qa (dict): The dictionary containing the QA pair details.

        Returns:
            str | None: The answer letter (A/B/C/D) chosen by the model, or None if failed.
        """
        # Extract details from the qa dictionary
        question = qa['question']
        choices = qa['choices']
        qa_type = qa['type']
        current_event_index = qa.get('current_event_index')

        # Get all comments for the match (uses caching)
        all_comments = self._extract_all_comments(match_id)
        if not all_comments:
            print(
                f"Warning: Could not extract comments for match {match_id}. Cannot answer question.")
            return None

        # Filter relevant commentary based on question type and current_event_index
        comments_lines = all_comments.split('\n')
        relevant_comments = ""

        if qa_type == "full" or current_event_index is None:  # Treat None index as 'full' for robustness
            relevant_comments = all_comments
            if qa_type != "full" and current_event_index is None:
                print(
                    f"Warning: Question type is '{qa_type}' but current_event_index is None. Using all comments for question index {qa.get('index', 'N/A')}.")
        else:  # 'past' or 'future' (filter comments up to current_event_index)
            relevant_comments_list = []
            for line in comments_lines:
                try:
                    # Extract index robustly
                    index_start = line.rfind("(Index: ")
                    if index_start != -1:
                        # Get content after "(Index: " and before ")"
                        index_str = line[index_start + 8:-1]
                        if index_str.isdigit():
                            comment_index = int(index_str)
                            # Provide comments up to and including the current event index
                            if comment_index <= current_event_index:
                                relevant_comments_list.append(line)
                except ValueError:
                    # This warning might be too verbose if many lines lack indices
                    # print(f"Warning: Could not parse index from line: {line}")
                    continue  # Skip lines where index parsing fails
            relevant_comments = "\n".join(relevant_comments_list)

        prompt = f"""You are analyzing a soccer match. Answer this multiple choice question based on the match events provided below.
ONLY RESPOND WITH A SINGLE LETTER (A/B/C/D) corresponding to your answer choice, DON"T MAKE ANY EXPLANATION.

Match Events:
{relevant_comments}

Question: {question}

Choices:
{chr(10).join(choices)}"""

        # It's recommended to handle potential API errors
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a soccer match analyst. Only respond with a single letter (A/B/C/D)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for deterministic answers
            )

            response = completion.choices[0].message.content.strip().upper()

            # Validate response format: Check if it's a single letter corresponding to a choice prefix
            valid_choice_prefixes = [choice.split(
                '.')[0] for choice in choices if '.' in choice]
            if len(response) == 1 and response in valid_choice_prefixes:
                return response
            else:
                print(
                    f"Warning: Received invalid response format: '{response}'. Expected one of {valid_choice_prefixes}. Match: {match_id}, Q-Index: {qa.get('index', 'N/A')}, Question: {question[:50]}...")
                return None  # Return None for invalid format

        except Exception as e:
            print(
                f"Error calling OpenAI API for match {match_id}, Q-Index: {qa.get('index', 'N/A')}: {e}")
            # Implement retry logic here if needed
            time.sleep(1)  # Simple backoff
            return None  # Return None if API call fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM-based evaluation for Tiny QA.")

    # --- Path Arguments ---
    parser.add_argument("--qa_dir", type=str, default="../../First_Half_QA",
                        help="Directory containing QA JSON files.")
    parser.add_argument("--match_data_folder", type=str, default="../../Subject_Data",
                        help="Root directory containing match data JSON files (for comment extraction).")
    parser.add_argument("--output_base_dir", type=str, default="evaluation_output",
                        help="Base directory to save evaluation results.")
    parser.add_argument("--test_name", type=str, default=f"llm_eval_{time.strftime('%Y%m%d_%H%M%S')}",
                        help="Unique name for this test run (used for output subfolder).")

    # --- LLM Configuration Arguments ---
    parser.add_argument("--model_name", type=str, default="deepseek-chat",
                        help="Name of the LLM model to use.")
    parser.add_argument("--api_type", type=str, default="az", choices=["az", "mine"],
                        help="API type ('az' or 'mine') to determine default API key/URL source (env vars).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (overrides environment variables like AZURE_API_KEY or DEEPSEEK_API_KEY if provided).")
    parser.add_argument("--base_url", type=str, default=None,
                        help="API base URL (overrides environment variables like AZURE_BASE_URL or DEEPSEEK_BASE_URL if provided).")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="API request timeout in seconds.")

    args = parser.parse_args()

    print("--- Initializing LLM Evaluator ---")
    # Instantiate the LLMEvaluator using parsed arguments
    try:
        evaluator = LLMEvaluator(
            qa_dir=args.qa_dir,
            match_data_folder=args.match_data_folder,
            output_base_dir=args.output_base_dir,
            test_name=args.test_name,
            model_name=args.model_name,
            api_type=args.api_type,
            api_key=args.api_key,     # Pass parsed api_key
            base_url=args.base_url,   # Pass parsed base_url
            timeout=args.timeout      # Pass parsed timeout
        )
    except ValueError as e:
        print(f"Error during initialization: {e}")
        exit(1)  # Exit if initialization fails (e.g., missing API key)

    print("\n--- Starting Evaluation Run ---")
    # Run the evaluation process
    evaluator.run_evaluation()

    print("\n--- Evaluation Script Finished ---")
