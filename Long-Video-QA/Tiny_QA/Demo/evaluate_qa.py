import json
import os
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm


def load_all_qa_files(example_dir=r"QA"):
    """加载Example目录下的所有QA文件"""
    qa_files = {}
    for file in os.listdir(example_dir):
        if file.endswith('.json'):
            file_path = os.path.join(example_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                event_type = file.replace('.json', '')
                qa_files[event_type] = json.load(f)
    return qa_files


def extract_all_comments(json_file_path):
    """提取比赛中的所有commentary"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    comments_list = data.get("comments", [])
    result = []

    for comment in comments_list:
        half = comment.get("half")
        if half not in [1, 2]:
            continue

        timestamp = comment.get("time_stamp", "")
        if not timestamp:
            continue

        comments_text = comment.get("comments_text", "")
        if not comments_text:
            continue

        index = comment.get("index", -1)
        half_str = "1st half" if half == 1 else "2nd half"
        commentary_line = f"{half_str} - {timestamp} \"{comments_text}\" (Index: {index})"
        result.append(commentary_line)

    return "\n".join(result)


def get_model_answer(question, choices, qa_type, current_event_index, all_comments, model="deepseek-chat"):
    """获取模型的回答"""
    # 根据问题类型和current_event_index筛选相关commentary
    comments_lines = all_comments.split('\n')
    if qa_type == "full":
        relevant_comments = all_comments
    else:  # past或future类型
        relevant_comments = "\n".join([
            line for line in comments_lines
            if int(line[line.find("Index: ") + 7:-1]) <= current_event_index
        ])

    prompt = f"""You are analyzing a soccer match. Answer this multiple choice question based on the match events provided below.
ONLY respond with a single letter (A/B/C/D) corresponding to your answer choice.

Match Events:
{relevant_comments}

Question: {question}

Choices:
{chr(10).join(choices)}"""

    client = OpenAI(
        api_key="sk-FOvSp53sdWa96yE71d74F52a4d074d7f9128E7Bb4f113cE7",
        base_url="https://az.gptplus5.com/v1",
        timeout=60.0
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
                "content": "You are a soccer match analyst. Only respond with a single letter (A/B/C/D)."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    response = completion.choices[0].message.content.strip().upper()
    return response[0] if len(response) > 0 and response[0] in "ABCD" else None


def evaluate_qa_pairs(qa_files, model="deepseek-chat"):
    """评估所有QA对"""
    results = {
        'overall': {'correct': 0, 'total': 0},
        'by_event_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_question_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_question_level': defaultdict(lambda: {'correct': 0, 'total': 0})
    }

    # 用于存储模型回答的列表
    model_answers = []

    # 获取完整比赛commentary
    match_path = "../Subject_Data/2017-08-12_watford-fc-liverpool-fc/2017-08-12_Watford_3-3_Liverpool_AaZvBO5T.json"
    all_comments = extract_all_comments(match_path)

    # 对每种事件类型进行评估
    for event_type, qa_pairs in qa_files.items():
        print(f"\nEvaluating {event_type} questions...")

        for qa in tqdm(qa_pairs, desc=f"Processing {event_type}"):
            model_answer = get_model_answer(
                qa['question'],
                qa['choices'],
                qa['type'],
                qa.get('current_event_index'),
                all_comments,
                model
            )

            if model_answer is None:
                continue

            # 将模型回答和问题信息存储到列表中
            model_answers.append({
                "question": qa['question'],
                "choices": qa['choices'],
                "model_answer": model_answer,
                "right_answer": qa['answer'],
                "type": qa['type'],
                "question_level": qa['question_level'],
                "related_event_index": qa.get('related_event_index', []),
                "current_event_index": qa.get('current_event_index'),
                "index": qa['index']
            })

            correct = (model_answer == qa['answer'])

            # 更新总体统计
            results['overall']['total'] += 1
            if correct:
                results['overall']['correct'] += 1

            # 更新事件类型统计
            results['by_event_type'][event_type]['total'] += 1
            if correct:
                results['by_event_type'][event_type]['correct'] += 1

            # 更新问题类型统计
            results['by_question_type'][qa['type']]['total'] += 1
            if correct:
                results['by_question_type'][qa['type']]['correct'] += 1

            # 更新问题级别统计
            results['by_question_level'][qa['question_level']]['total'] += 1
            if correct:
                results['by_question_level'][qa['question_level']]['correct'] += 1

    # 将模型回答保存到JSON文件
    output_file = "model_answers.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(model_answers, f, indent=2, ensure_ascii=False)
    print(f"\nModel answers saved to {output_file}")

    return results


def print_results(results):
    """打印评估结果"""
    def calc_accuracy(stats):
        return (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0

    print("\n=== Overall Results ===")
    overall_acc = calc_accuracy(results['overall'])
    print(
        f"Total Accuracy: {overall_acc:.2f}% ({results['overall']['correct']}/{results['overall']['total']})")

    print("\n=== Results by Event Type ===")
    for event_type, stats in results['by_event_type'].items():
        acc = calc_accuracy(stats)
        print(
            f"{event_type}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    print("\n=== Results by Question Type ===")
    for qtype, stats in results['by_question_type'].items():
        acc = calc_accuracy(stats)
        print(
            f"{qtype.capitalize()}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    print("\n=== Results by Question Level ===")
    for level, stats in results['by_question_level'].items():
        acc = calc_accuracy(stats)
        print(
            f"{level.replace('_', ' ').title()}: {acc:.2f}% ({stats['correct']}/{stats['total']})")


def main():
    # 加载所有QA文件
    qa_files = load_all_qa_files()
    if not qa_files:
        print("No QA files found in Demo directory!")
        return

    print(f"Found {len(qa_files)} QA files to evaluate")

    # 评估
    results = evaluate_qa_pairs(qa_files)

    # 打印结果
    print_results(results)

    # 保存结果
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
