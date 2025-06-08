import json
import os
import random
from abc import ABC, abstractmethod  # Import Abstract Base Classes
from collections import defaultdict
from tqdm import tqdm
import time


class BaseEvaluator(ABC):
    """
    Base class for evaluators, defining the framework for the evaluation process.
    Subclasses need to implement the get_model_answer method to provide specific model response logic.
    """

    def __init__(self, qa_dir, match_data_folder, output_base_dir, test_name, shuffle_choices=False, shuffle_times=4):
        """
        Initialize the evaluator.

        Args:
            qa_dir (str): Absolute or relative path to the directory containing QA JSON files.
            match_data_folder (str): Absolute or relative path to the root directory containing match JSON files.
                                     (May be needed by subclasses for context extraction).
            output_base_dir (str): Absolute or relative path to the base directory for saving evaluation results.
            test_name (str): Name of this test run, used to create an output subfolder.
            shuffle_choices (bool): Whether to shuffle the choices during evaluation.
            shuffle_times (int): Number of times to shuffle the choices.
        """
        # Use the provided paths directly
        self.qa_dir = qa_dir
        # Store match_data_folder for potential use in subclasses
        self.match_data_folder = match_data_folder
        self.output_base_dir = output_base_dir
        self.test_name = test_name
        # Output directory is still created based on output_base_dir and test_name
        self.output_dir = os.path.join(self.output_base_dir, self.test_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"QA directory: {self.qa_dir}")
        # Keep print for info, as subclasses might use this path
        print(f"Match data folder: {self.match_data_folder}")
        print(f"Output directory: {self.output_dir}")
        self.shuffle_choices = shuffle_choices
        self.shuffle_times = shuffle_times

    def load_qa_files(self):
        """加载所有QA文件并直接返回扁平化的QA列表"""
        flat_qa_list = []
        if not os.path.isdir(self.qa_dir):
            print(f"Error: QA directory not found: {self.qa_dir}")
            return flat_qa_list

        for file in os.listdir(self.qa_dir):
            if file.endswith('.json'):
                match_id = file.replace('.json', '')
                file_path = os.path.join(self.qa_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)
                        for qa in qa_data:
                            if isinstance(qa, dict) and 'event_type' in qa:
                                flat_qa_list.append({
                                    "match_id": match_id,
                                    "event_type": qa['event_type'],
                                    "qa": qa
                                })
                except Exception as e:
                    print(f"Error loading QA file {file_path}: {e}")
        return flat_qa_list

    @abstractmethod
    def get_model_answer(self, match_id, qa):
        """模型回答接口，子类实现"""
        pass

    def evaluate_single_qa(self, match_id, event_type, qa):
        """评估单个QA（非shuffle）"""
        model_answer = self.get_model_answer(match_id, qa)
        return {
            "match_id": match_id,
            "event_type": event_type,
            "question": qa['question'],
            "choices": qa['choices'],
            "model_answer": model_answer,
            "right_answer": qa['answer'],
            "type": qa['type'],
            "question_level": qa['question_level'],
            "related_event_index": qa.get('related_event_index', []),
            "current_event_index": qa.get('current_event_index'),
            "index": qa['index']
        }

    def evaluate_single_qa_with_shuffle(self, match_id, event_type, qa):
        """评估单个QA（shuffle）"""
        model_answers = []
        model_answer_contents = []
        correct_contents = []
        correct_labels = []
        correct_count = 0

        for seed in range(self.shuffle_times):
            qa_shuffled = qa.copy()
            choices = qa['choices'][:]
            random.Random(seed).shuffle(choices)
            # 重新分配A/B/C/D
            new_choices = []
            label_map = {}
            content_map = {}
            for idx, c in enumerate(choices):
                label = chr(ord('A') + idx)
                label_map[c[0]] = label
                content = c[3:]
                new_choices.append(f"{label}. {content}")
                content_map[label] = content
            qa_shuffled['choices'] = new_choices
            qa_shuffled['answer'] = label_map[qa['answer']]
            qa_shuffled['answer_content'] = content_map[qa_shuffled['answer']]
            model_answer = self.get_model_answer(match_id, qa_shuffled)
            model_answers.append(model_answer)
            model_answer_content = content_map.get(model_answer, None)
            model_answer_contents.append(model_answer_content)
            correct_contents.append(qa_shuffled['answer_content'])
            correct_labels.append(qa_shuffled['answer'])
        return {
            "match_id": match_id,
            "event_type": event_type,
            "question": qa['question'],
            "index": qa['index'],
            "type": qa['type'],
            "question_level": qa['question_level'],
            "shuffle_model_answers": model_answers,
            "shuffle_model_answer_contents": model_answer_contents,
            "shuffle_correct_contents": correct_contents,
            "shuffle_correct_labels": correct_labels
        }

    def evaluate_all_qa(self, qa_list):
        """评估所有QA，同时维护临时统计用于显示实时准确率"""
        model_answers = []

        # 临时统计，仅用于显示
        temp_stats = {'correct': 0, 'total': 0}
        temp_shuffle_vote = {'correct': 0, 'total': 0}
        temp_shuffle_all = {'correct': 0, 'total': 0,
                            'seed_counts': [0] * self.shuffle_times}

        with tqdm(total=len(qa_list), desc="Evaluating QA pairs") as pbar:
            for item in qa_list:
                match_id = item["match_id"]
                event_type = item["event_type"]
                qa = item["qa"]

                pbar.set_description(
                    f"Match {match_id} | {event_type} | Q{qa.get('index', 'N/A')}"
                )

                if not self.shuffle_choices:
                    # 非shuffle模式
                    answer_data = self.evaluate_single_qa(
                        match_id, event_type, qa)
                    correct = (answer_data["model_answer"]
                               == answer_data["right_answer"])

                    # 更新临时统计
                    temp_stats['total'] += 1
                    if correct:
                        temp_stats['correct'] += 1

                    model_answers.append(answer_data)

                    # 显示实时准确率
                    acc = (temp_stats['correct'] / temp_stats['total']
                           ) * 100 if temp_stats['total'] > 0 else 0
                    pbar.set_postfix({
                        'Accuracy': f'{acc:.2f}%',
                        'Correct': f"{temp_stats['correct']}/{temp_stats['total']}"
                    })
                else:
                    # shuffle模式
                    shuffle_data = self.evaluate_single_qa_with_shuffle(
                        match_id, event_type, qa)

                    # 计算投票结果
                    from collections import Counter
                    vote_content = Counter(
                        shuffle_data["shuffle_model_answer_contents"]).most_common(1)[0][0]
                    vote_correct = (
                        vote_content == shuffle_data["shuffle_correct_contents"][-1])

                    # 更新Vote统计
                    temp_shuffle_vote['total'] += 1
                    if vote_correct:
                        temp_shuffle_vote['correct'] += 1

                    # 更新All统计
                    correct_count = 0
                    for i, (model_content, correct_content) in enumerate(
                        zip(shuffle_data["shuffle_model_answer_contents"],
                            shuffle_data["shuffle_correct_contents"])):

                        temp_shuffle_all['total'] += 1
                        if model_content == correct_content:
                            correct_count += 1
                            temp_shuffle_all['correct'] += 1

                    # 记录每个seed的正确率
                    for i in range(min(self.shuffle_times, len(shuffle_data["shuffle_model_answers"]))):
                        if i < len(shuffle_data["shuffle_model_answer_contents"]) and \
                           i < len(shuffle_data["shuffle_correct_contents"]):
                            if shuffle_data["shuffle_model_answer_contents"][i] == \
                               shuffle_data["shuffle_correct_contents"][i]:
                                temp_shuffle_all['seed_counts'][i] += 1

                    model_answers.append(shuffle_data)

                    # 显示实时准确率（包括投票和所有轮次）
                    vote_acc = (temp_shuffle_vote['correct'] / temp_shuffle_vote['total']) * 100 \
                        if temp_shuffle_vote['total'] > 0 else 0
                    all_acc = (temp_shuffle_all['correct'] / temp_shuffle_all['total']) * 100 \
                        if temp_shuffle_all['total'] > 0 else 0

                    # 计算每个seed的准确率
                    seed_accs = []
                    for i in range(self.shuffle_times):
                        seed_acc = (temp_shuffle_all['seed_counts'][i] / temp_shuffle_vote['total']) * 100 \
                            if temp_shuffle_vote['total'] > 0 else 0
                        seed_accs.append(f'{seed_acc:.1f}')

                    pbar.set_postfix({
                        'VoteAcc': f'{vote_acc:.2f}%',
                        'AllAcc': f'{all_acc:.2f}%',
                        'Seeds': '/'.join(seed_accs)
                    })

                pbar.update(1)

        return model_answers

    def calculate_statistics(self, model_answers):
        """根据模型答案计算各种统计数据"""
        # 初始化结果字典
        results = {
            'overall': {'correct': 0, 'total': 0},
            'by_event_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'by_question_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'by_question_level': defaultdict(lambda: {'correct': 0, 'total': 0})
        }

        # 如果是shuffle模式，再创建两个结果字典
        if self.shuffle_choices:
            results_vote = {
                'overall': {'correct': 0, 'total': 0},
                'by_event_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
                'by_question_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
                'by_question_level': defaultdict(lambda: {'correct': 0, 'total': 0})
            }
            results_all = {
                'overall': {'correct': 0, 'total': 0},
                'by_event_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
                'by_question_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
                'by_question_level': defaultdict(lambda: {'correct': 0, 'total': 0})
            }

        # 统计
        print("Calculating statistics...")
        for answer in tqdm(model_answers, desc="Processing statistics"):
            event_type = answer["event_type"]
            qa_type = answer["type"]
            question_level = answer["question_level"]

            if not self.shuffle_choices:
                # 非shuffle统计
                correct = (answer["model_answer"] == answer["right_answer"])
                results['overall']['total'] += 1
                results['by_event_type'][event_type]['total'] += 1
                results['by_question_type'][qa_type]['total'] += 1
                results['by_question_level'][question_level]['total'] += 1
                if correct:
                    results['overall']['correct'] += 1
                    results['by_event_type'][event_type]['correct'] += 1
                    results['by_question_type'][qa_type]['correct'] += 1
                    results['by_question_level'][question_level]['correct'] += 1
            else:
                # shuffle统计
                from collections import Counter

                # 投票统计
                vote_content = Counter(
                    answer["shuffle_model_answer_contents"]).most_common(1)[0][0]
                vote_correct = (
                    vote_content == answer["shuffle_correct_contents"][-1])

                results_vote['overall']['total'] += 1
                results_vote['by_event_type'][event_type]['total'] += 1
                results_vote['by_question_type'][qa_type]['total'] += 1
                results_vote['by_question_level'][question_level]['total'] += 1

                if vote_correct:
                    results_vote['overall']['correct'] += 1
                    results_vote['by_event_type'][event_type]['correct'] += 1
                    results_vote['by_question_type'][qa_type]['correct'] += 1
                    results_vote['by_question_level'][question_level]['correct'] += 1

                # 所有shuffle轮次统计
                for i, (model_content, correct_content) in enumerate(
                        zip(answer["shuffle_model_answer_contents"], answer["shuffle_correct_contents"])):

                    results_all['overall']['total'] += 1
                    results_all['by_event_type'][event_type]['total'] += 1
                    results_all['by_question_type'][qa_type]['total'] += 1
                    results_all['by_question_level'][question_level]['total'] += 1

                    if model_content == correct_content:
                        results_all['overall']['correct'] += 1
                        results_all['by_event_type'][event_type]['correct'] += 1
                        results_all['by_question_type'][qa_type]['correct'] += 1
                        results_all['by_question_level'][question_level]['correct'] += 1

        if self.shuffle_choices:
            return {'vote': results_vote, 'all': results_all}
        else:
            return results

    def save_json(self, data, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"- Saved to: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    def save_statistics_txt(self, results, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                def calc_accuracy(stats):
                    if isinstance(stats, dict) and 'correct' in stats and 'total' in stats and stats['total'] > 0:
                        return (stats['correct'] / stats['total'] * 100)
                    return 0.0

                def get_counts(stats):
                    if isinstance(stats, dict) and 'correct' in stats and 'total' in stats:
                        return f"({stats.get('correct', 0)}/{stats.get('total', 0)})"
                    return "(0/0)"

                f.write(f"=== Evaluation Statistics: {self.test_name} ===\n\n")

                if self.shuffle_choices and isinstance(results, dict) and 'vote' in results and 'all' in results:
                    for mode in ['vote', 'all']:
                        mode_name = "Vote (投票)" if mode == 'vote' else "All (所有轮次)"
                        f.write(f"--- {mode_name} ---\n")
                        mode_results = results[mode]

                        overall_stats = mode_results.get('overall', {})
                        overall_acc = calc_accuracy(overall_stats)
                        f.write(
                            f"Overall Accuracy: {overall_acc:.2f}% {get_counts(overall_stats)}\n\n")

                        f.write("--- Results by Event Type ---\n")
                        event_type_results = mode_results.get(
                            'by_event_type', {})
                        if event_type_results:
                            for event_key in sorted(event_type_results.keys()):
                                stats = event_type_results[event_key]
                                acc = calc_accuracy(stats)
                                f.write(
                                    f"{event_key}: {acc:.2f}% {get_counts(stats)}\n")
                        else:
                            f.write("No data available.\n")
                        f.write("\n")

                        f.write("--- Results by Question Type ---\n")
                        q_type_results = mode_results.get(
                            'by_question_type', {})
                        if q_type_results:
                            for qtype in sorted(q_type_results.keys()):
                                stats = q_type_results[qtype]
                                acc = calc_accuracy(stats)
                                f.write(
                                    f"{qtype.capitalize()}: {acc:.2f}% {get_counts(stats)}\n")
                        else:
                            f.write("No data available.\n")
                        f.write("\n")

                        f.write("--- Results by Question Level ---\n")
                        q_level_results = mode_results.get(
                            'by_question_level', {})
                        if q_level_results:
                            for level in sorted(q_level_results.keys()):
                                stats = q_level_results[level]
                                acc = calc_accuracy(stats)
                                f.write(
                                    f"{level.replace('_', ' ').title()}: {acc:.2f}% {get_counts(stats)}\n")
                        else:
                            f.write("No data available.\n")
                        f.write("\n")
                else:
                    overall_stats = results.get('overall', {})
                    overall_acc = calc_accuracy(overall_stats)
                    f.write(
                        f"Overall Accuracy: {overall_acc:.2f}% {get_counts(overall_stats)}\n\n")

                    f.write("--- Results by Event Type ---\n")
                    event_type_results = results.get('by_event_type', {})
                    if event_type_results:
                        for event_key in sorted(event_type_results.keys()):
                            stats = event_type_results[event_key]
                            acc = calc_accuracy(stats)
                            f.write(
                                f"{event_key}: {acc:.2f}% {get_counts(stats)}\n")
                    else:
                        f.write("No data available.\n")
                    f.write("\n")

                    f.write("--- Results by Question Type ---\n")
                    q_type_results = results.get('by_question_type', {})
                    if q_type_results:
                        for qtype in sorted(q_type_results.keys()):
                            stats = q_type_results[qtype]
                            acc = calc_accuracy(stats)
                            f.write(
                                f"{qtype.capitalize()}: {acc:.2f}% {get_counts(stats)}\n")
                    else:
                        f.write("No data available.\n")
                    f.write("\n")

                    f.write("--- Results by Question Level ---\n")
                    q_level_results = results.get('by_question_level', {})
                    if q_level_results:
                        for level in sorted(q_level_results.keys()):
                            stats = q_level_results[level]
                            acc = calc_accuracy(stats)
                            f.write(
                                f"{level.replace('_', ' ').title()}: {acc:.2f}% {get_counts(stats)}\n")
                    else:
                        f.write("No data available.\n")

            print(f"- Statistics summary saved to: {filename}")
        except IOError as e:
            print(f"Error saving statistics TXT: {e}")

    def run_evaluation(self):
        """主评估流程"""
        print("Starting evaluation...")
        qa_list = self.load_qa_files()
        if not qa_list:
            print(f"No QA files found in '{self.qa_dir}'. Aborting.")
            return
        print(f"Found {len(qa_list)} QA pairs to evaluate.")

        # 第一阶段：评估，获取模型答案
        print("Evaluating QA pairs...")
        model_answers = self.evaluate_all_qa(qa_list)

        # 第二阶段：统计，计算各种指标
        print("Calculating statistics...")
        results = self.calculate_statistics(model_answers)

        # 第三阶段：保存结果
        print("Saving results...")
        self.save_json(results, os.path.join(
            self.output_dir, "evaluation_results.json"))
        self.save_statistics_txt(results, os.path.join(
            self.output_dir, "statistics.txt"))

        if self.shuffle_choices:
            self.save_json(model_answers, os.path.join(
                self.output_dir, "shuffle_results.json"))
        else:
            self.save_json(model_answers, os.path.join(
                self.output_dir, "model_answers.json"))

        print("\nEvaluation complete.")
