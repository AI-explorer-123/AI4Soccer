import os
import json
from tqdm import tqdm
from Demo.generate_event_qa import generate_qa_pairs, generate_first_half_qa_pairs

def process_all_matches(subject_data_dir, output_dir):
    # 遍历Subject_Data目录下的所有比赛文件夹
    matches = [d for d in os.listdir(subject_data_dir) if os.path.isdir(
        os.path.join(subject_data_dir, d))]
    event_types = ["goal", "yellow card", "corner"]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for match in tqdm(matches, desc="Processing matches"):
        match_dir = os.path.join(subject_data_dir, match)
        json_files = [f for f in os.listdir(match_dir) if f.endswith('.json')]

        if not json_files:
            continue

        # 获取比赛JSON文件路径
        match_json = os.path.join(match_dir, json_files[0])

        # 为每种事件类型生成QA对
        match_qa_pairs = []
        for event_type in event_types:
            output_file = os.path.join(
                output_dir, f"{match}_{event_type.replace(' ', '_')}.json")

            try:
                # generate_qa_pairs(match_json, output_file, event_type)
                generate_first_half_qa_pairs(match_json, output_file, event_type)

                # 读取生成的QA对并添加event_type
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        qa_pairs = json.load(f)
                        for qa in qa_pairs:
                            qa['event_type'] = event_type
                        match_qa_pairs.extend(qa_pairs)

                    # 删除临时文件
                    os.remove(output_file)

            except Exception as e:
                print(f"Error processing {match} for {event_type}: {e}")
                continue

        # 保存包含所有事件类型的QA对
        if match_qa_pairs:
            combined_output = os.path.join(output_dir, f"{match}.json")
            try:
                with open(combined_output, 'w', encoding='utf-8') as f:
                    json.dump(match_qa_pairs, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error saving combined QA pairs for {match}: {e}")


def main():
    subject_data_dir = "./Subject_Data"
    output_dir = "./First_Half_QA"
    process_all_matches(subject_data_dir, output_dir)


if __name__ == "__main__":
    main()
