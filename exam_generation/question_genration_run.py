import re
from question_generation_prompt import *
import os
import random
import jsonlines
from tqdm import tqdm
import asyncio
import argparse
from script.gpt_usage import chunks, OpenAIChat_Embed
import csv
from script.calculate_length import num_tokens_from_messages

def convert_txt2jsonl(example_data, is_morality=True):
    if not is_morality:
        example_data = chunks(example_data, 11)  # 11 means there are 11 lines for each example in the txt file
        new_format_data = []
        for item in example_data:
            # print(item)
            principle = item[0].strip().split('法律:')[1].replace("\u3000", "  ")
            question = item[1].strip().split('题干:')[1]
            options = [option.strip() for option in item[3:7]]
            analysis = item[7].strip().split("选项分析:")[1]
            answer = item[8].strip().split("答案:")[1]
            label = item[9].strip().split("label:")[1]
            if label == "1":
                action = "设计一个法律情境"
            else:
                action = "针对给定法律中的某个概念"
            new_data = {"principle": principle, "question": question, "options": options,
                        "analysis": analysis, "answer": answer, "action": action}
            new_format_data.append(new_data)
    else:
        example_data = chunks(example_data, 11)
        new_format_data = []
        for item in example_data:
            principle = item[0].strip().split('道德准则:')[1].replace("\u3000", "  ")
            question = item[1].strip().split('题干:')[1]
            options = [option.strip() for option in item[3:7]]
            analysis = item[7].strip().split("选项分析:")[1]
            answer = item[8].strip().split("答案:")[1]
            label = item[9].strip().split("label:")[1]
            if label == "1":
                action = "设计一个道德情境"
            else:
                action = "针对给定道德准则中的某个概念"
            new_data = {"principle": principle, "question": question, "options": options,
                        "analysis": analysis, "answer": answer, "action": action}
            new_format_data.append(new_data)
    return new_format_data


def generate_questions(input_items, examples, example_template, exam_mode, mode_type, save_path):
    messages_all = []  # store all messages sent to question generation model
    example_use_all = [] # store all question examples
    input_principle_all = [] # store all input principle, which are used as the context for generating the questions
    input_location_all = [] # store the location related to each principle

    for i, input_item in tqdm(enumerate(input_items), total=len(input_items)):
        # we randomly sample an example when generating questions
        example_ptr = random.randint(0, len(examples) - 1)
        example_use = examples[example_ptr]

        # example setting
        # example option
        choices = ["A", "B", "C", "D"]
        example_options = '\n'.join([choice+'. '+ example_use[choice] for choice in choices])
        # example stem + option
        example_question = example_use["question"]

        input_example = example_template.replace("{example_principle}", example_use["principle"])\
            .replace("{question}", example_question).replace("{options}", example_options)\
            .replace("{analysis}", example_use["analysis"])\
            .replace("{answer}", example_use["answer"])

        # input prompt setting
        if "情境" in example_use["action"]:
            action_type = "情境"
        else:
            action_type = "概念"
        input_principle = input_item["principle"]
        input_location = input_item["location"]
        user_prompt_ = user_prompt.replace("{input_principle}", input_principle)\
            .replace("{example}", input_example) \
            .replace("{action}", example_use["action"])\
            .replace("{action_type}",action_type)\
            .replace("{mode_type}", mode_type)
        if "law" in exam_mode:
            if input_location!="":
                # add the location information
                system_prompt_ = system_prompt.replace("{additional_principle}", "13.生成的题干中必须包含地点<{}>".format(input_location))
            else:
                system_prompt_ = system_prompt.replace("{additional_principle}", "")
        elif "morality" in exam_mode:
            system_prompt_ = system_prompt.replace("{additional_principle}",
                                                   "13.生成的题干中必须包含公司名称(或者协会名称)<{}>".format(input_location))

        system_prompt_ = system_prompt_.replace("{mode_type}", mode_type)
        messages = [
            {"role": "system", "content": system_prompt_},
            {"role": "user", "content": user_prompt_}
        ]
        # print(system_prompt)
        # print(user_prompt_)
        messages_all.append(messages)
        example_use_all.append(example_use)
        input_principle_all.append(input_principle)
        input_location_all.append(input_location)
    tokens = num_tokens_from_messages(messages_all)
    openai_model = OpenAIChat_Embed(chat_model_name=args.generation_model,
                                    temperature=args.temperature,
                                    )
    # generate the response based on the input message
    message_batch_num = min(5, len(messages_all))
    message_batch = chunks(messages_all, message_batch_num)
    responses_all = []
    for cnt, messages in tqdm(enumerate(message_batch), total=len(messages_all) / message_batch_num):
        cur_responses = asyncio.run(openai_model.async_run(
            messages_list=messages,
            mode="chat"
        ))
        responses_all.extend(cur_responses)

    # extract stem, options, answer and analysis from the response
    out_questions = []
    for idx, response in enumerate(responses_all):
        stem_rule = r"题干:([\s\S]*?)备选项:"
        options_rule = r'备选项:([\s\S]*?)选项分析:'
        analysis_rule = r'选项分析:([\s\S]*?)答案:'
        # answer_rule = r'答案:([\s\S]*?)'

        option_a_rule = r'A\.([\s\S]*?)B\.'
        option_b_rule = r'B\.([\s\S]*?)C\.'
        option_c_rule = r'C\.([\s\S]*?)D\.'
        option_d_rule = r'D\.([\s\S]*?)'
        # option_a_rule = r'A.([\s\S]*?)B.'
        # option_b_rule = r'B.([\s\S]*?)C.'
        # option_c_rule = r'C.([\s\S]*?)D.'
        # option_d_rule = r'D.([\s\S]*?)'
        response = response.replace("：", ":")
        try:
            stem = re.findall(stem_rule, response)[0].strip()
            options = re.findall(options_rule, response)[0].strip()
            analysis = re.findall(analysis_rule, response)[0].strip()
            answer = response.split("答案:")[1].strip()

            option_a = re.findall(option_a_rule, options)[0].strip()
            option_b = re.findall(option_b_rule, options)[0].strip()
            option_c = re.findall(option_c_rule, options)[0].strip()
            option_d = options.split("D.")[1].strip()
            # stem = re.findall(stem_rule, response)[0]
            # options = re.findall(options_rule, response)[0]
            # analysis = re.findall(analysis_rule, response)[0]
            # answer = re.findall(answer_rule, response)[0]
            #
            # option_a = re.findall(options, option_a_rule)[0]
            # option_b = re.findall(options, option_b_rule)[0]
            # option_c = re.findall(options, option_c_rule)[0]
            # option_d = re.findall(options, option_d_rule)[0]
            cur_data = {"question": stem, "A": option_a, "B": option_b, "C": option_c, "D": option_d,
                        "answer": answer, "analysis": analysis, "principle": input_principle_all[idx],
                        "location": input_location_all[idx]}
            out_questions.append(cur_data)
        except:
            print("there are something wrong with generated question {}".format(response))
    header_list = ["question", "A", "B", "C", "D", "answer", "analysis","principle", "location"]
    with open(save_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, header_list)
        writer.writeheader()
        writer.writerows(out_questions)

def main(args):

    if args.exam_mode=="law":
        mode_type = "法律"
    else:
        mode_type = "道德准则"

    input_example_template = "{mode_type}: {example_principle}\n\n" + "题干: {question}\n\n" + "备选项: {options}\n\n" + \
                             "选项分析: {analysis}\n\n" + "答案: {answer}"
    input_example_template = input_example_template.replace("{mode_type}", mode_type)
    # load examples
    example_path = args.input_example_path.replace("{exam_mode}", args.exam_mode if "law" in args.exam_mode else "morality")
    with open(example_path, encoding="utf-8", mode="r") as f:
        reader = csv.DictReader(f)
        examples = [obj for obj in reader]

    # load selected principles for question generation
    input_data_for_question_generation_path = args.input_data_for_question_generation_path.replace("{exam_mode}", args.exam_mode)
    with jsonlines.open(input_data_for_question_generation_path, 'r') as reader:
        input_data_for_question_generation = [obj for obj in reader]
    # here we select the first 500 principles
    input_data_for_question_generation = input_data_for_question_generation[:500]
    save_path = args.generated_questions_save_path.replace("{exam_mode}", args.exam_mode)
    if not os.path.exists('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]))

    generate_questions(input_data_for_question_generation, examples, input_example_template,args.exam_mode, mode_type, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation_model",
        type=str,
        default="gpt-4-0613"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--exam_mode",
        type=str,
        default="law",
        help="choose the mode from [law, professional_morality]"
    )
    parser.add_argument(
        "--input_example_path",
        type=str,
        default="../data/example_data/examples_{exam_mode}.csv",
        help="the file path which stores the principles used for question generation"
    )

    parser.add_argument(
        "--input_data_for_question_generation_path",
        type=str,
        default="../data/sampled_data_for_question_generation/extracted_{exam_mode}_data_for_question_generation.jsonl",
        help="the file path which stores the principles used for question generation"
    )
    parser.add_argument(
        "--generated_questions_save_path",
        type=str,
        default="./generated_questions_data/machine_generated_original_{exam_mode}_questions.csv",
        help="the file path which stores the generated questions"
    )

    args = parser.parse_args()
    main(args)

