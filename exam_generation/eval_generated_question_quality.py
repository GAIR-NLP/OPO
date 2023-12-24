import sys
sys.path.append('..')
import openai
from eval_generated_question_quality_prompt import eval_user_prompt, eval_system_prompt
from time import sleep
import os
import random
import jsonlines
from tqdm import tqdm
import asyncio
import tiktoken
import argparse
import pandas as pd
from script.gpt_usage import chunks, OpenAIChat_Embed
import re

choices = ["A", "B", "C", "D"]
def eval_data_quality(args):
    generated_questions = pd.read_csv(args.input_generated_question_path.replace("{exam_mode}", args.exam_mode))
    messages_all = []
    for index, item in generated_questions.iterrows():
        input_principle = item["principle"]
        que = "题干:"+item["question"]
        options = "备选项:"
        for choice in choices:
            options+=f'\n{choice}. {item[f"{choice}"]}'
        analysis = "选项分析:"+item["analysis"]
        answer = "答案:" + item["answer"]
        full_que = que+"\n"+options+"\n"+analysis+"\n"+answer
        user_prompt_ = eval_user_prompt.replace("{supplementary_information}", input_principle).replace("{question}", full_que)
        if args.exam_mode=="law":
            if str(item["location"])!="nan":
                system_prompt_ = eval_system_prompt.replace("{additional_content}", "，是否包含指定地点{}".format(item["location"]))
            else:
                system_prompt_ = eval_system_prompt.replace("{additional_content}","")
        elif args.exam_mode == "morality":
            if str(item["location"])!="nan":
                system_prompt_ = eval_system_prompt.replace("{additional_content}", "，是否包含指定公司名称(或者协会名称){}".format(item["location"]))
            else:
                system_prompt_ = eval_system_prompt.replace("{additional_content}","")
        else:
            raise NotImplementedError
        messages = [
            {"role": "system", "content": system_prompt_},
            {"role": "user", "content": user_prompt_}
        ]
        print(system_prompt_)
        print(user_prompt_)
        messages_all.append(messages)
    messages_all = messages_all
    openai_model = OpenAIChat_Embed(chat_model_name=args.evaluation_model,
                                    temperature=args.temperature,
                                    )
    # generate the response based on the input message
    message_batch_num = min(50, len(messages_all))
    message_batch = chunks(messages_all, message_batch_num)
    responses_all = []
    for cnt, messages in tqdm(enumerate(message_batch), total=len(messages_all) / message_batch_num):
        cur_responses = asyncio.run(openai_model.async_run(
            messages_list=messages,
            mode="chat"
        ))
        responses_all.extend(cur_responses)
    
    # get the perfect questions and save them
    valid_index = []
    stem_all, option_all, analysis_all, answer_all = [], [], [], []
    for idx, response in enumerate(responses_all):
        stem_rule = r'题干是否合理:([\s\S]*?)备选项是否合理:'
        option_rule = r'备选项是否合理:([\s\S]*?)选项分析是否正确:'
        analysis_rule = r'选项分析是否正确:([\s\S]*?)答案是否正确:'
        # answer_rule = r'答案是否正确:([\s\S]*?)'
        try:
            stem = re.findall(stem_rule, response)[0].strip()
            option = re.findall(option_rule, response)[0].strip()
            analysis = re.findall(analysis_rule, response)[0].strip()
            answer = response.split("答案是否正确:")[1].strip()

            stem_all.append(stem)
            option_all.append(option)
            analysis_all.append(analysis)
            answer_all.append(answer)
            # only save the question when the judgements for all items are yes
            if stem.startswith("是") and option.startswith("是") and analysis.startswith("是") and answer.startswith("是"):
                valid_index.append(idx)

        except:
            stem_all.append('')
            option_all.append('')
            analysis_all.append('')
            answer_all.append('')
            print("there are something wrong with generated question {}".format(response))

    # extract the valid questions
    perfect_questions = generated_questions.iloc[valid_index]
    # save the perfect questions
    perfect_questions.to_csv(args.perfect_questions_save_path.replace("{exam_mode}", args.exam_mode), index=False)

    # save the quality evaluation results
    generated_questions["stem_quality"] = stem_all
    generated_questions["option_quality"] = option_all
    generated_questions["analysis_quality"] = analysis_all
    generated_questions["answer_quality"] = answer_all
    questions_quality_eval_save_path = args.questions_quality_eval_save_path.replace("{exam_mode}", args.exam_mode)
    generated_questions.to_csv(questions_quality_eval_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--evaluation_model",
        type=str,
        default="gpt-4-0613"
    )
    parser.add_argument(
        "--exam_mode",
        type=str,
        default="law",
        help="choose the mode from [law, professional_morality]"
    )
    parser.add_argument(
        "--input_generated_question_path",
        type=str,
        default="./generated_questions_data/machine_generated_original_{exam_mode}_questions_re.csv",
        help="the file path which stores the generated questions"
    )
    parser.add_argument(
        "--questions_quality_eval_save_path",
        type=str,
        default="./generated_questions_data/machine_generated_{exam_mode}_questions_quality_eval.csv",
        help="the file path which stores the quality evaluation results for the generated questions"
    )
    parser.add_argument(
        "--perfect_questions_save_path",
        type=str,
        default="./generated_questions_data/machine_generated_{exam_mode}_perfect_questions.csv",
        help="the file path which stores the quality evaluation results for the generated questions"
    )
    args = parser.parse_args()

    eval_data_quality(args)
