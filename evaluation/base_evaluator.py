import re
import string
from evaluation.evaluation_prompt import *
class Evaluator:
    def __init__(self, args):
        if args.exam_mode == "social_morality":
            self.choices = ["A", "B", "C"]  # there are three options for professional moral questions
            # because the language of social norm question is en, so we ues prompt of en version
            self.system_prompt = system_prompt_en
            self.user_prompt_base = user_prompt_base_en
            self.user_prompt_with_retrieval = user_prompt_with_retrieval_en
            self.user_prompt_full = user_prompt_full_en

            if "Qwen" in args.evaluation_model:
                self.system_prompt = system_prompt_en_qwen
                self.user_prompt_base = user_prompt_base_en_qwen
                self.user_prompt_with_retrieval = user_prompt_with_retrieval_en_qwen
                self.user_prompt_full = user_prompt_full_en_qwen
            if "Llama-2-7b-chat" in args.evaluation_model or "Llama-2-13b-chat" in args.evaluation_model:
                self.system_prompt = system_prompt_en_llama_2
        else:
            self.choices = ["A", "B", "C", "D"]

            self.system_prompt = system_prompt
            self.user_prompt_base = user_prompt_base
            self.user_prompt_with_retrieval = user_prompt_with_retrieval
            self.user_prompt_full = user_prompt_full
        self.puncs = list(string.punctuation)

    # def format_example(self, line, include_answer=True):
    #     example = line['question']
    #     # print(example)
    #     for choice in self.choices:
    #         example += f'\n{choice}. {line[f"{choice}"]}'
    #     example += '\n答案：'
    #     if include_answer:
    #         example += f'{line["answer"]}\n\n'
    #     return example



    # def generate_few_shot_prompt(self, subject, dev_df):
    #     prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
    #     k = self.k
    #     if self.k == -1:
    #         k = dev_df.shape[0]
    #     for i in range(k):
    #         prompt += self.format_example(dev_df.iloc[i, :])
    #     return prompt


    def normalize_answer(self,s):

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude=set(self.puncs)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self,pred, target):
        return self.normalize_answer(pred)==self.normalize_answer(target)

    def extract_answer(self, responses):
        pattern = [
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案应该是\s?选?项?\s?([A-D])",
            r"答案选项应该是\s?([A-D])",
            r"答案选项为\s?([A-D])",
            r"答案选项是\s?([A-D])",
            r"选项\s?([A-D])是正确",
            r"选项\s?([A-D])为正确",
            r"选项\s?([A-D])正确",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案选项：\s?选?项?\s?([A-D])",
            r"答案选项对应的字母是([A-D])"
        ]

        ans_list_all = []  # store the extracted answer for all response
        for response in responses:
            cur_ans_list = []

            if response[0] in ["A", 'B', 'C', 'D']:
                cur_ans_list.append(response[0])
            for p in pattern:
                if len(cur_ans_list) == 0:
                    cur_ans_list = re.findall(p, response)
                else:
                    break
            if len(cur_ans_list):
                cur_ans = cur_ans_list[-1]
            else:
                cur_ans_list = re.findall(r"([A-D])", response)
                if len(cur_ans_list)==1:
                    cur_ans = cur_ans_list[0]
                else:
                    cur_ans = ''
            ans_list_all.append(cur_ans)
        return ans_list_all

    def calculate_accuracy(self, gen_answers, rows):
        correct_num = 0
        scores = []
        for idx, (gen_ans, tgt_answer) in enumerate(zip(gen_answers, rows)):
            if gen_ans==tgt_answer:
                correct_num+=1
                scores.append(1)
            else:
                scores.append(0)
        correct_ratio = 100*correct_num/len(gen_answers)
        return scores, correct_ratio