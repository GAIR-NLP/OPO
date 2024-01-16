import os
import re
from tqdm import tqdm
import random
import numpy as np
import torch
from transformers import GenerationConfig
from evaluation.base_evaluator import Evaluator

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    LlamaForCausalLM,
    LlamaTokenizer,

)
from script.gpt_usage import OpenAIChat_Embed,chunks
from script.calculate_length import limit_length, num_tokens_from_string
from retrieval.retrieval_relevant_docs import Retrieval
import gc


# model_name: model, tokenizer, maximum token length
MODEL_CLASSES = {
    "Llama-2-7b-chat": (LlamaForCausalLM, LlamaTokenizer, 4000),
    "Llama-2-13b-chat": (LlamaForCausalLM, LlamaTokenizer, 4000),
    "Llama-2-70b-chat": (LlamaForCausalLM, LlamaTokenizer, 4000),
    # "Llama-2-7b": (LlamaForCausalLM, LlamaTokenizer, 4000),
    # "Llama-2-13b": (LlamaForCausalLM, LlamaTokenizer, 4000),
    # "Llama-2-70b": (LlamaForCausalLM, LlamaTokenizer, 4000),
    "THUDM/chatglm2-6b": (AutoModel, AutoTokenizer, 8000),
    "THUDM/chatglm3-6b": (AutoModel, AutoTokenizer, 8000),
    "internlm/internlm-chat-7b": (AutoModelForCausalLM, AutoTokenizer, 8000),
    "internlm/internlm-chat-20b": (AutoModelForCausalLM, AutoTokenizer, 8000),
    "Qwen/Qwen-7B-Chat": (AutoModelForCausalLM, AutoTokenizer, 8000),
    "Qwen/Qwen-14B-Chat": (AutoModelForCausalLM, AutoTokenizer, 8000),
    "ShengbinYue/DISC-LawLLM":(AutoModel, AutoTokenizer, 4000),
    "Duxiaoman-DI/XuanYuan-70B":(LlamaForCausalLM, LlamaTokenizer, 4000),
    "xverse/XVERSE-7B-Chat": (AutoModelForCausalLM, AutoTokenizer, 8000),
    "xverse/XVERSE-13B-Chat": (AutoModelForCausalLM, AutoTokenizer, 8000),
}

class Weight_Model_Evaluator(Evaluator):
    def __init__(self, args, verbose=True, temperature=0.2):
        super(Weight_Model_Evaluator, self).__init__(args)
        self.args = args
        # embedding model in contained in the OpenAIChat_Embed
        self.openai_model = OpenAIChat_Embed(embed_model_name='text-embedding-ada-002')

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        model_class, tokenizer_class, self.max_length = MODEL_CLASSES[args.evaluation_model]
        # If the args.model_path is set, load the local model. If not, load the model from Huggingface.
        model_path = args.model_path if args.model_path else args.evaluation_model
        self.tokenizer = tokenizer_class.from_pretrained(model_path, trust_remote_code=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model_class.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        self.generation_config = GenerationConfig(
            temperature=0,
            top_k=40,
            top_p=0.9,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            max_length=self.max_length
        )

        self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
        self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]


    def eval(self,
             test_df,
             dev_df=None,
             few_shot=False,
             save_result_dir=None,
             cot=False,
             use_retrieval=True,
             use_options=True,
             use_note=True,
             do_test=False
        ):
        if self.args.constrained_decoding is True:
            self.generation_config.output_scores = True
            self.generation_config.return_dict_in_generate = True
            self.generation_config.max_new_tokens = 1
            self.generation_config.top_p = 1.0
            self.generation_config.top_k = 0

        if save_result_dir:
            result = []
            score = []
        correct_num = 0
        all_answers = []
        if few_shot:
            # Few_shot is not implemented now because of the length limit.
            raise NotImplementedError  # Maybe we need to implement it.
            # history = self.generate_llama2_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = self.system_prompt

        # retrieval similar docs from constructed data
        if use_retrieval:
            question_all = []
            for row_index, row in test_df.iterrows():
                cur_question = row['question']
                if use_options:  # if use options for retrieval, we concat the question stem and the options
                    for choice in self.choices:
                        cur_question += f'\n{choice}. {row[f"{choice}"]}'
                question_all.append(cur_question)
            # retrieve related docs from the constructed retrieval source
            retrieval_model = Retrieval(embed_model=self.openai_model, args=self.args)
            retrieval_results = retrieval_model.retrieve(question_all)

        answers = ['NA'] * len(test_df) if do_test is True else list(test_df['answer'])
        with torch.no_grad():
            for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
                if use_retrieval:
                    # calculate the token number for the question
                    example = row['question']
                    for choice in self.choices:
                        example += f'\n{choice}. {row[f"{choice}"]}'
                    example_num_tokens = num_tokens_from_string(example, tokenizer=self.tokenizer)

                    cur_retrieval = retrieval_results[row_index]
                    # limit the input retrieval text length, 1000 tokens is remained for the output and other few tokens (e.g., <>, prompt)
                    retrieval_token_max = self.args.retrieval_token_max
                    cur_retrieval = limit_length(cur_retrieval,
                                                 token_max_limit=retrieval_token_max,
                                                 #token_max_limit=self.max_length-example_num_tokens-1000,
                                                 tokenizer=self.tokenizer
                                                 )

                    question = self.format_example(row,
                                                   include_answer=False,
                                                   cot=cot,
                                                   use_note=use_note,
                                                   retrieval_content=cur_retrieval
                                                   )
                else:
                    # if we don't use retrieval, the default value for use_note is False
                    question = self.format_example(row,
                                                   include_answer=False,
                                                   cot=cot
                                                   )
                # if with_prompt:
                #     prompt_template = (
                #                             "[INST] <<SYS>>\n"
                #                             "{system_prompt}\n"
                #                             "<</SYS>>\n\n"
                #                             "{instruction} [/INST]"
                #                         )
                #
                #     instruction = prompt_template.format_map({'instruction': instruction,'system_prompt':DEFAULT_SYSTEM_PROMPT})
                instruction = history + '\n' + question
                inputs = self.tokenizer(instruction, return_tensors="pt")
                generation_output = self.model.generate(
                        input_ids = inputs["input_ids"].to(self.model.device),
                        attention_mask = inputs['attention_mask'].to(self.model.device),
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        generation_config = self.generation_config
                    )

                batch_size, length = inputs.input_ids.shape
                # print(
                #     "for {}, A: {}, B: {}, C;{}, D:{}".format(self.args.evaluation_model, self.sA_id, self.sB_id, self.sC_id,
                #                                               self.sD_id))

                if self.args.constrained_decoding is True:
                    logits = generation_output.scores[0][0]

                    logits = logits.float().cpu().detach()
                    if self.args.exam_mode=="social_morality":  # there are 3 options for social questions
                        choices_logits = logits[[self.sA_id,self.sB_id,self.sC_id]].numpy()
                        ans = {0: "A", 1: "B", 2: "C"}[np.argmax(choices_logits)]
                    else:
                        choices_logits = logits[[self.sA_id, self.sB_id, self.sC_id, self.sD_id]].numpy()
                        ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choices_logits)]
                    # print(choices_logits)
                    response = self.tokenizer.decode([logits.argmax(-1).item()])
                else:
                    response = self.tokenizer.decode(generation_output[0, length:], skip_special_tokens=True)
                    ans, direct_extract = self.extract_answer(row, response)
                if ans == answers[row_index]:
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
                if self.verbose is True:
                    print(f"\n======={str(row_index)}=======")
                    print(f"question: {question}\n")
                    print(f"response: {response}\n")
                    print(f"extracted answer: {ans}")
                    print(f"ground truth: {answers[row_index]} \n")
                if save_result_dir:
                    result.append(response)
                    score.append(correct)

                all_answers.append(ans)

        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output'] = result
            test_df['generated_answer'] = all_answers
            test_df['correctness'] = score
            test_df.to_csv(save_result_dir, encoding="utf-8", index=False)
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        return correct_ratio, all_answers

    def format_example(self, line, include_answer=False, cot=False, use_note=False, retrieval_content=None):

        example = line['question']

        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + \
                    line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += '\n答案：' + line["answer"] + '\n\n'
        else:
            if retrieval_content:
                if use_note:
                    user_prompt = self.user_prompt_full
                else:
                    user_prompt = self.user_prompt_with_retrieval
                example = user_prompt.replace("{question}", example)\
                    .replace("{supplementary_information}", retrieval_content)
            else:
                example = self.user_prompt_base.replace("{question}", example)

        return example

    def extract_answer(self, line, gen_ans):
        m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案应该是\s?选?项?\s?([A-D])",
            r"答案选项应该是\s?([A-D])",
            r"答案选项为\s?([A-D])",
            r"答案选项是\s?([A-D])",
            r"选项\s?([A-D])是正确",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
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
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', gen_ans, re.M)
        if len(m) >= 1:
            answer = m[0]
            return answer, False
        # only containing one choice-context
        choices_dict = {}
        pattern = ""
        for c in self.choices:
            choices_dict[str(line[f'{c}'])] = c
            pattern += re.escape(str(line[f'{c}']))+"|"
        pattern = pattern[:-1]
        m = re.findall(pattern, gen_ans, re.M)
        print("w/ escape:",repr(pattern),gen_ans,(len(m)>=1))
        if len(m) >= 1:
            answer = choices_dict[m[0]]
            return answer, False
        return  '', False


