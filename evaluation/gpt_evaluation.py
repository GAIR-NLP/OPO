from tqdm import tqdm
from evaluation.base_evaluator import Evaluator
from script.gpt_usage import OpenAIChat_Embed,chunks
from retrieval.retrieval_relevant_docs import Retrieval
from script.calculate_length import limit_length
import asyncio
class GPT_Evaluator(Evaluator):
    def __init__(self, args):
        super(GPT_Evaluator, self).__init__(args)
        self.args = args
        # embedding model in contained in the OpenAIChat_Embed
        self.openai_model = OpenAIChat_Embed(chat_model_name=args.evaluation_model,
                                             temperature=args.temperature,
                                             embed_model_name='text-embedding-ada-002',
                                             )

    def format_example(self,line,include_answer=False,cot=False, use_note=False, retrieval_content=None):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            if retrieval_content:
                if use_note:
                    user_prompt = self.user_prompt_full
                else:
                    user_prompt = self.user_prompt_with_retrieval
                user_prompt = user_prompt.replace("{question}", example) \
                    .replace("{supplementary_information}", retrieval_content)
            else:
                user_prompt = self.user_prompt_base.replace("{question}", example)
            return [
                {"role": "user", "content": user_prompt}
            ]


    def eval(self,
             test_df,
             dev_df=None,
             few_shot=False,
             save_result_dir=None,
             cot=False,
             use_retrieval=False,
             use_options=False,
             use_note=False
             ):

        if few_shot:
            raise NotImplementedError
        else:
            few_shot_prompt=[
                {
                    "role":"system",
                    "content":self.system_prompt
                }
            ]

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
            # # limit the input retrieval text length
            # retrieval_results = limit_length(retrieval_results, token_max_limit=4000)
            full_message_all = []
            for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
                cur_retrieval = retrieval_results[row_index]
                # limit the input retrieval text length
                if "gpt-3.5" in self.args.evaluation_model:
                    retrieval_token_max = 2000
                elif "gpt-4" in self.args.evaluation_model:
                    retrieval_token_max = 4000
                else:
                    retrieval_token_max = 2000
                retrieval_token_max = self.args.retrieval_token_max
                cur_retrieval = limit_length(cur_retrieval, token_max_limit=retrieval_token_max)
                question = self.format_example(row, include_answer=False,cot=cot, use_note=use_note,
                                           retrieval_content=cur_retrieval)
                full_message = few_shot_prompt + question
                full_message_all.append(full_message)
        else:
            full_message_all = []
            for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = self.format_example(row, include_answer=False, cot=cot)
                full_message = few_shot_prompt + question
                full_message_all.append(full_message)

        message_batch_num = min(100, len(full_message_all))
        message_batch = chunks(full_message_all, message_batch_num)
        responses_all = []
        for cnt, messages in tqdm(enumerate(message_batch), total=len(full_message_all)/message_batch_num):
            cur_responses = asyncio.run(self.openai_model.async_run(
                messages_list=messages,
                mode="chat"
            ))
            responses_all.extend(cur_responses)
        print(responses_all)
        gen_ans = self.extract_answer(responses_all)
        print(gen_ans)
        scores, correct_ratio = self.calculate_accuracy(gen_ans, test_df.loc[:,"answer"].values)
        if save_result_dir:
            test_df.loc[:, 'model_output'] = responses_all
            test_df.loc[:, 'generated_answer'] = gen_ans
            test_df.loc[:, "correctness"] = scores
            test_df.to_csv(save_result_dir, encoding="utf-8", index=False)
        return correct_ratio, gen_ans

