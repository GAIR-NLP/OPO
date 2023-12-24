# Chinese prompts
system_prompt = """你是一个中文人工智能助手，你需要做一个单项选择题，从4个选项中选出一个正确答案。"""

user_prompt_base = """问题：{question}
答案："""

user_prompt_with_retrieval = """辅助信息：{supplementary_information}

问题：{question}
答案："""

# user_prompt_with_retrieval = """辅助信息：{supplementary_information}
#
# 问题：{question}
# 答案："""

user_prompt_full = """辅助信息：{supplementary_information}
注意：因为辅助信息有限，不一定包含对答题有用的信息，同时辅助信息中没有的内容不一定是错误的，因此你需要合理利用辅助信息，并结合自身知识与常识选出正确答案，不能完全依赖辅助信息。

问题：{question}
答案："""

# English prompts for social norm evaluation
system_prompt_en = """You are an artificial intelligence assistant. You need to answer a multiple-choice question and choose the correct answer from 3 options.
The following definitions are used for 'expected', 'normal', or 'taboo':
Expected: Behaviors that a certain kind of person would be predicted or required to do in this setting. There is often some kind of obligation here.
Normal: Behaviors that a certain kind of person would reasonably do in this setting, but there is no obligation.
Taboo: Behaviors that may be socially or culturally taboo, forbidden, discouraged, or generally surprising and "not normal" for a certain kind of person to do in this setting."""

user_prompt_base_en = """Question: {question}
Answer:"""

user_prompt_with_retrieval_en = """Supplementary information: {supplementary_information}

Question: {question}
Answer:"""

user_prompt_full_en = """Supplementary information: {supplementary_information}
Attention: Due to limited auxiliary information, it may not necessarily include helpful information for answering questions. Additionally, the absence of content in the auxiliary information does not necessarily mean it is incorrect. Therefore, it is important to use auxiliary information reasonably and combine it with personal knowledge and common sense to select the correct answer, rather than relying entirely on the auxiliary information.

Question: {question}
Answer:"""



system_prompt_en_llama_2 = """You are an artificial intelligence assistant. You need to answer a multiple-choice question and choose a correct answer from 3 options."""


system_prompt_en_qwen = """<|im_start|>system\nYou are an artificial intelligence assistant. You need to do a multiple-choice question and choose a correct answer from 3 options.
The following definitions are used for 'expected', 'normal', or 'taboo':
Expected: Behaviors that a certain kind of person would be predicted or required to do in this setting. There is often some kind of obligation here.
Normal: Behaviors that a certain kind of person would reasonably do in this setting, but there is no obligation.
Taboo: Behaviors that may be socially or culturally taboo, forbidden, discouraged, or generally surprising and "not normal" for a certain kind of person to do in this setting.<|im_end|>\n"""

user_prompt_base_en_qwen = """<|im_start|>user\nQuestion: {question}
Answer:<|im_end|>\n<|im_start|>assistant\n"""

user_prompt_with_retrieval_en_qwen = """<|im_start|>user\nSupplementary information: {supplementary_information}

Question: {question}
Answer:<|im_end|>\n<|im_start|>assistant\n"""

user_prompt_full_en_qwen = """<|im_start|>user\nSupplementary information: {supplementary_information}
Attention: Due to limited auxiliary information, it may not necessarily include helpful information for answering questions. Additionally, the absence of content in the auxiliary information does not necessarily mean it is incorrect. Therefore, it is important to use auxiliary information reasonably and combine it with personal knowledge and common sense to select the correct answer, rather than relying entirely on the auxiliary information.

Question: {question}
Answer:<|im_end|>\n<|im_start|>assistant\n"""




# 注意：辅助信息只是一个工具，请谨慎使用辅助信息。辅助信息可能包含对答题有用的内容，也可能包含一些无用的噪声信息，同时辅助信息中没有的内容不一定是错误的。因此你在答题时，需要合理利用辅助信息，结合辅助信息进行推理，并充分利用自身知识与常识选出正确答案，不能依赖辅助信息。
