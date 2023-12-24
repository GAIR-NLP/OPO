import tiktoken
import sys
sys.path.append(".")
sys.path.append("..")


import os
import jsonlines

import tiktoken
def num_tokens_from_string(string, encoding_name="cl100k_base", tokenizer=None) -> int:
    """Returns the number of tokens in a text string."""
    if tokenizer:
        encoding = tokenizer
    else:
        encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages_all, model="gpt-4"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages_all, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages_all, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for messages in messages_all:
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# def limit_length(input_texts, token_max_limit, tokenizer=None):
#     """
#     limit the input length
#     """
#     cur_token_num = 0
#     out_text = ""
#     for text in input_texts:
#         text_length = num_tokens_from_string(text, tokenizer=tokenizer)
#         cur_token_num +=text_length
#         if cur_token_num<token_max_limit:
#             out_text += "<{}> ".format(text)
#         else:
#             break
#     return out_text
def limit_length(input_texts, token_max_limit, tokenizer=None):
    """
    limit the input length
    """
    cur_token_num = 0
    out_text = ""
    for text in input_texts:
        text_length = num_tokens_from_string(text, tokenizer=tokenizer)
        cur_token_num +=text_length
        if cur_token_num<token_max_limit:
            out_text += text
            # out_text += "<{}> ".format(text)
        else:
            break
    return out_text



def calculate_total_length(messages, responses):
    """
    calculate the input length and output length
    """
    message_length = num_tokens_from_messages(messages)
    response_length = sum([num_tokens_from_string(response) for response in responses])
    return message_length+response_length





if __name__ == "__main__":
    # system_prompt_len = num_tokens_from_string(system_prompt, "cl100k_base")
    # print(system_prompt_len)
    # user_prompt_len = num_tokens_from_string(user_prompt, "cl100k_base")
    # print(user_prompt_len)
    #
    #
    # # calculate the example length
    # input_example_template = "法律：{example_law}\n\n" + "题目：{que}\n\n" + "选项分析：{analysis}\n\n" + "答案：{ans}\n\n"
    #
    # # first we need to convert the examples of txt format to jsonl format
    # example_txt_folder = '../examples_txt_'
    # example_files = os.listdir(example_txt_folder)
    # example_data_all = []
    # for exmaple_file in example_files:
    #     file_path = os.path.join(example_txt_folder, exmaple_file)
    #     with open(file_path, 'r') as f:
    #         cur_data = f.readlines()
    #         new_format_data = convert_txt2jsonl(cur_data)
    #         example_data_all.extend(new_format_data)
    # max_len = 0
    # for example in example_data_all:
    #     options = []
    #     for key, value in example["options"].items():
    #         option = key + '.' + value
    #         options.append(option)
    #     example_option = '\n'.join(options)
    #     que = example["que"] + "\n" + example_option
    #
    #     template = input_example_template.replace("{example_law}", example["law"]).replace("{que}", que) \
    #         .replace("{analysis}", example["analysis"]).replace("{ans}", example["ans"])
    #     length = num_tokens_from_string(template, "cl100k_base")
    #     if length > max_len:
    #         max_len = length
    #     # print(length)
    # print("max length for example is {}".format(str(max_len)))
    #
    # # calculate the input law length
    # # load input laws
    # input_laws_all = []
    # local_law_folder = "../法律文件/地方法规"
    # regulation_law_folder = "../法律文件/行政法规"
    # input_law_files = []
    # local_files = [os.path.join(local_law_folder, file) for file in os.listdir(local_law_folder)
    #                if file.endswith('.jsonl')]
    # regulation_files = [os.path.join(regulation_law_folder, file) for file in os.listdir(regulation_law_folder)
    #                     if file.endswith('.jsonl')]
    # input_law_files.extend(local_files)
    # input_law_files.extend(regulation_files)
    # for input_law_file in input_law_files:
    #     with jsonlines.open(input_law_file, 'r') as reader:
    #         for obj in reader:
    #             obj["law"] = obj["law"].replace("\u3000", "  ")
    #             # print(obj)
    #             input_laws_all.append(obj["law"])
    #
    # max_len = 0
    # for law in input_laws_all:
    #     length = num_tokens_from_string(law, "cl100k_base")
    #     if length > max_len:
    #         max_len = length
    #     # print(length)
    # print("max length for law is {}".format(str(max_len)))
    #
    #
    # # calculate the output length
    # gen_file = "/Users/XCP/Desktop/Code/pengfei_liu/safe_code/laws/safety_new/data_generation/output_gpt4.jsonl"
    # with jsonlines.open(gen_file, 'r') as reader:
    #     data = [obj["gen_data"] for obj in reader]
    # max_len = 0
    # for item in data:
    #     length = num_tokens_from_string(item, num_tokens_from_string)
    #     if length > max_len:
    #         max_len = length
    #     # print(length)
    # print("max length for generated data is {}".format(str(max_len)))

    print("calculate cost for generating data")
    total_input_len = 0
    total_output_len = 0
    gen_file = "../output_gpt4_delimiter.jsonl"
    with jsonlines.open(gen_file, "r") as reader:
        for item in reader:
            example = item["example"]
            example_input = example["law"]+' '.join(list(example["options"].values()))+example["analysis"]
            gen_data = item["gen_data"]
            prompt = system_prompt + user_prompt
            total_input_len += num_tokens_from_string(example_input+prompt+item["law"], "cl100k_base")
            total_output_len += num_tokens_from_string(gen_data, "cl100k_base")
    print("total input len: {}, total output len: {}".format(total_input_len, total_output_len))

    print("calculate cost for evaluating the generated data")
    total_input_len = 0
    total_output_len = 0
    eval_file = "../output_gpt4_eval_delimiter.jsonl"
    with jsonlines.open(eval_file, "r") as reader:
        for item in reader:
            eval_data = item["eval_obj"]
            input_law = item["input_law"]

            output = item["eval_result"]

            prompt = system_prompt + user_prompt
            total_input_len += num_tokens_from_string(eval_data + input_law + prompt, "cl100k_base")
            total_output_len += num_tokens_from_string(output, "cl100k_base")
    print("total input len: {}, total output len: {}".format(total_input_len, total_output_len))


