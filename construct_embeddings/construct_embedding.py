import os
import json
import argparse
import openai
import jsonlines
import asyncio
import tiktoken
from tqdm import tqdm
import pickle
from script.gpt_usage import OpenAIChat_Embed


def load_retrieval_data(data_root_path):
    """
    Load the raw retrieval texts
    """
    data_all = []  # store all data
    # read all files
    folders = [folder for folder in os.listdir(data_root_path) if folder!=".DS_Store"]
    for folder in folders:
        folder_path = os.path.join(data_root_path, folder)
        files = [file for file in os.listdir(folder_path) if file!=".DS_Store"]
        for file in tqdm(files, desc="read raw files"):
            file_path = os.path.join(data_root_path, folder, file)
            with jsonlines.open(file_path, "r") as reader:
                items = [obj for obj in reader]
            data_all.extend(items)

    #  change the data format: add the source information to the text
    text_all = [item["source"].split("/")[1]+item["text"].replace("*****", "：")  for item in data_all]
    source_all = [item["source"] for item in data_all]
    return  text_all, source_all



def truncate_text_tokens(texts, encoding_name='cl100k_base', max_tokens=8191):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    truncated_tokens = [encoding.encode(text)[:max_tokens] for text in texts]
    return truncated_tokens

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def embedding_check(embed_data_all, tokens_all):
    # check the data, return the text whose embedding is None
    remain_data_idx = []
    for idx, embed in enumerate(embed_data_all):
        if embed == None:
            remain_data_idx.append(idx)
    remain_tokens = [tokens_all[idx] for idx in remain_data_idx]
    return embed_data_all, tokens_all, remain_data_idx, remain_tokens

def extract_embedding_from_text(data_root_path, embedding_final_save_path,
                                source_save_path, exam_mode):
    """
    data_root_path: a folder which stores the original texts
    embedding_partial_save_path: a path, where each file stores the extracted embeddings related to the batch
    embedding_final_save_path: a path, which store the combined embeddings
    source_save_path: a path, which save all sources of the files
    """
    data_root_path = data_root_path.format(exam_mode)

    embedding_final_save_path = os.path.join(embedding_final_save_path, exam_mode)
    texts, sources = load_retrieval_data(data_root_path)

    # texts = texts[:500]
    # sources = sources[:500]

    tokens_all = truncate_text_tokens(texts, encoding_name='cl100k_base', max_tokens=400)
    print("total tokens are {}".format(sum([len(token) for token in tokens_all])))

    if not os.path.exists(embedding_final_save_path):
        os.makedirs(embedding_final_save_path)
    if not os.path.exists(source_save_path):
        os.makedirs(source_save_path)


    embed_model = OpenAIChat_Embed()
    # generate the response based on the input message
    token_batch_num = min(100, len(tokens_all))
    token_batch = chunks(tokens_all, token_batch_num)
    embeddings_all = []
    for cnt, batch in tqdm(enumerate(token_batch), total=len(tokens_all) / token_batch_num):
        cur_embeddings = asyncio.run(embed_model.async_run(
            messages_list=batch,
            mode="embed"
        ))
        embeddings_all.extend(cur_embeddings)

    text_embed_pairs = list(zip(texts, embeddings_all))

    with open(os.path.join(embedding_final_save_path, f"{exam_mode}_embed_text_pairs.pkl"), "wb") as f:
        pickle.dump(text_embed_pairs, f)
    # we also need to store the sources of files for the subsequent retrieval
    json.dump(sources, open(os.path.join(source_save_path, f"{exam_mode}_source.json"), "w", encoding="utf8"),
              ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root_path",
        type=str,
        default="../data/retrieval_raw_data/{}_rules",
    )
    parser.add_argument(
        "--embedding_final_save_path",
        type=str,
        default="../data/retrieval_processed_embed_text",
        help="store the final embeddings",
    )
    parser.add_argument(
        "--source_save_path",
        type=str,
        default="../data/retrieval_source_info",
    )
    parser.add_argument(
        "--exam_modes",
        type=list,
        default=["social_norm"],
        help="choose the exam_mode from [law_ndlr, law_ndgr, basic_morality, professional_morality, social_morality]"
    )
    args = parser.parse_args()

    for exam_mode in args.exam_modes:
        extract_embedding_from_text(args.data_root_path, args.embedding_final_save_path,
                                    args.source_save_path, exam_mode)