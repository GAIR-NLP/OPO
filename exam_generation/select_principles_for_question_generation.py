import json
import sys
sys.path.append("../")
import os

import cn2an
import jsonlines
import openai
import asyncio
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import jsonlines
from typing import List
import random
from script.calculate_length import num_tokens_from_string
import re
import argparse
from script.project_province_file import project_file2province
from construct_embeddings.construct_embedding import chunks,embedding_check,truncate_text_tokens
from script.gpt_usage import OpenAIChat_Embed





def search(query_embed, retrieval_embed, input_data, top_n=5):
    scores = cosine_similarity(query_embed, retrieval_embed)[0]
    index = np.argsort(-scores)[:top_n]
    retrieved_laws = [input_data[i] for i in index]
    return retrieved_laws

def convert_text2embedding(api_base: str, data_store: List) -> List[str]:
    # place all the texts in a file into a list.
    texts_all = [item["text"] for item in data_store]
    embed_model = OpenAIChat_Embed(api_base=api_base)
    text_per_batch_num = min(100, len(texts_all))
    texts_all = truncate_text_tokens(texts_all, encoding_name='cl100k_base', max_tokens=8000)
    text_batch = chunks(texts_all, text_per_batch_num)
    text_embed_all = []
    for cnt, texts in tqdm(enumerate(text_batch), total=len(texts_all) / text_per_batch_num):
        cur_embeddings = asyncio.run(embed_model.async_run(
            messages_list=texts,
            mode="embed"
        ))
        text_embed_all.extend(cur_embeddings)

    return text_embed_all

def retrieve_similar_principles(query_text_info: dict, qeuery_embedding: List[int],  texts_info: List[dict],
                              embeddings: List[List[int]], token_max_limit: int, num_max_limit: int):
    retrieved_texts = search([qeuery_embedding], embeddings, texts_info)
    retrieved_texts = retrieved_texts[1:]  # 去除第一个，第一个是它自身
    assert query_text_info not in retrieved_texts

    out_texts = [query_text_info]
    cur_len = num_tokens_from_string(query_text_info["text"], "cl100k_base")
    for retrieved_text in retrieved_texts:
        next_len = num_tokens_from_string(retrieved_text["text"], "cl100k_base")
        cur_len+=next_len
        if cur_len < token_max_limit and len(out_texts)<num_max_limit:
            out_texts.append(retrieved_text)
        else:
            break
    return out_texts



def search_similar_texts(query_item, file_data, embeddings, token_max_limit, query_num_max_limit)->List[List[dict]]:

    query_idx = cn2an.cn2an(re.findall(r"第(.+?)条", query_item["num"])[0])-1  # "-1" because the data starts with "第一条"
    query_embedding = embeddings[query_idx]

    scores = cosine_similarity([query_embedding], embeddings)[0]
    index = np.argsort(-scores)[:query_num_max_limit][1:]
    retrieved_texts = [file_data[i] for i in index]
    out_texts = [query_item]
    cur_len = num_tokens_from_string(query_item["text"], "cl100k_base")
    for retrieved_text in retrieved_texts:
        next_len = num_tokens_from_string(retrieved_text["text"], "cl100k_base")
        cur_len += next_len
        if cur_len < token_max_limit and len(out_texts) < query_num_max_limit:
            out_texts.append(retrieved_text)
        else:
            break
    return out_texts



def read_file(file):
    with jsonlines.open(file, 'r') as reader:
        data = [obj for obj in reader]
    if len(data)==1:
        return None
    data_store = []
    for item in data:
        num, text = item["text"].split("*****")
        data_store.append({"num": num, "text": text, "source": file})
    return data_store

def load_files(data_path):
    data_all = []
    folders = sorted(os.listdir(data_path))
    for folder in folders:
        if not folder.startswith("."):
            files = os.listdir(os.path.join(data_path, folder))
            for file in files:
                if not file.startswith("."):
                    file_name = file.split(".")[0]
                    if file_name not in set(excluded_laws):
                        file_path = os.path.join(data_path, folder, file)
                        cur_file_data = read_file(file_path)
                        if cur_file_data!=None: # remove the file with one line, we need to search similar texts
                            data_all.extend(cur_file_data)
    return data_all
def main(args):

    # set the save path
    if not os.path.exists('/'.join(args.extracted_principle_save_path.split('/')[:-1])):
        os.makedirs('/'.join(args.extracted_principle_save_path.split('/')[:-1]))
    extracted_principle_save_path = args.extracted_principle_save_path.replace("{exam_mode}", args.exam_mode)
    # select a text and obtain similar text. Those texts are regarded as a principle for generating questions
    # step 1: randomly select a text,
    # step 2: find similar texts
    # here is the step 1:
    if args.exam_mode == "law":
        input_data_path = args.input_data_root_path.format("law_ndlr")  # load law_ndlr data
        # data_all = load_files(input_data_path)
        data_all = []
        input_data_path = args.input_data_root_path.format("law_ndgr")  # load law_ndgr data
        data_all.extend(load_files(input_data_path))
    else:
        input_data_path = args.input_data_root_path.format(args.exam_mode)  # exam_mode: morality
        data_all = load_files(input_data_path)
    random.seed(42) # for reproducibility
    sampled_items = random.sample(data_all, args.sample_principle_num)  # select principles
    # sampled_items = sampled_items[:500]
    # check the length of the text and ensure that it is below 2000 tokens due to the context limit of GPT.
    sampled_items_ = []
    for item in sampled_items:
        token_num = num_tokens_from_string(item["text"])
        if token_num<2000:
            sampled_items_.append(item)


    law_file2province_data = project_file2province(args.province2law_file_path)  # load the file2province info for law
    # step 2:
    embed_model = OpenAIChat_Embed()

    saved_texts = []
    for query_item in tqdm(sampled_items_, desc="search similar texts for query"):
        file_path = query_item["source"]
        file_data = read_file(file_path) # for each selected principle, load the data of related file

        texts_all = [item["text"] for item in file_data]
        tokens_all = truncate_text_tokens(texts_all, encoding_name='cl100k_base', max_tokens=8000)
        # generate the response based on the input message

        token_batch_num = min(1000, len(tokens_all))
        token_batch = chunks(tokens_all, token_batch_num)
        embeddings_all = []
        for cnt, batch in tqdm(enumerate(token_batch), total=len(tokens_all) / token_batch_num):
            cur_embeddings = asyncio.run(embed_model.async_run(
                messages_list=batch,
                mode="embed"
            ))
            embeddings_all.extend(cur_embeddings)

        input_embeds = embeddings_all
        # input_embeds = convert_text2embedding(args.openai_api_base, file_data)
        extracted_texts = search_similar_texts(query_item, file_data, input_embeds, args.principle_token_max,
                                               args.selected_text_num)
        file_name = file_path.split("/")[-1].split(".jsonl")[0]
        # add the province and city information to the file_name for "law", and don't need to add info for "morality"
        if args.exam_mode=="law":
            province_city_info = law_file2province_data[file_name]
            if province_city_info["province"]=="全国":  # province is "全国",
                    add_location_info = ''
            else:
                province = province_city_info["province"]
                city = province_city_info["city"]
                add_location_info = ''
                if province not in file_name:
                    add_location_info += province

                if city!="省直" and city not in file_name:
                    add_location_info+=city
            file_name = add_location_info+file_name

        last_text_template = "{}规定:".format(file_name)
        # sort the text according to "第x条"
        texts = sorted(extracted_texts, key=lambda x: cn2an.cn2an(re.findall(r"第(.+?)条", x["num"])[0]))
        for i, law in enumerate(texts):
            num = law["num"]
            text = law["text"]
            cur_text = "<" + ":".join([num, text]) + "> "
            if i == 0:
                last_text = last_text_template + cur_text
            else:
                last_text = last_text + cur_text

        location_info = ''
        if args.exam_mode=="law":
            if province_city_info["province"]!="全国":
                location_info = province_city_info["province"]
                if province_city_info["city"] != "省直" and province_city_info["city"] not in location_info:
                    location_info+=province_city_info["city"]
        elif args.exam_mode=="professional_morality":
            location_info = file_name
        else:
            raise NotImplementedError
        saved_texts.append({"principle": last_text, "location": location_info})
    with jsonlines.open(args.extracted_principle_save_path.replace("{exam_mode}", args.exam_mode), 'w') as writer:
        writer.write_all(saved_texts)


if __name__ == "__main__":
    # Note: there maybe duplicate laws in the exclude_laws
    excluded_laws = [
                    # 宪法相关的法律法规
                    "中华人民共和国宪法（2018年修正文本）", "反分裂国家法", "中华人民共和国国家安全法", "中华人民共和国立法法",
                    "中华人民共和国全国人民代表大会和地方各级人民代表大会选举法", "中华人民共和国全国人民代表大会和地方各级人民代表大会代表法",
                    "中华人民共和国地方各级人民代表大会和地方各级人民政府组织法", "中华人民共和国全国人民代表大会组织法", "中华人民共和国国务院组织法",
                    "中华人民共和国监察法", "中华人民共和国各级人民代表大会常务委员会监督法", "中华人民共和国民族区域自治法", "中华人民共和国香港特别行政区基本法",
                    "中华人民共和国香港特别行政区基本法附件一香港特别行政区行政长官的产生办法", "中华人民共和国香港特别行政区基本法附件二香港特别行政区立法会的产生办法和表决程序",
                    "全国人民代表大会关于建立健全香港特别行政区维护国家安全的法律制度和执行机制的决定", "中华人民共和国香港特别行政区维护国家安全法",
                    "中华人民共和国澳门特别行政区基本法", "中华人民共和国村民委员会组织法", "中华人民共和国城市居民委员会组织法", "中华人民共和国国家勋章和国家荣誉称号法",
                    "中华人民共和国国旗法", "中华人民共和国国徽法", "中华人民共和国国歌法", "全国人民代表大会常务委员会关于实行宪法宣誓制度的决定",

                    # 国际法相关的法律法规
                    "中华人民共和国缔结条约程序法", "中华人民共和国领海及毗连区法", "中华人民共和国专属经济区和大陆架法", "中华人民共和国国籍法",
                    "中华人民共和国出境入境管理法", "中华人民共和国引渡法",

                    # 司法制度与法律道德相关的法律法规
                    "中华人民共和国人民法院组织法", "中华人民共和国法官法", "中华人民共和国法官职业道德基本准则", "人民法院工作人员处分条例",
                    "中华人民共和国人民检察院组织法","中华人民共和国检察官法", "中华人民共和国检察官职业道德基本准则","检察人员纪律处分条例",
                    "最高人民法院、最高人民检察院、司法部关于建立健全禁止法官、检察官与律师不正当接触交往制度机制的意见",
                    "最高人民法院、最高人民检察院、司法部关于进一步规范法院、检察院离任人员从事律师职业的意见","中华人民共和国律师法",
                    "律师事务所管理办法", "律师执业管理办法", "律师和律师事务所违法行为处罚办法","律师职业道德基本准则", "律师执业行为规范（试行）",
                    "律师协会会员违规行为处分规则（试行）","中华人民共和国法律援助法", "法律援助值班律师工作办法", "中华人民共和国公证法",
                    "公证机构执业管理办法", "公证员执业管理办法","公证员职业道德基本准则", "中华人民共和国公职人员政务处分法"
                    
                    # 刑法相关的法律法规
                    "中华人民共和国刑法", "中华人民共和国刑法修正案", "中华人民共和国刑法修正案（二）", "中华人民共和国刑法修正案（三）",
                    "中华人民共和国刑法修正案（四）", "中华人民共和国刑法修正案（五）","中华人民共和国刑法修正案（六）", "中华人民共和国刑法修正案（七）",
                    "中华人民共和国刑法修正案（八）", "中华人民共和国刑法修正案（九）","中华人民共和国刑法修正案（十）", "中华人民共和国刑法修正案（十一）",
                    "全国人民代表大会常务委员会关于惩治骗购外汇、逃汇和非法买卖外汇犯罪的决定", "全国人民代表大会常务委员会关于《中华人民共和国刑法》第九十三条第二款的解释",
                    "全国人民代表大会常务委员会关于《中华人民共和国刑法》第二百二十八条、第三百四十二条、第四百一十条的解释", "全国人民代表大会常务委员会关于《中华人民共和国刑法》第二百九十四条第一款的解释",
                    "全国人民代表大会常务委员会关于《中华人民共和国刑法》第三百八十四条第一款的解释", "全国人民代表大会常务委员会关于《中华人民共和国刑法》第三百一十三条的解释",
                    "全国人民代表大会常务委员会关于《中华人民共和国刑法》第九章漾职罪主体适用问题的解释", "全国人民代表大会常务委员会关于《中华人民共和国刑法》有关出口退税、抵扣税款的其他发票规定的解释",
                    "全国人民代表大会常务委员会关于《中华人民共和国刑法》第三十条的解释", "全国人民代表大会常务委员会关于《中华人民共和国刑法》第一百五十八条、第一百五十九条的解释",
                    "全国人民代表大会常务委员会关于《中华人民共和国刑法》第二百六十六条的解释", "全国人民代表大会常务委员会关于《中华人民共和国刑法》第三百四十一条、第三百一十二条的解释",
                    "最高人民法院、最高人民检察院、公安部关于依法适用正当防卫制度的指导意见", "最高人民法院关于适用财产刑若干问题的规定",
                    "最高人民法院关于处理自首和立功具体应用法律若干问题的解释", "最高人民法院关于处理自首和立功若干具体问题的意见",
                    "最高人民法院关于办理减刑、假释案件具体应用法律的规定", "最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释",
                    "最高人民法院、最高人民检察院关于办理生产、销售伪劣商品刑事案件具体应用法律若干问题的解释", "最高人民法院、最高人民检察院关于办理坊害信用卡管理刑事案件具体应用法律若干问题的解释",
                    "最高人民法院关于审理非法集资刑事案件具体应用法律若干问题的解释", "最高人民法院、最高人民检察院关于办理侵犯知识产权刑事案件具体应用法律若干问题的解释",
                    "最高人民法院关于审理拐卖妇女儿童犯罪案件具体应用法律若干问题的解释", "最高人民法院关于审理抢动案件具体应用法律若干问题的解释",
                    "最高人民法院、最高人民检察院关于办理盗窃刑事案件适用法律若干问题的解释", "最高人民法院、最高人民检察院、公安部关于办理盗窃油气、破坏油气设备等刑事案件适用法律若干问题的意见",
                    "最高人民法院、最高人民检察院关于办理诈骗刑事案件具体应用法律若干问题的解释", "最高人民法院、最高人民检察院关于办理抢夺刑事案件适用法律若干问题的解释",
                    "最高人民法院、最高人民检察院关于办理敲诈勒索刑事案件适用法律若干问题的解释", "最高人民法院、最高人民检察院关于办理组织考试作整等刑事案件适用法律若干间题的解释",
                    "最高人民法院、最高人民检察院关于办理寻鲜滋事刑事案件适用法律若干问题的解释", "最高人民法院关于审理黑社会性质组织犯罪的案件具体应用法律若干问题的解释",
                    "最高人民法院、最高人民检察院关于办理赌博刑事案件具体应用法律若干问题的解释", "最高人民法院、最高人民检察院关于办理虚假诉讼刑事案件适用法律若干问题的解释",
                    "最高人民法院、最高人民检察院关于办理窝藏、包底刑事案件适用法律若干问题的解释", "最高人民法院关于审理拖饰、隐睛犯罪所得、犯罪所得收益刑事案件适用法律若干问题的解释",
                    "最高人民法院关于审理非法行医刑事案件具体应用法律若干问题的解释", "最高人民法院、最高人民检察院关于办理环境污染刑事案件适用法律若干问题的解释",
                    "最高人民法院关于审理毒品犯罪案件适用法律若干问题的解释", "最高人民法院、最高人民检察院关于办理组织、强迫、引透、容留、介绍卖浮刑事案件适用法律若干问题的解释",
                    "最高人民法院、最高人民检察院关于办理贪污贴路刑事案件适用法律若干问题的解释", "最高人民法院关于审理挪用公款案件具体应用法律若干问题的解释",
                    "最高人民法院、最高人民检察院关于办理受赔刑事案件适用法律若干问题的意见", "最高人民法院、最高人民检察院关于办理读职刑事案件适用法律若干间题的解释（一）",

                    # 刑事诉讼法相关的法律法规
                    "中华人民共和国刑事诉讼法", "中华人民共和国人民陪审员法", "最高人民法院关于适用《中华人民共和国人民陪审员法》若干问题的解释",
                    "中华人民共和国社区矫正法", "全国人民代表大会常务委员会关于《中华人民共和国刑事诉讼法》第七十九条第三款的解释",
                    "全国人民代表大会常务委员会关于《中华人民共和国刑事诉讼法》第二百五十四条第五款、第二百五十七条第二款的解释",
                    "全国人民代表大会常务委员会关于《中华人民共和国刑事诉论法》第二百七十一条第二款的解释", "最高人民法院、最高人民检察院、公安部、国家安全部、司法部、全国人大常委会法制工作委员会关于实施刑事诉讼法若干问题的规定",
                    "最高人民法院、最高人民检察院、公安部、国家安全部、司法部关于推进以审判为中心的刑事诉讼制度改革的意见", "最高人民法院关于适用《中华人民共和国刑事诉讼法》的解释",
                    "人民检察院刑事诉讼规则", "公安机关办理刑事案件程序规定", "最高人民法院、最高人民检察院、公安部、国家安全部、司法部关于适用认罪认罚从宽制度的指导意见",
                    "人民检察院办理认罪认罚案件开展量刑建议工作的指导意见", "最高人民法院、最高人民检察院、公安部、国家安全部、司法部关于依法保障律师执业权利的规定",
                    "法律援助值班律师工作办法", "最高人民法院、司法部关于为死刑复核案件被告人依法提供法律援助的规定（试行）",
                    "最高人民法院、最高人民检察院、公安部、国家安全部、司法部关于办理刑事案件严格排除非法证据若干问题的规定",
                    "最高人民法院、最高人民检察院、公安部关于办理刑事案件收集提取和审查判断电子数据若干问题的规定",
                    "最高人民法院、最高人民检察院、公安部、国家安全部、司法部关于规范量刑程序若干问题的意见", "最高人民法院关于办理死刑复核案件听取辩护律师意见的办法",
                    "最高人民法院关于死刑复核及执行程序中保障当事人合法权益的若干规定","最高人民法院关于死刑缓期执行限制减刑案件审理程序若干问题的规定",
                    "最高人民法院关于刑事裁判涉财产部分执行的若干规定","最高人民法院关于刑事再审案件开庭审理程序的具体规定（试行）",
                    "最高人民法院关于减刑、假释案件审理程序的规定","人民检察院办理减刑、假释案件规定","最高人民法院、最高人民检察院、公安部关于办理网络犯罪案件适用刑事诉讼程序若干问题的意见",
                    "最高人民法院、最高人民检察院、中国海警局关于海上刑事案件管辖等有关问题的通知","人民检察院办理未成年人刑事案件的规定",
                    "最高人民法院关于建立健全防范刑事免假错案工作机制的意见","最高人民法院、最高人民检察院关于适用犯罪嫌疑人、被告人逃、死亡案件违法所得没收程序若干问题的规定","","","","","","","","","","",

                    # 行政法与行政诉讼法相关的法律法规
                    "中华人民共和国公务员法", "行政机关公务员处分条例", "国务院行政机构设置和编制管理条例", "国务院行政机构设置和编制管理条例",
                    "地方各级人民政府机构设置和编制管理条例","中华人民共和国立法法","行政法规制定程序条例","规章制定程序条例","中华人民共和国行政许可法",
                    "最高人民法院关于审理行政许可案件若干问题的规定","中华人民共和国行政处罚法","中华人民共和国治安管理处罚法","中华人民共和国行政强制法",
                    "中华人民共和国行政复议法","中华人民共和国行政复议法实施条例","中华人民共和国政府信息公开条例","中华人民共和国行政诉讼法",
                    "最高人民法院关于适用《中华人民共和国行政诉讼法》的解释","最高人民法院关于行政诉讼证据若干问题的规定","最高人民法院关于审理国际贸易行政案件若干问题的规定",
                    "最高人民法院关于审理反倾销行政案件应用法律若干问题的规定","最高人民法院关于审理反补贴行政案件应用法律若干间题的规定",
                    "最高人民法院关于印发《关于审理行政案件适用法律规范问题的座谈会纪要》的通知","最高人民法院关于行政诉讼撤诉若干问题的规定",
                    "最高人民法院关于审理行政协议案件若干问题的规定","中华人民共和国国家赔偿法","最高人民法院关于人民法院赔偿委员会审理国家赔偿案件程序的规定",
                    "最高人民法院关于人民法院赔偿委员会适用质证程序审理国家赔偿案件的规定","最高人民法院关于人民法院执行《中华人民共和国国家赔偿法》几个问题的解释",
                    "最高人民法院关于适用《中华人民共和国国家赔偿法》若干问题的解释（一）","最高人民法院关于审理行政赔偿案件若干问题的规定",
                    "最高人民法院关于审理民事、行政诉讼中司法赔偿案件适用法律若干问题的解释","最高人民法院关于审理国家赔偿案件确定精神损害赔偿责任适用法律若干问题的解释",

                    # 民法相关的法律法规
                    "中华人民共和国民法典","中华人民共和国个人信息保护法","最高人民法院关于适用《中华人民共和国民法典》时间效力的若干规定",
                    "最高人民法院关于适用《中华人民共和国民法典》总则编若干问题的解释", "最高人民法院关于审理民事案件适用诉讼时效制度若干问题的规定",
                    "最高人民法院关于适用《中华人民共和国民法典》物权编的解释（一）","最高人民法院关于适用《中华人民共和国民法典》有关担保制度的解释",
                    "最高人民法院关于审理建筑物区分所有权纠纷案件适用法律若干问题的解释","最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释",
                    "最高人民法院关于审理商品房买卖合同纠纷案件适用法律若干问题的解释","最高人民法院关于审理建设工程施工合同纠纷案件适用法律问题的解释（一）",
                    "最高人民法院关于审理技术合同纠纷案件适用法律若干问题的解释","最高人民法院关于审理城镇房屋租赁合同纠纷案件具体应用法律若干问题的解释",
                    "最高人民法院关于审理融资租赁合同纠纷案件适用法律问题的解释","最高人民法院关于审理民间借贷案件适用法律若干问题的规定",
                    "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）","最高人民法院关于适用《中华人民共和国民法典》继承编的解释（一）",
                    "最高人民法院关于确定民事侵权精神损害赔偿责任若干问题的解释","最高人民法院关于审理人身损害赔偿案件适用法律若干问题的解释",
                    "最高人民法院关于审理利用信息网络侵害人身权益民事纠纷案件适用法律若干问题的规定","最高人民法院关于审理道路交通事故损害赔偿案件适用法律若干问题的解释",

                    # 知识产权法相关的法律法规
                    "中华人民共和国著作权法","中华人民共和国著作权法实施条例","最高人民法院关于审理著作权民事纠纷案件适用法律若干问题的解释",
                    "计算机软件保护条例","信息网络传播权保护条例","中华人民共和国专利法","中华人民共和国专利法实施细则","最高人民法院关于审理专利纠纷案件适用法律问题的若干规定",
                    "最高人民法院关于审理侵犯专利权纠纷案件应用法律若干问题的解释","最高人民法院关于审理侵犯专利权纠纷案件应用法律若干问题的解释（二）",
                    "中华人民共和国商标法","中华人民共和国商标法实施条例","最高人民法院关于审理商标民事纠纷案件适用法律若干问题的解释",
                    "最高人民法院关于审理涉及驰名商标保护的民事纠纷案件应用法律若干问题的解释",

                    # 商法相关的法律法规
                    "中华人民共和国公司法","最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（一）","最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（二）",
                    "最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（三）","最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（四）",
                    "最高人民法院关于适用《中华人民共和国公司法》若干问题的规定（五）","中华人民共和国个人独资企业法","中华人民共和国外商投资法",
                    "中华人民共和国企业破产法","最高人民法院关于适用《中华人民共和国企业破产法》若干问题的规定（一）","最高人民法院关于适用《中华人民共和国企业破产法》若干问题的规定（二）",
                    "最高人民法院关于适用《中华人民共和国企业破产法》若干问题的规定（三）","中华人民共和国票据法","中华人民共和国证券法",
                    "中华人民共和国证券投资基金法","中华人民共和国保险法","最高人民法院关于适用《中华人民共和国保险法》若干问题的解释（一）",
                    "最高人民法院关于适用《中华人民共和国保险法》若干问题的解释（二）","最高人民法院关于适用《中华人民共和国保险法》若干问题的解释（三）",
                    "最高人民法院关于适用《中华人民共和国保险法》若干问题的解释（四）","中华人民共和国海商法","中华人民共和国信托法",

                    # 经济法相关的法律法规
                    "中华人民共和国反药断法","中华人民共和国反不正当竞争法","最高人民法院关于适用《中华人民共和国反不正当竞争法》若干问题的解释",
                    "最高人民法院关于审理侵犯商业秘密民事案件适用法律若干问题的规定","中华人民共和国消费者权益保护法","中华人民共和国产品质量法",
                    "中华人民共和国食品安全法","最高人民法院关于审理食品药品纠纷案件适用法律若干问题的规定","中华人民共和国商业银行法",
                    "中华人民共和国银行业监督管理法","中华人民共和国税收征收管理法","中华人民共和国税收征收管理法实施细则","中华人民共和国个人所得税法",
                    "中华人民共和国企业所得税法","中华人民共和国车船税法","中华人民共和国审计法","中华人民共和国土地管理法","中华人民共和国城市房地产管理法",
                    "中华人民共和国城乡规划法","不动产登记暂行条例",

                    # 环境资源法相关的法律法规
                    "中华人民共和国环境保护法","中华人民共和国环境影响评价法","中华人民共和国森林法","中华人民共和国矿产资源法",

                    # 劳动与社会保障法相关的法律法规
                    "中华人民共和国劳动法","中华人民共和国劳动合同法","中华人民共和国劳动合同法实施条例","中华人民共和国劳动争议调解仲裁法",
                    "中华人民共和国社会保险法","中华人民共和国军人保险法",

                    # 国际私法相关的法律法规
                    "中华人民共和国涉外民事关系法律适用法", "最高人民法院关于适用《中华人民共和国涉外民事关系法律适用法》若干问题的解释（一）",
                    "最高人民法院关于适用《中华人民共和国民事诉讼法》的解释","外国人在中华人民共和国收养子女登记办法","最高人民法院关于涉外民商事案件诉讼管辖若干问题的规定",
                    "关于向国外送达民事或商事司法文书和司法外文书公约","关于从国外调取民事或商事证据的公约","跨国收养方面保护儿童及合作公约",
                    "最高人民法院、外交部、司法部关于执行《关于向国外送达民事或商事司法文书和司法外文书公约》有关程序的通知",
                    "最高人民法院、外交部、司法部关于我国法院和外国法院通过外交途径相互委托送达法律文书若干问题的通知",
                    "最高人民法院关于涉外民事或商事案件司法文书送达问题若干规定","最高人民法院关于依据国际公药和双边司法协助条约办理民商事案件司法文书送达和调查取证司法协助请求的规定",
                    "最高人民法院关于中国公民申请承认外国法院离婚判决程序问题的规定","承认及执行外国仲裁裁决公约","最高人民法院关于执行我国加入的《承认及执行外国仲裁裁决公约》的通知",
                    "最高人民法院关于人民法院处理与涉外仲裁及外国仲裁事项有关问题的通知","最高人民法院关于适用《中华人民共和国仲裁法》若干问题的解释",
                    "最高人民法院关于内地与香港特别行政区法院相互委托送达民商事司法文书的安排","最高人民法院关于内地与香港特别行政区法院就仲裁程序相互协助保全的安排",
                    "最高人民法院关于内地与澳门特别行政区就仲裁程序相互协助保全的安排","最高人民法院关于内地与香港特别行政区相互执行仲裁裁决的安排",
                    "最高人民法院关于内地与香港特别行政区相互执行仲裁裁决的补充安排","最高人民法院、香港特别行政区政府关于内地与香港特别行政区法院相互认可和执行民商事案件判决的安排",
                    "最高人民法院关于内地与香港特别行政区法院相互认可和执行婚姻家庭民事案件判决的安排","最高人民法院关于内地与香港特别行政区法院就民商事案件相互委托提取证据的安排",
                    "最高人民法院关于内地与澳门特别行政区法院就民商事案件相互委托送达司法文书和调取证据的安排","最高人民法院关于沙港澳民商事案件司法文书送达问题若干规定",
                    "最高人民法院关于内地与澳门特别行政区相互认可和执行民商事判决的安排","最高人民法院关于内地与澳门特别行政区相互认可和执行仲裁裁决的安排",
                    "最高人民法院关于认可和执行台湾地区法院民事判决的规定","最高人民法院关于认可和执行台湾地区仲裁裁决的规定","最高人民法院关于涉台民事诉讼文书送达的若干规定",
                    "最高人民法院关于审理涉台民商事案件法律适用问题的规定","最高人民法院关于人民法院受理涉及特权与豁免的民事案件有关问题的通知",
                    "最高人民法院关于仲裁司法审查案件报核问题的有关规定", "最高人民法院关于审理仲裁司法审查案件若干问题的规定","最高人民法院关于设立国际商事法庭若干问题的规定",

                    # 国际经济法相关的法律法规
                    "联合国国际货物销售合同公约","国际贸易术语解释通则2020","ICC跟单信用证统一惯例（UCP600）","最高人民法院关于审理信用证纠纷案件若干问题的规定",
                    "最高人民法院关于审理无正本提单交付货物案件适用法律若干问题的规定","国际商会托收统一规则","中华人民共和国对外贸易法",
                    "中华人民共和国出口管制法","中华人民共和国反倾销条例","中华人民共和国反补贴条例","中华人民共和国保障措施条例","中华人民共和国外商投资法",
                    "中华人民共和国外商投资法实施条例","关于解决国家和他国国民之间投资争端公约","最高人民法院关于审理独立保函纠纷案件若干问题的规定",

                    # 民事诉讼法与仲裁制度相关的法律法规
                    "中华人民共和国民事诉讼法", "最高人民法院关于适用《中华人民共和国民事诉讼法》的解释", "最高人民法院关于互联网法院审理案件若干问题的规定",
                    "最高人民法院关于审理民事级别管辖异议案件若干问题的规定", "最高人民法院关于民事诉讼证据的若干规定", "最高人民法院关于在审理经济纠纷案件中涉及经济犯罪嫌疑若干问题的规定",
                    "中华人民共和国人民调解法", "最高人民法院关于人民法院民事调解工作若干问题的规定", "最高人民法院关于适用简易程序审理民事案件的若干规定",
                    "最高人民法院关于适用《中华人民共和国民事诉讼法》审判监督程序若干问题的解释", "最高人民法院关于适用《中华人民共和国民事诉讼法》执行程序若干问题的解释",
                    "最高人民法院关于人民法院民事执行中查封、扣押、冻结财产的规定", "最高人民法院关于人民法院民事执行中拍卖、变卖财产的规定",
                    "最高人民法院关于人民法院办理执行异议和复议案件若干问题的规定", "最高人民法院关于执行和解若干问题的规定", "最高人民法院关于人民法院执行工作若干问题的规定（试行）",
                    "中华人民共和国仲裁法", "最高人民法院关于适用《中华人民共和国仲裁法》若干问题的解释"
                    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exam_mode",
        type=str,
        default="professional_morality",
        help="choose the mode from [law, professional_morality]"
    )
    parser.add_argument(
        "--input_data_root_path",
        type=str,
        default = "../data/retrieval_raw_data/{}_rules",
        # default="/Volumes/mac_extend/safety_code/safe_final_last_local/data/retrieval_raw_data/{}_data",
        help="law_data folder store the data collected from https://flk.npc.gov.cn/, rule_data folder store the data"
             "collected from https://www.gov.cn/zhengce/xxgk/gjgzk/ssgz.htm, and morality_data folder store the data "
             "collected from https://basic.smartedu.cn/."
    )
    parser.add_argument(
        "--extracted_principle_save_path",
        type=str,
        default="../data/sampled_data_for_question_generation/extracted_{exam_mode}_data_for_question_generation_test.jsonl"
    )
    parser.add_argument(
        "--sample_principle_num",
        type=int,
        default=300, # law: 1000, morality: 300
        help="the number of generated questions"
    )
    parser.add_argument(
        "--principle_token_max",
        type=int,
        default=2000,
        help="the number of tokens for the principles, including the randomly selected principle and retrieved similar principles"
    )


    parser.add_argument(
        "--selected_text_num",
        type=int,
        default=4, # law: 4, morality: 2
        help="the number of selected texts in each principle"
    )
    parser.add_argument(
        "--province2law_file_path",
        type=str,
        default="../data/retrieval_location_info/projection_province2file.json"
    )

    args = parser.parse_args()
    main(args)
