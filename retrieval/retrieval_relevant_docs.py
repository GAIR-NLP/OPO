# -*- coding:utf-8 -*-

import sys
sys.path.append('../')
import re
import pickle
import os
import jsonlines
import asyncio
from tqdm import tqdm
from construct_embeddings.construct_embedding import chunks,truncate_text_tokens
import argparse
import faiss
from script.city2province import city2province, province_all
from script.project_province_file import project_province2index
import numpy as np
import time
import json
from collections import defaultdict
import re

def extract_principles_from_oracle(text):
    start_name = text.split("规定:", maxsplit=1)[0]
    rule_content = text.split("规定:", maxsplit=1)[1]
    pattern = "<(.*?)>"
    rules = re.findall(pattern, rule_content)
    rules = [start_name+rule for rule in rules]
    return rules

# text = "青海省人民政府规章制定办法规定:<第二十三条:省政府法制机构在审查过程中，发现规章送审稿存有下列情形之一的，可予以缓办或者退回责任单位:(一)制定条件发生重大变化或者制定时机尚不成熟的;(二)不符合立法技术规范基本要求，需作全面调整或者修改的;(三)违反相关规定增加部门权力和部门利益，需要作重大修改的;(四)重大问题协调不一致的;(五)不符合本办法第十九条规定，需要补充相关内容材料的;(六)其他不宜制定规章的情形。> <第三十二条:规章规定需要进一步明确具体含义，或者制定后出现新的情况需要明确适用规章依据的，由省政府法制机构提出意见，报请省政府批准后向社会公布。规章的解释同规章具有同等效力。> <第三十四条:实施部门应当根据经济社会发展需要对规章不定期进行清理，向省政府法制机构提出清理的建议。必要时，省政府法制机构可以组织有关实施部门进行清理。> <第三十五条:规章有下列情形之一的，实施部门或者省政府法制机构应当及时向省政府提出修改、废止的建议:(一)与法律、法规或者国家新制定的方针、政策相抵触的;(二)所依据的法律、法规已经修改或废止的;(三)已经被新颁布的法律、法规、规章取代的;(四)调整对象已经消失或者发生变化的;(五)实施主体发生变化的;(六)国务院及国家有关部门提出了专项清理要求;(七)其他应当进行修改、废止的情形。> "
# extract_principles_from_oracle(text)

class Retrieval:
    def __init__(self, embed_model, args):
        self.embed_model = embed_model
        self.args = args


        # set the question embedding save path and retrieval results save path
        if args.use_options:
            self.question_embedding_save_path = os.path.join(args.question_embedding_folder,
                                                        "question_embed_{question_mode}_{exam_mode}_with_options.jsonl".
                                                        replace("{question_mode}", args.question_mode).
                                                        replace("{exam_mode}", args.exam_mode))
            self.retrieval_save_path = os.path.join(args.retrieval_result_folder,
                                               "retrieval_{exam_mode}_{question_mode}_with_options.jsonl".
                                               replace("{exam_mode}", args.exam_mode).
                                               replace("{question_mode}", args.question_mode))
        else:
            self.question_embedding_save_path = os.path.join(args.question_embedding_folder,
                                                        "question_embed_{question_mode}_{exam_mode}_wo_options.jsonl".
                                                        replace("{question_mode}", args.question_mode).
                                                        replace("{exam_mode}", args.exam_mode))
            self.retrieval_save_path = os.path.join(args.retrieval_result_folder,
                                               "retrieval_{question_mode}_{exam_mode}_wo_options.jsonl".
                                               replace("{exam_mode}", args.exam_mode).
                                               replace("{question_mode}", args.question_mode))

    def embed_query(self, questions):
        text_per_batch_num = min(200, len(questions))
        ques_all = truncate_text_tokens(questions, encoding_name='cl100k_base', max_tokens=8000)
        ques_batch = chunks(ques_all, text_per_batch_num)
        ques_embed_all = []
        for cnt, ques in tqdm(enumerate(ques_batch), total=len(ques_all) / text_per_batch_num):
            cur_embeddings = asyncio.run(self.embed_model.async_run(
                messages_list=ques,
                mode="embed"
            ))
            ques_embed_all.extend(cur_embeddings)

        text_embed_pairs = list(zip(questions, ques_embed_all))
        if not os.path.exists('/'.join(self.question_embedding_save_path.split('/')[:-1])):
            os.makedirs('/'.join(self.question_embedding_save_path.split('/')[:-1]))
        with open(self.question_embedding_save_path, "wb") as f:
            pickle.dump(text_embed_pairs, f)

    def detect_question_location(self, questions, province_all):
        question_location_all = []
        for question in questions:
            cur_ques_location = ["全国"]  # the default location should contain "全国"
            for province in province_all:
                location = re.findall(province, question)
                if len(location):
                    cur_ques_location.extend(location)
            question_location_all.append(set(cur_ques_location))
        return question_location_all

    def project_province2index(self, province2file_input_path, source_file):
        # source_file contain the order of all texts. The order is also consistent with the embeddings
        # province->file->text index
        # align each law with related province
        sources = json.load(open(source_file, "r"))

        province2file = json.load(open(province2file_input_path, "r"))

        file2province = {}
        for province, data in province2file.items():
            if isinstance(data, list):  # for example: "全国": [....]
                for item in data:
                    file2province[item] = province
            else:
                for _, values in data.items():  # for example: "湖南": {"省直":[], "长沙":[]}
                    for value in values:
                        file2province[value] = province

        province2index = defaultdict(list)
        # align each source with related province based on law2province
        for idx, source in enumerate(sources):
            source_law = source.split("/")[1]
            related_prov = file2province[source_law]
            province2index[related_prov].append(idx)
        return province2index

    def similarity_search(self, source_text_embed_pairs, query_text_embed_pairs,exam_mode, question2index=None,
                          retrieval_doc_num=10, gpu_id=-1, normalize_L2=False):
        """
        source_text_embed_pairs: [(source_text_1, source_embed_1), .....]
        query_text_embed_pairs: [(query_text_1, query_embed_1), .....]
        question2index: project each question to the embedding index, [[], []....]
        gpu_id: gpu to conduct the retrieval
        retrieval_doc_num: the num of retrieved similar documents
        """
        # set the retriever
        source_embeddings, source_texts = [], []
        for text_embed_pair in source_text_embed_pairs:
            source_embeddings.append(text_embed_pair[1])
            source_texts.append(text_embed_pair[0])
        query_embeddings, query_texts = [], []
        for text_embed_pair in query_text_embed_pairs:
            query_embeddings.append(text_embed_pair[1])
            query_texts.append(text_embed_pair[0])
        embed_dim = len(source_embeddings[0])
        index = faiss.IndexFlatL2(embed_dim)
        if int(gpu_id) >= 0:  # use GPU for retrieval
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, int(gpu_id), index)

        source_vector = np.array(source_embeddings, dtype=np.float32)
        if normalize_L2:
            faiss.normalize_L2(source_vector)
        index.add(source_vector)
        query_vector = np.array(query_embeddings, dtype=np.float32)

        time_start = time.time()
        # retrieval similar docs from all sources
        out_distances, out_indexes = index.search(query_vector, embed_dim)

        # TODO: improve the retrieval efficiency
        # https://github.com/facebookresearch/faiss/wiki/FAQ#is-it-possible-to-dynamically-exclude-vectors-based-on-some-criterion
        if exam_mode=="law":
            # filter the retrieved texts based on the location
            last_retrieval_indexes = []
            for i, out_index in enumerate(out_indexes):
                cur_retrieval_indexes = []
                cur_question_location_idx = set(question2index[i])  # use set for improving search speed
                for idx in out_index:
                    if len(cur_retrieval_indexes) > retrieval_doc_num:  # break when we collect enough index
                        break
                    else:
                        if idx in cur_question_location_idx:
                            cur_retrieval_indexes.append(idx)  # save the idx of specifical location
                last_retrieval_indexes.append(cur_retrieval_indexes)
        else:
            last_retrieval_indexes = out_indexes[:, :retrieval_doc_num]

        # obtain the text related to the idx
        last_retrieval_texts = [[source_texts[index] for index in indexes] for indexes in last_retrieval_indexes]

        time_end = time.time()
        print("The time of retrieving {} questions from the {} retrieval data is {}".format(
            len(query_embeddings), len(source_embeddings), time_end - time_start))
        return last_retrieval_texts

    def retrieve(self, questions):

        # if there are no ques_embedding_save_path, we need to embed the questions first
        if not os.path.exists(self.question_embedding_save_path):
            self.embed_query(questions)

        # if there exists the retrieval_save_path, we directly return the results
        if os.path.exists(self.retrieval_save_path):
            with jsonlines.open(self.retrieval_save_path, "r") as reader:
                retrieval_texts = [obj for obj in reader]
            return retrieval_texts

        if self.args.exam_mode=="law":
            # obtain the question locations
            question_locations = self.detect_question_location(questions, province_all)
            # obtain the projection of province to embedding index
            province2index = self.project_province2index(self.args.province2file_input_path, self.args.source_file)

            question2index = []
            for question_location in question_locations:
                cur_index = []
                for loca in question_location:
                    cur_index.extend(province2index[loca])
                question2index.append(cur_index)
        else:
            question2index = None  # don't need to extract location for morality questions
        # search similar docs from the constructed retrieval data
        with open(self.question_embedding_save_path, "rb") as f:
            ques_text_embed_pairs = pickle.load(f)
        # load retrieval source data
        # if self.args.exam_mode=="morality" and self.args.question_mode=="machine_generated":
        #     with open(self.args.input_retrieval_text_embed_file.replace("{exam_mode}", "professional_morality"), "rb") as f:
        #         retrieval_source_text_embed_pairs = pickle.load(f)
        # else:
        with open(self.args.input_retrieval_text_embed_file.replace("{exam_mode}", self.args.exam_mode), "rb") as f:
            retrieval_source_text_embed_pairs = pickle.load(f)

        retrieval_texts = self.similarity_search(retrieval_source_text_embed_pairs, ques_text_embed_pairs,
                                                 self.args.exam_mode, question2index,
                                                    self.args.retrieval_doc_num, self.args.gpu_retrieval_id)

        # save the retrieval results
        if not os.path.exists('/'.join(self.retrieval_save_path.split('/')[:-1])):
            os.makedirs('/'.join(self.retrieval_save_path.split('/')[:-1]))
        with jsonlines.open(self.retrieval_save_path, "w") as writer:
            writer.write_all(retrieval_texts)
        return retrieval_texts
