import json
import os
from collections import defaultdict
import argparse
def project_file2province(province2file_file_path):
    law2province = {}
    province2file_data = json.load(open(province2file_file_path, "r"))
    for province, data in province2file_data.items():
        if isinstance(data, list):  # for example: "全国": [....]
            for item in data:
                law2province[item] = {"province": province}
        else:
            for city, values in data.items():  # for example: "湖南": {"省直":[], "长沙":[]}
                for value in values:
                    law2province[value] = {"province": province, "city": city}
    return law2province



def project_province2index(province2file_input_path, source_file):
    # sources contain the order of all texts in the file. The order is also consistent with the embeddings
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

def main(args):
    sources = json.load(open(args.source_file, "r"))

    province2file = json.load(open(args.province2file_input_path, "r"))
    province2index = project_province2index(province2file, sources)
    # province2index_save_path = "./data/retrieval_info/projection_province2index"
    # json.dump(province2index, open(province2index_save_path, "w", encoding="utf8"), ensure_ascii=False)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_file",
        type=str,
        default="../construct_embeddings/file_sources/law_rule/source_all.json",
    )
    parser.add_argument(
        "--province2file_input_path",
        type=str,
        default="../data/retrieval_info/projection_province2file.json",
    )
    parser.add_argument(
        "--province2index_save_path",
        type=str,
        default="../data/retrieval_info/projection_province2index.json",
    )

    args = parser.parse_args()
    main(args)







