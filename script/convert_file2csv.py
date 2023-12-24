import os
import csv
from script.gpt_usage import chunks
import jsonlines

def convert_txt2csv(file_path):

    # load data
    with open(file_path, 'r') as f:
        example_data = f.readlines()
    example_data = chunks(example_data, 11)  # 11 means there are 11 lines for each example in the txt file
    new_format_data = []
    if "law" in file_path:
        for id, item in enumerate(example_data):
            print(item)
            principle = item[0].strip().split('法律:')[1].replace("\u3000", "  ")
            question = item[1].strip().split('题干:')[1]
            options = [option.strip() for option in item[3:7]]
            analysis = item[7].strip().split("选项分析:")[1]
            answer = item[8].strip().split("答案:")[1]
            label = item[9].strip().split("label:")[1]
            if label == "1":
                action = "设计一个法律情境"
            else:
                action = "针对给定法律中的某个概念"
            new_data = {"id": id, "principle": principle, "question": question, "A": options[0], "B": options[1],
                        "C": options[2], "D": options[3], "analysis": analysis, "answer": answer, "action": action}
            new_format_data.append(new_data)
    else:
        for id, item in enumerate(example_data):
            principle = item[0].strip().split('道德准则:')[1].replace("\u3000", "  ")
            question = item[1].strip().split('题干:')[1]
            options = [option.strip() for option in item[3:7]]
            analysis = item[7].strip().split("选项分析:")[1]
            answer = item[8].strip().split("答案:")[1]
            label = item[9].strip().split("label:")[1]
            if label == "1":
                action = "设计一个道德情境"
            else:
                action = "针对给定道德准则中的某个概念"
            new_data = {"id":id, "principle": principle, "question": question, "A": options[0], "B": options[1],
                        "C": options[2], "D": options[3], "analysis": analysis, "answer": answer, "action": action}
            new_format_data.append(new_data)
    # convert data
    header_list = ["id", "question", "A", "B", "C", "D", "answer", "analysis", "action", "principle"]
    with open(file_path.replace("txt","csv"), mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, header_list)
        writer.writeheader()
        writer.writerows(new_format_data)


def convert_jsonl2csv(file_path):

    # load data
    with jsonlines.open(file_path, 'r') as f:
        example_data = [obj for obj in f]
    new_format_data = []
    for id, item in enumerate(example_data):
        question = item["question"].strip()
        options = item["options"]
        assert "A." in options[0]
        assert "B." in options[1]
        assert "C." in options[2]
        assert "D." in options[3]
        option_a = options[0].replace("A.", "")

        option_b = options[1].replace("B.", "")
        option_c = options[2].replace("C.", "")
        option_d = options[3].replace("D.", "")
        answer = item["answer"]
        analysis = item["analysis"]
        if question=="" or option_a=="" or option_b=="" or option_c=="" or option_d=="" \
                or  answer=="" or analysis=="":
            print(item)
            continue

        new_data = {"question": question, "A": option_a, "B": option_b,
                    "C": option_c, "D": option_d, "analysis": analysis, "answer": answer}
        new_format_data.append(new_data)

    # split the data into val and test based on the ratio 1:9
    val_num = int(len(new_format_data)*0.1)
    val_data = new_format_data[:val_num]
    # add id
    for idx in range(len(val_data)):
        val_data[idx]["id"] = idx+1


    test_data = new_format_data[val_num:]
    for idx in range(len(test_data)):
        test_data[idx]["id"] = idx+1


    # convert data
    header_list = ["id", "question", "A", "B", "C", "D", "answer", "analysis", "action", "principle"]
    save_path = file_path.replace("jsonl","csv").split('/')
    save_path[-1] = "val_"+save_path[-1]
    with open('/'.join(save_path), mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, header_list)
        writer.writeheader()
        writer.writerows(val_data)

    save_path = file_path.replace("jsonl", "csv").split('/')
    save_path[-1] = "test_" + save_path[-1]
    with open('/'.join(save_path), mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, header_list)
        writer.writeheader()
        writer.writerows(test_data)





if __name__ == "__main__":
    # convert example data
    # example_txt_folder = "../data/example_data"
    # files = [file for file in os.listdir(example_txt_folder) if file.endswith("txt")]
    # for file in files:
    #     file_path = os.path.join(example_txt_folder, file)
    #     convert_txt2csv(file_path)

    # convert question data
    example_txt_folder = "../data/exam_questions"
    files = [file for file in os.listdir(example_txt_folder) if file.endswith("jsonl")]
    for file in files:
        file_path = os.path.join(example_txt_folder, file)
        convert_jsonl2csv(file_path)