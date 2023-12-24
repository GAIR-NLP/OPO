import os

import pandas as pd
def isNaN(num):
    return num != num

# question,A,B,C,D,answer,analysis,action,principle
folder = "data/exam_questions"
files = [file  for file in os.listdir(folder) if file.endswith(".csv")]
for file in files:
    file_path = os.path.join(folder, file)
    data = pd.read_csv(file_path)
    for row_index, row in data.iterrows():
        question = row["question"]
        option_a = row["A"]
        option_b = row["B"]
        option_c = row["C"]
        option_d = row["D"]
        answer = row["answer"]
        analysis = row["analysis"]
        if isNaN(question) or isNaN(option_a) or isNaN(option_b) or isNaN(option_c) or \
            isNaN(option_d) or isNaN(answer) or isNaN(analysis):
            print(file_path, row)


