import pandas as pd

def rule_check(file, save_path, mode="law"):
    data = pd.read_csv(file, keep_default_na=False)
    reserve_idx = []
    for idx, item in data.iterrows():
        question_stem = item["question"]
        option_a = item["A"]
        option_b = item["B"]
        option_c = item["C"]
        option_d = item["D"]
        answer = item["answer"]
        analysis = item["analysis"]

        if question_stem!="" and option_a!="" and option_b!="" and option_c!= "" and option_d!="" and answer!="" and analysis!="":
            # step 2: check the answer
            if len(answer)==1 and answer in "ABCD":
                reserve_idx.append(idx)
            else:
                print("reason: option, remove question {}".format(idx))
                print(item)
        else:
            print("reason: nan, remove question {}".format(idx))
            print(item)
    data_ = data.iloc[reserve_idx]
    save_path = save_path.format(mode)
    data_.to_csv(save_path, index=False)

# use rule to further check the format
if __name__ == "__main__":
    law_gpt_perfect_file = "./generated_questions_data/machine_generated_law_perfect_questions.csv"
    professional_morality_gpt_perfect_file = "./generated_questions_data/machine_generated_professional_morality_perfect_questions.csv"
    save_path_file = "./generated_questions_data/machine_generated_{}_perfect_re_questions.csv"
    rule_check(law_gpt_perfect_file, save_path_file, "law")
    rule_check(professional_morality_gpt_perfect_file, save_path_file, "professional_morality")




