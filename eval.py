# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import pandas as pd
# import torch
import json
from evaluation.weight_model_evaluation import Weight_Model_Evaluator
from evaluation.gpt_evaluation import GPT_Evaluator

import time


def main(args, evaluator, take=1):
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    # set the sava path
    save_path = args.evaluation_result_save_path
    if args.do_save_csv:
        sava_path = save_path.replace("{take}", f"take{take}") \
            .replace("{evaluation_model}", args.evaluation_model) \
            .replace("{question_mode}", args.question_mode) \
            .replace("{exam_mode}", args.exam_mode) \
            .replace("{date}", run_date) \
            .replace("{retrieval_token_max}", str(args.retrieval_token_max))

        if args.use_options:
            sava_path = sava_path.replace("{use_options}", "use_options")
        else:
            sava_path = sava_path.replace("_{use_options}", "")
        if args.use_retrieval:
            sava_path = sava_path.replace("{use_retrieval}", "use_retrieval")
        else:
            sava_path = sava_path.replace("_{use_retrieval}", "")
        if args.use_note:
            sava_path = sava_path.replace("{use_note}", "use_note")
        else:
            sava_path = sava_path.replace("_{use_note}", "")
        sava_path = sava_path
        if not os.path.exists('/'.join(sava_path.split("/")[:-1])):
            os.makedirs('/'.join(sava_path.split("/")[:-1]), exist_ok=True)

    print(
        f"Inference starts at {run_date} on {args.evaluation_model} with {args.question_mode} {args.exam_mode} questions!")

    val_file_path = args.input_question_path.replace("{split}", "val") \
                        .replace("{question_mode}", args.question_mode) \
                        .replace("{exam_mode}", args.exam_mode)
    # dev_file_path = args.input_question_path.replace("{split}", "dev") \
    #                     .replace("{question_mode}", args.question_mode) \
    #                     .replace("{exam_mode}", args.exam_mode)
    test_file_path = args.input_question_path.replace("{split}", "test") \
                        .replace("{question_mode}", args.question_mode) \
                        .replace("{exam_mode}", args.exam_mode)

    val_df = pd.read_csv(val_file_path) if args.do_test is False else pd.read_csv(test_file_path)
    # dev_df = pd.read_csv(dev_file_path) if args.few_shot else None
    correct_ratio, answers = evaluator.eval(val_df,
                                                    save_result_dir=sava_path if args.do_save_csv else None,
                                                    few_shot=args.few_shot,
                                                    cot=args.cot,
                                                    use_retrieval=args.use_retrieval,
                                                    use_options=args.use_options,
                                                    use_note=args.use_note
                                                    )
    print(f"Exams: {args.question_mode} {args.exam_mode}")
    print(f"Acc: {correct_ratio}")
    summary = {"score": correct_ratio,
                "num": len(val_df),
                "correct": correct_ratio * len(val_df) / 100}
    json.dump(summary,open(sava_path.replace(".csv", "_summary.json"),'w'),ensure_ascii=False,indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation_model",
        type=str,
        default="Qwen-7B-Chat",
        help="the model to be evaluated, select from "
             "gpt-3.5-turbo-0613, gpt-4-0613,"
             "THUDM/chatglm2-6b, THUDM/chatglm3-6b"
             "internlm/internlm-chat-7b, internlm/internlm-chat-20b,"
             "Qwen/Qwen-7B-Chat, Qwen/Qwen-14B-Chat,"
             "xverse/XVERSE-7B-Chat, xverse/XVERSE-13B-Chat,"
             "ShengbinYue/DISC-LawLLM,"
             "Duxiaoman-DI/XuanYuan-70B"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",  # ["gpt-4", "gpt-3.5-turbo"]
        help="path to the model to be evaluated. we need to set it for weight model"

    )
    # input setting
    # human_annotated: [law, morality, social_norm], and machine_generated: [law, morality]
    parser.add_argument(
        "--exam_mode",
        type=str,
        default="law",
        help="choose the mode from [law, basic_morality, professional_morality, social_morality]"
    )
    parser.add_argument(
        "--question_mode",
        type=str,
        default="human_annotated",
        help="choose the question mode from [human_annotated, machine_generated]"
    )
    parser.add_argument(
        "--input_question_path",
        type=str,
        default="./data/exam_questions/{split}_questions_{question_mode}_{exam_mode}.csv",
        help="the file path which stores the questions"
    )

    # retrieval setting
    parser.add_argument(
        "--question_embedding_folder",
        type=str,
        default="./data/questions_embedding",
        help="the file path which stores the questions"
    )
    # TODO: modify the path here
    parser.add_argument(
        "--input_retrieval_text_embed_file",
        type=str,
        default="./data/retrieval_processed_embed_text/{exam_mode}_embed_text_pairs.pkl",
    )
    parser.add_argument(
        "--retrieval_result_folder",
        type=str,
        default="./data/retrieval_results",
        help="the file path which stores the retrieval results. If this file doesn't exist, we would retrieve "
             "and save the results to this path. Otherwise, we would reuse the file"
    )
    parser.add_argument(
        "--gpu_retrieval_id",
        type=str,
        default='0',
        help="the gpu id for retrieval. '-1' means we don't use GPU for retrieval"
    )
    # parser.add_argument(
    #     "--gpu_model_id",
    #     type=str,
    #     default='1',
    #     help="the gpu for loading model. -1 means we don't use GPU"
    # )
    parser.add_argument(
        "--retrieval_doc_num",
        type=int,
        default=200,  # 10 for law and 5 for morality
        help="the number of retrieved docs"
    )
    parser.add_argument(
        "--retrieval_token_max",
        type=int,
        default=1000,
        help="the number of retrieved tokens"
    )

    # only for law
    parser.add_argument(
        "--source_file",
        type=str,
        default="./data/retrieval_source_info/law_source.json",
        help="the file which contains the meta information of the embeddings"
    )
    # only for law
    parser.add_argument(
        "--province2file_input_path",
        type=str,
        default="./data/retrieval_location_info/projection_province2file.json",
        help="the file which contains the information of projecting province to file"
    )

    # parameter setting
    parser.add_argument(
        "--temperature",
        type=float,
        default=0
    )
    parser.add_argument(
        "--use_retrieval",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use retrieval when answer the question"
    )

    parser.add_argument(
        "--use_note",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use note in the eval_user_prompt"  # i.e., 注意：因为辅助信息有限.....
    )
    parser.add_argument(
        "--use_options",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use options for retrieval besides the question stem"
    )
    parser.add_argument(
        "--cot",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--few_shot",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--constrained_decoding",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    # save setting
    parser.add_argument(
        "--evaluation_result_save_path",
        type=str,
        default="./data/experimental_results/eval_{question_mode}/eval_{exam_mode}/{evaluation_model}/{date}/result_{take}_{question_mode}_{exam_mode}_{use_options}_{use_retrieval}_{use_note}_{retrieval_token_max}.csv",
        help="the file path which stores the evaluation results for the generated questions"
    )

    parser.add_argument(
        "--ntrain",
        "-k",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=1
    )
    parser.add_argument(
        "--do_test",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--do_save_csv",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()
    if args.question_mode == "machine_generated":
        assert args.exam_mode in ["law", "professional_morality"]
    else:
        assert args.exam_mode in ["law", "basic_morality", "social_morality"]
    if "gpt-3.5" in args.evaluation_model or "gpt-4" in args.evaluation_model:
        args.constrained_decoding = False
        evaluator = GPT_Evaluator(args)
    else:
        args.constrained_decoding = True
        evaluator = Weight_Model_Evaluator(args)
    main(args, evaluator=evaluator)
