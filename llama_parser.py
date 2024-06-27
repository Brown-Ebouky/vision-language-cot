import json
from utils_data import load_data_img
import argparse
import random
import re
import os
from utils_evaluate import get_scores
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE', "QCMU-E", "QCMU-UE", "QCMUE-A", "QCMUG-A"])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--load_ua', action='store_true', help='wether to load ua')
    parser.add_argument('--method', type=str, default="ua", help='whether to load ua or predictions ("preds")', choices=['ua', 'preds'])
    parser.add_argument('--sampling_true_percentage', type=float, default=None, help='Percentage to sample',)


    args = parser.parse_args()
    return args

def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED" 
        return answer  

def main(args):
    with open("/dccstor/niccoloav/mm-cot/llama_qcmg_a_70b_seed_5.json") as f:
        data = json.load(f)

    count_parse_problems = 0
    parsed_dict = {"preds" : []}

    correct_answer_parser = True
    none_of_the_above_parser = True

    for s in data["preds"]:
        if correct_answer_parser:
            s = s.replace("The correct answer is ", "")
        if none_of_the_above_parser:
            s = s.replace("None of the above", "")
        ans = s[:17]
        ans = ans.replace("(", "")
        ans = ans.replace("\n", "")
        ans = ans.replace(")", "")
        ans = ans.replace("The answer is", "")
        ans = ans.strip()
        if len(ans) != 1:
            count_parse_problems += 1
            print(ans)
            try:
                ans = ans[0]
            except:
                print(ans)
        parsed_dict["preds"].append(f"The answer is ({ans}).")

    with open('/dccstor/niccoloav/mm-cot/parsed_llama_qcma.json', 'w') as f:
        json.dump(parsed_dict, f, indent=4)
    print(f"The Number of weird parsing is: {count_parse_problems}")

    with open("/dccstor/niccoloav/mm-cot/models/mm-cot-base-ans/predictions_ans_test.json") as lb:
        dat = json.load(lb)
        targets = dat["labels"]

    results_ans = {}
    results_rationale = {}
    results_reference = {}
    preds = parsed_dict["preds"]
    problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
    # dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    test_qids = qids['test']
    num_fail = 0
    for idx, qid in enumerate(test_qids):
        pred = preds[int(idx)]
        ref = targets[int(idx)]
        extract_pred = extract_ans(pred)
        
        if extract_pred != "FAILED":
            if extract_pred in args.options:
                extract_pred = args.options.index(extract_pred)
            else:
                num_fail += 1
                print(extract_pred, data["preds"][idx], dat["preds"][idx])
                extract_pred = random.choice(range(0,len(args.options)))
        else:
            extract_pred = random.choice(range(len(args.options))) # random choose one option
        results_ans[str(qid)] = extract_pred
        results_rationale[str(qid)] = pred
        results_reference[str(qid)] = ref
    print(num_fail)

    scores = get_scores(results_ans, results_rationale, results_reference, os.path.join(args.data_root, "scienceqa/problems.json"))
    print(scores)

if __name__ == '__main__':
    args = parse_args()
    main(args)