import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from model import T5ForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg, ScienceQADatasetSimple
from utils_prompt import *
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
from torch.utils.data import DataLoader

####### Llama related #######

from typing import List, Optional
from tqdm import tqdm
import fire

from llama import Dialog, Llama


nltk.download('punkt')

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
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE', "QCMU-E", "QCMU-UE", "QCMUE-A"])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--load_ua', action='store_true', help='wether to load ua')
    parser.add_argument('--method', type=str, default="ua", help='whether to load ua or predictions ("preds")', choices=['ua', 'preds'])


    ### LLAMA Related
    
    parser.add_argument('--ckpt_dir', type=str, default="ua", help='path to the ckpt dir of llama')
    parser.add_argument('--tokenizer_path', type=str, default="ua", help='path to the tokenizer of llama')
    parser.add_argument('--max_seq_len', type=int, default=512, help='max sequence length')
    parser.add_argument('--max_gen_len', type=int, default=512, help='max  generated sequence length')
    parser.add_argument('--max_inter_gen_len', type=int, default=512, help='max intermediate generated sequence length')
    parser.add_argument('--max_batch_size', type=int, default=6, help='max batch size')
    parser.add_argument('--save_outputs', type=str, default="./output.json", help='output json file')
    parser.add_argument('--temperature', type=float, default=0.6, help='max batch size')
    parser.add_argument('--top_p', type=float, default=0.9, help='max batch size')
    


    args = parser.parse_args()
    return args
        

def create_llama_generator(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 64,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=args.seed
    )
    
    return generator
        

def LlamaTrainer(
    dataframe, args, llama_generator=None,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    # print(args.seed)
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    # torch.backends.cudnn.deterministic = True
    
    # if args.evaluate_dir is not None:
    #     args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)

    # model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=16) 
    train_set = ScienceQADatasetSimple(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            # test_le=args.test_le
        )
    test_set = ScienceQADatasetSimple(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            test_le=args.test_le
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    
    dataloader_params = {
        "batch_size": args.bs,
        # "collate_fn": datacollator,
        "num_workers": 0,
        "pin_memory": True,
        # "persistent_workers": self.args.dataloader_persistent_workers,
    }
    dataloader = DataLoader(
        test_set,
        **dataloader_params
    ) 


    llama_generator = create_llama_generator(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    answers = []
    prompts = []
    
    shots = 3
    
    for data in tqdm(iter(dataloader)):

        results = llama_generator.text_completion(
            data[0],
            max_gen_len=args.max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # prompt_feedback = "Is the answer to the question correct? Please give a short feedback.\n"
        
        # ask_feedback = [prompt_feedback + data[0][i] + results[i]['generation'] for i in range(len(results))]
        
        # inter_results = llama_generator.text_completion(
        #     ask_feedback,
        #     max_gen_len=args.max_inter_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        # )
        
        # prompt_final_ans = "Considering all these details, provide the final correct answer. \n"

        # prompt_final_letter =  "\nPlease give the answer only with the letter:"
        
        # ask_final_ans = [prompt_final_ans + data[0][i] + "\n" + results[i]['generation'] + "\n" + inter_results[i]['generation'] + prompt_final_letter for i in range(len(inter_results))]
        
        # final_results = llama_generator.text_completion(
        #     ask_final_ans,
        #     max_gen_len=args.max_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        # )
        
        
        for prompt, result in zip(data[0], results):
            answers.append("The answer is" + result['generation'])
            prompts.append(prompt)
            
            # print(f"Prompt: {prompt} \\")
            # print("//"*30)
            # print(f"Result_generation: {result['generation']}\\")
            # print('----' * 30)
    
    pred_dict = {"preds": answers, "prompts": prompts}


    
    with open(args.save_outputs, 'w') as outfile:
        json.dump(pred_dict, outfile, indent=4)

    
if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    # random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids}

    LlamaTrainer(
        dataframe=dataframe,
        args=args,
        top_p=args.top_p,
        temperature=args.temperature,
    )
