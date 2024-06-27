import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
}

def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]
    # if args.load_caption is None:
    #     captions = ["" for _ in captions]

    # print(args.use_caption)
    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

        # # In case we dont want to use the captions
        # problems[qid]['caption'] = problems[qid]['caption'] if args.use_caption else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids,

def load_data_img(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]
    name_maps = json.load(open('data/name_map.json'))

    # check
    if args.img_type == "resnet":
        image_features = np.load('vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load('vision_features/clip.npy')
    elif args.img_type == "detr":
        image_features = np.load('vision_features/detr.npy')
    elif args.img_type == "vit":
        image_features = torch.load("vision_features/vit.pth")
    else:
        image_features = np.load('vision_features/detr.npy')
    print("img_features size: ", image_features.shape)

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, name_maps, image_features

class ScienceQADatasetSimple(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None, load_ua=None, method="ua"
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.user_attempt_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
            if load_ua:
                ua_data = json.load(open(test_le))[method]
        else:
            test_le_data = None
        idx = 0
        ua = None
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                if load_ua:
                    ua = ua_data[idx]
                    if method == "preds":
                        ua.removeprefix("The answer is ")
                idx += 1
            else:
                curr_le_data = None
            prompt, target, user_attempt = build_train_pair(problems, qid, args, curr_le_data, ua)
            self.target_text.append(target)
            self.source_text.append(prompt)
            self.user_attempt_text.append(user_attempt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())

        # prompt = "Instructions: Let's think step by step for each of the options, give an explanation for why it is or it is not correct. Using those explanations, provide the actual correct answer. "
        # source_text = prompt + source_text

        # source_text = "Please provide the answer to the question only as a letter. " + source_text
        # source_text = "Please provide the answer in the format 'The answer is: (_)'. Stop after that. " + source_text
        target_text = " ".join(target_text.split())


        return (source_text, target_text)

class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None, load_ua=None, method="ua", sampling_true_percentage=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.user_attempt_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
            if load_ua:
                ua_data = json.load(open(test_le))[method]
        else:
            test_le_data = None
        idx = 0
        ua = None
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                if load_ua:
                    ua = ua_data[idx]
                    if method == "preds":
                        ua.replace("The answer is ", "")
                        ua.replace(".", "")
                idx += 1
            else:
                curr_le_data = None
            prompt, target, user_attempt = build_train_pair(problems, qid, args, curr_le_data, ua, sampling_true_percentage)
            self.target_text.append(target)
            self.source_text.append(prompt)
            self.user_attempt_text.append(user_attempt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }
        
class ScienceQADatasetSimple(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None, load_ua=None, method="ua"
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.user_attempt_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
            if load_ua:
                ua_data = json.load(open(test_le))[method]
        else:
            test_le_data = None
        idx = 0
        ua = None
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                if load_ua:
                    ua = ua_data[idx]
                    if method == "preds":
                        ua.removeprefix("The answer is ")
                idx += 1
            else:
                curr_le_data = None
            prompt, target, user_attempt = build_train_pair(problems, qid, args, curr_le_data, ua)
            self.target_text.append(target)
            self.source_text.append(prompt)
            self.user_attempt_text.append(user_attempt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        
        # prompt = "Instructions: Let's think step by step for each of the options, give an explanation for why it is or it is not correct. Using those explanations, provide the actual correct answer. "
        # source_text = prompt + source_text
        
        # source_text = "Please provide the answer to the question only as a letter. " + source_text
        # source_text = "Please provide the answer in the format 'The answer is: (_)'. Stop after that. " + source_text
        target_text = " ".join(target_text.split())

        
        return (source_text, target_text)


class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None, load_ua=None, method="ua", sampling_true_percentage=None,
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.user_attempt_text = []
        self.image_ids = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
            if load_ua:
                ua_data = json.load(open(test_le))[method]
        else:
            test_le_data = None
        idx = 0
        ua = None
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                curr_le_data = curr_le_data.replace("The answer is ", "")
                curr_le_data = curr_le_data.replace(".", "")
                if load_ua:
                    ua = ua_data[idx]
                    if method == "preds":
                        ua = ua.replace("The answer is ", "")
                        ua = ua.replace(".", "")
                idx += 1
            else:
                curr_le_data = None
            prompt, target, user_attempt = build_train_pair(problems, qid, args, curr_le_data, ua, sampling_true_percentage)
            self.target_text.append(target)
            self.source_text.append(prompt)
            self.user_attempt_text.append(user_attempt)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))

    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        image_ids = torch.tensor(image_ids).squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }
