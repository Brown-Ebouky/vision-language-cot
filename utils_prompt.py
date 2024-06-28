'''
Adapted from https://github.com/lupantech/ScienceQA
'''

import random
from dataclasses import dataclass
from typing import List, Optional


def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_origin_answer(problem, options):
    return problem['choices'][problem['answer']]


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def get_user_attempt(problem, options, percentage=None, answer=None):
    choices = problem['choices']
    choice_list = []
    for i in range(len(choices)):
        choice_list.append(f"({options[i]})")
    if percentage:
        weights = [(1 / (len(choice_list) - 1)) * (1 - percentage)
                   for _ in range(len(choice_list))]
        weights[choice_list.index(answer)] = percentage
        example = random.choices(choice_list, weights=weights, k=1)[0]
    else:
        example = random.sample(choice_list, 1)[0]
    return example


def create_one_example(format,
                       question,
                       context,
                       choice,
                       answer,
                       lecture,
                       solution,
                       test_example=True,
                       WithOutput=False,
                       curr_le_data=None,
                       user_attempt=None):

    input_format, output_format = format.split("-")
    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"
    elif input_format == "QCMU":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nTentative answer: The answer is {user_attempt}.\n"
    elif input_format == "QCMUE":
        if user_attempt == answer:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nTentative answer: The answer is {user_attempt}. The tentative answer is correct. BECAUSE: {solution}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nTentative answer: The answer is {user_attempt}. The tentative answer not is correct. BECAUSE: {solution}\n"
    elif input_format == "QCMUG":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nTentative answer: The answer is {user_attempt}.\n{curr_le_data}\n"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'UE':
            if input_format == "QCMU":
                output = "The tentative answer is "
            else:
                output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution}."

    elif output_format == 'E':
        output = f"Solution: {solution}"
    elif output_format == 'UE':
        solution = solution.replace('\n', ' ').strip()
        if user_attempt == answer:
            output = f"The tentative answer is correct. BECAUSE: {solution}"
        else:
            output = f"The tentative answer is not correct. BECAUSE: {solution}"
    # print(f"{input=}")
    # print(f"{output=}")
    # print(f"{output_format=}")
    # print(f"{input_format=}")
    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        elif output_format == "UE":
            text = input + "The tentative answer is "
        else:
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output
    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def build_prompt(problems, shot_qids, test_qid, args):

    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input


def build_train_pair(problems,
                     test_qid,
                     args,
                     curr_le_data=None,
                     user_attempt=None,
                     sampling_true_percentage=None):

    examples = []

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])
    answer_option = get_answer(problems[test_qid], args.options)
    answer = "(" + answer_option + ")"
    # print(f"{user_attempt=}")
    if user_attempt == None:
        user_attempt = get_user_attempt(problems[test_qid], args.options,
                                        sampling_true_percentage, answer)

    test_example, target = create_one_example(args.prompt_format,
                                              question,
                                              context,
                                              choice,
                                              answer,
                                              lecture,
                                              solution,
                                              test_example=False,
                                              WithOutput=True,
                                              curr_le_data=curr_le_data,
                                              user_attempt=user_attempt)

    examples.append(test_example)

    target = target.replace("Answer:", "").strip()
    # create the prompt input
    prompt_input = '\n\n'.join(examples)
    # print(f"{curr_le_data=}")
    # print(f"{prompt_input=}")
    # print(f"{target=}")
    # print(f"{user_attempt=}")
    # assert False
    return prompt_input, target, user_attempt


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
