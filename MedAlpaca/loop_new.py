# coding: utf-8

import argparse
import os
import time
from tqdm import tqdm

from transformers import GenerationConfig
import jsonlines
import re
import sys
import time
import string
from medAlpaca.medalpaca.inferer import Inferer

# import evaluation
import evaluate
from evaluate.sent_similarity import Sent_Similar
from evaluate.CTRLEval.ctrleval import CTRLEval
import numpy as np

# evaluate the fusion score of factuality and the consistency of the response
from evaluate.loop_FusionScorer_utils import evaluate_response

# evaluate the entailment of the response by human
# from loop_utils_new import main_loop

import torch

def strip_special_chars(input_str):
    "Remove special characters from string start/end"
    if not input_str:
        return input_str

    start_index = 0
    end_index = len(input_str) - 1

    while start_index < len(input_str) and input_str[start_index] not in string.ascii_letters + string.digits:
        start_index += 1

    while end_index >= 0 and input_str[end_index] not in string.ascii_letters + string.digits:
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index:end_index + 1]
    else:
        return ""


#
def generate_step(args, model, instruction, question):
    response = model(
        instruction=instruction,
        input=question,
        output="Re:",
        temperature=1,
        max_new_tokens=128,
    )
    response = strip_special_chars(response)
    res = re.search("### References", response)
    if res:
        response = response[:res.span()[0]].strip()
        response = response_disposal(response)
    return response


# define the answer disposal
def response_disposal(response):
    response_list = response.split(".")[:-1]
    for index, item in enumerate(response_list):
        response_list[index] = str(item).replace(".", "") + "."
    response_new = ''.join(response_list)
    return response_new

parser = argparse.ArgumentParser()
parser.add_argument("--gptscore_model", type=str, default="gpt3.5-turbo-ChatCompletion")
parser.add_argument("--demo_num", type=int, default=0)
parser.add_argument("--weight",type=float, default=0.8)
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str, default="/root/autodl-tmp/Results_Loop_medAlpaca-7b/")
parser.add_argument("--data_name", type=str)
parser.add_argument("--model-name", type=str, default="medAlpaca-7b")
parser.add_argument("--undertest_model_name", type=str, default="medAlpaca-7b")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--early_stopping", type=bool, default=False)
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--load_8bit", action="store_true")

parser.add_argument("--MAX_LOOP_1", type=int, default=1)
parser.add_argument("--MAX_LOOP_2", type=int, default=1)

parser.add_argument("--random_seed", type=int, default=20)
parser.add_argument("--eval_nums", type=int, default=50)

args = parser.parse_args()

args.do_sample = True
# args.load_8bit = False
# args.debug = False

prompt_users_list = ["Can you give me the background knowledge to answer the given question?",
                     "Can you give me the background knowledge to answer the given question?",
                     "Refer to the knowledge: {}",
                     "Please refine the answer you just made to improve its factuality and consistency with the question ",
                     "The answer is not entailed with the knowledge: {} and the question: {}. Please re-explain an answer with one paragraph."]


# Firstly, define Fusion knowledge
def Fusion_Knowledge(args, model, line):
    print("Fusion knowledge-----------")

    subject = line['subject']
    context = line['context']
    question = line['question']
    prompt_user = prompt_users_list[0]
    fusion_knowledge = []

    # retrieved message
    if subject == "" or subject == " " or subject == "  ":
        prompt_arm1 = f'Question: {question}' # no subject
    else:
        prompt_arm1 = f'Question: {question.split("?")[0]} about the subject: {subject}?' # there is a subject
    instruction = prompt_user + prompt_arm1
    retrieved_message = generate_step(args, model, "Give me the background knowledge with one paragraph.", instruction)
    retrieved_message = response_disposal(retrieved_message)
    # print("retrieved message:", retrieved_message)

    # acquired knowledge
    if context != "" and context != " " and context != "  ":
        prompt_user = prompt_users_list[1]
        prompt_arm2 = f'Question: {question}'
        instruction = f'According: {context}, {prompt_user} {prompt_arm2}'
        # print(instruction)
        acquired_knowledge = generate_step(args, model, "Give me the background knowledge.", instruction)
        acquired_knowledge = response_disposal(acquired_knowledge)
        # print("acquired knowledge:", acquired_knowledge)

        fusion_knowledge = f'{retrieved_message}{acquired_knowledge}'
    elif context == "" or context == " " or context == "  ":
        fusion_knowledge = f'{retrieved_message}'

    return context, question, fusion_knowledge

# # ------------------------------------------------------------------------------------------------
# model = Inferer(
#     model_name = "/root/autodl-fs/medalpaca-7b",
#     prompt_template = "/root/autodl-tmp/medAlpaca/medAlpaca/medalpaca/prompt_templates/medalpaca.json",
#     base_model = "/root/autodl-fs/llama-7b-hf",
#     peft=True,
#     load_in_8bit=True,
# )
#
# with jsonlines.open(args.input_file) as reader:
#     reader = list(reader)
#     for i, line in tqdm(enumerate(reader), total=len(reader)):
#         context, question, fusion_knowledge = Fusion_Knowledge(args, model, line)
#         print("fusion_knowledge:", fusion_knowledge)
# # --------------------------------------------------------------------------------------------------

# Loop 1:
def response_loop1(args, model, fusion_knowledge, prompt_loop1_1, context, question, response_loop1_list):
    history = []
    print("Loop 1: answering loop-----------")
    # prompt_user = prompt_users_list[2]
    # prompt_user = prompt_user.split("{}")
    # prompt_loop1_1 = f'{prompt_user[0]}{fusion_knowledge}{prompt_user[1]}{question}{prompt_uyser[2]}'

    # medAlpaca-7b
    generated_answer = generate_step(args, model, f'According: {fusion_knowledge}', prompt_loop1_1)
    generated_answer = generated_answer.replace("\n", "")
    for i in range(10):
        generated_answer = generated_answer.replace(str(i)+".", "")
        generated_answer = response_disposal(generated_answer)
    generated_answer = generated_answer +'.'
    generated_answer = generated_answer.replace("..", ".")
    if len(generated_answer.split("?")) > 1:
        generated_answer = generated_answer.split("?")[1]
    print("generated_answer1:", generated_answer)
    loop_now = 0
    # fusion-score here
    if args.gptscore_model == 'gpt3':
        factuality_score, entailment_score, consistency_score, fusion_score = evaluate_response('gpt3',
                                                                                                args.undertest_model_name,
                                                                                                args.data_name,
                                                                                                args.demo_num,
                                                                                                context,
                                                                                                question,
                                                                                                fusion_knowledge,
                                                                                                generated_answer,
                                                                                                entailment_scorer,
                                                                                                ctrleval_scorer,
                                                                                                args.weight)
    elif args.gptscore_model == 'OPT':
        factuality_score, entailment_score, consistency_score, fusion_score = evaluate_response(args.gptscore_model,
                                                                                                args.undertest_model_name,
                                                                                                args.data_name,
                                                                                                args.demo_num,
                                                                                                context,
                                                                                                question,
                                                                                                fusion_knowledge,
                                                                                                generated_answer,
                                                                                                entailment_scorer,
                                                                                                ctrleval_scorer,
                                                                                                args.weight)
    else:
        factuality_score, entailment_score, consistency_score, fusion_score = evaluate_response(args.gptscore_model,
                                                                                                args.undertest_model_name,
                                                                                                args.data_name,
                                                                                                args.demo_num,
                                                                                                context,
                                                                                                question,
                                                                                                fusion_knowledge,
                                                                                                generated_answer,
                                                                                                entailment_scorer,
                                                                                                ctrleval_scorer,
                                                                                                args.weight)
    # load to history
    import time
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    history.append([formatted_time, loop_now+1, question, generated_answer, fusion_score, factuality_score, consistency_score, entailment_score])

    # load to response_loop1_list
    response_loop1_list.append([question, generated_answer, fusion_score, factuality_score, consistency_score, entailment_score])
    highest_fusion_score = fusion_score
    highest_fusion_index = loop_now

    # loop1
    # MAX_LOOP_1 = 3
    loop_now += 1
    prompt_loop1_2 = prompt_users_list[3]
    while loop_now < args.MAX_LOOP_1:
        instruction = prompt_loop1_2
        # print("instruction:", instruction)
        generated_answer = generate_step(args, model, f'{instruction}', question)
        generated_answer = generated_answer.replace("\n", "")
        for i in range(10):
            generated_answer = generated_answer.replace(str(i) + ".", "")
            generated_answer = response_disposal(generated_answer)
        generated_answer = generated_answer+"."
        generated_answer = generated_answer.replace("\n", ".")
        generated_answer = generated_answer.replace("..", ".")
        if len(generated_answer.split("?")) >1:
            generated_answer = generated_answer.split("?")[1]
        print("generated answer in Loop1:", generated_answer)
        # fusion-scorer
        if args.gptscore_model == 'gpt3':
            factuality_score, entailment_score, consistency_score, fusion_score = evaluate_response('gpt3',
                                                                                                    args.undertest_model_name,
                                                                                                    args.data_name,
                                                                                                    args.demo_num,
                                                                                                    context,
                                                                                                    question,
                                                                                                    fusion_knowledge,
                                                                                                    generated_answer,
                                                                                                    entailment_scorer,
                                                                                                    ctrleval_scorer,
                                                                                                    args.weight)
        else:
            factuality_score, entailment_score, consistency_score, fusion_score = evaluate_response(args.gptscore_model,
                                                                                                    args.undertest_model_name,
                                                                                                    args.data_name,
                                                                                                    args.demo_num,
                                                                                                    context, question,
                                                                                                    fusion_knowledge,
                                                                                                    generated_answer,
                                                                                                    entailment_scorer,
                                                                                                    ctrleval_scorer,
                                                                                                    args.weight)
        # load to history
        import time
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        history.append([formatted_time, loop_now+1, question, generated_answer, fusion_score, factuality_score,
                        consistency_score, entailment_score])

        # load to response_loop1_list
        response_loop1_list.append(
            [question, generated_answer, fusion_score, factuality_score, consistency_score, entailment_score])

        if highest_fusion_score < fusion_score:
            highest_fusion_index = loop_now
        loop_now += 1

    best_response = response_loop1_list[highest_fusion_index]

    # write history to file
    with open("/root/autodl-tmp/Results_Loop_medAlpaca-7b/{}/history_Loop1.txt".format(args.data_name), "a") as log_loop1:
        for item in history:
            for index in range(len(item)-1):
                log_loop1.write(str(item[index])+", ")
            log_loop1.write(str(item[len(item)-1])+"\n")

    return best_response, response_loop1_list, history


# Loop 2:
def response_loop2(args, model, fusion_knowledge, context, question, best_response):
    # MAX_LOOP_2 = 3
    print("Loop 2: Human entailment judge loop-----------------")
    history = []
    final_answer = []  # used to record the best response
    alternative_response = []  # used to record a better response when there always has no Entailed responses
    loop2_now = 0

    # print to the terminal
    print("Question:", question)  #
    print("Best Response from loop1:", best_response[1])
    alternative_response = best_response

    # human judge
    human_get = input("Entailed(1/\'Enter\') or Unentailed(0):")
    if human_get == 'Entailed' or human_get == '1' or human_get == 'y' or human_get == '':  # Entailed
        final_answer = best_response

        # load to history
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        history.append([formatted_time, loop2_now, final_answer[0], final_answer[1], final_answer[2], final_answer[3], final_answer[4], final_answer[5], human_get])

        # write history to file
        with open("/root/autodl-tmp/Results_Loop_medAlpaca-7b/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
            for item in history:
                for index in range(len(item) - 1):
                    log_loop2.write(str(item[index]) + ", ")
                log_loop2.write(str(item[len(item) - 1]) + "\n")
        return final_answer

    elif human_get == 'Unentailed' or human_get =='0' or human_get == 'n': # Unentailed
        response_loop1_list = []

        # load to history
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        history.append(
            [formatted_time, loop2_now, alternative_response[0], alternative_response[1], alternative_response[2],
             alternative_response[3], alternative_response[4], alternative_response[5], human_get])

        # instruction here
        prompt_user = prompt_users_list[4].split("{}")
        instruction = f'{prompt_user[0]}{fusion_knowledge}{prompt_user[1]}{question}{prompt_user[2]}'

        # turn into loop1
        best_response, response_loop1_list, _ = response_loop1(args, model, fusion_knowledge, instruction, context, question, response_loop1_list)

        if 0.2*best_response[2] + 0.2*best_response[3] + 0.6*best_response[5] > 0.2*alternative_response[2] + 0.2*alternative_response[3] + 0.6*alternative_response[5]:
            alternative_response = best_response
        else:
            alternative_response = alternative_response

        # print to the terminal
        print("Question:", question)
        print("Best Response now:", alternative_response[1])
        human_get = input("Entailed(1/\'Enter\') or Unentailed(0):")

        # load to history
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        history.append([formatted_time, loop2_now + 1, alternative_response[0], alternative_response[1], alternative_response[2], alternative_response[3], alternative_response[4], alternative_response[5], human_get])

        loop2_now += 1

        while loop2_now < args.MAX_LOOP_2:
            if human_get == 'Entailed' or human_get == '1' or human_get == 'y' or human_get == '':  # Entailed
                final_answer = alternative_response
                # # load to history
                # current_time = time.localtime()
                # formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
                # history.append(
                #     [formatted_time, loop2_now + 1, final_answer[0], final_answer[1], final_answer[2], final_answer[3],
                #      final_answer[4], final_answer[5], human_get])
                # write history to file
                with open("/root/autodl-tmp/Results_Loop_medAlpaca-7b/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
                    for item in history:
                        for index in range(len(item) - 1):
                            log_loop2.write(str(item[index]) + ", ")
                        log_loop2.write(str(item[len(item) - 1]) + "\n")
                return final_answer
            elif human_get == 'Unentailed' or human_get =='0' or human_get == 'n': # Unentailed
                # turn into loop1
                response_loop1_list = []
                best_response, response_loop1_list, _ = response_loop1(args,
                                                                      model,
                                                                      fusion_knowledge,
                                                                      instruction,
                                                                      context,
                                                                      question,
                                                                      response_loop1_list)

                if 0.2*best_response[2] + 0.2*best_response[3] + 0.6*best_response[5] > 0.2*alternative_response[2] + 0.2*alternative_response[3] + 0.6*alternative_response[5]:
                    alternative_response = best_response
                else:
                    alternative_response = alternative_response

                # print to the terminal
                print("Question:", question)  #
                print("Best Response now:", alternative_response[1])
                human_get = input("Entailed(1/\'Enter\') or Unentailed(0):")

                # load to history
                current_time = time.localtime()
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
                history.append([formatted_time, loop2_now + 1, alternative_response[0], alternative_response[1], alternative_response[2], alternative_response[3], alternative_response[4], alternative_response[5], human_get])

                loop2_now += 1

        # write history to file
        with open("/root/autodl-tmp/Results_Loop_medAlpaca-7b/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
            for item in history:
                for index in range(len(item) - 1):
                    log_loop2.write(str(item[index]) + ", ")
                log_loop2.write(str(item[len(item) - 1]) + "\n")
        return alternative_response


#####################################################
task = "topic"
ctrleval_scorer = CTRLEval(iwf_dir="/root/autodl-tmp/evaluate/CTRLEval/iwf_full.txt",
                           prompt_dir="/root/autodl-tmp/evaluate/CTRLEval/prompt/prompt_{}.txt".format(task),
                           verbal_dir="/root/autodl-tmp/evaluate/CTRLEval/prompt/verbal_{}.txt".format(task),
                           device='cuda',
                           model_name_or_path="/root/autodl-fs/pegasus-large")
entailment_scorer = Sent_Similar()

model = Inferer(
    model_name = "/root/autodl-fs/medalpaca-7b",
    prompt_template = "/root/autodl-tmp/medAlpaca/medAlpaca/medalpaca/prompt_templates/medalpaca.json",
    base_model = "/root/autodl-fs/llama-7b-hf",
    peft=True,
    load_in_8bit=True,
)

outfile = args.out_file + args.data_name + "/" + args.undertest_model_name + "_final_response.jsonl"
inputfile = args.input_file

import random

random.seed(args.random_seed)
print("loading the dataset file to evaluate (nums of 50)")
with jsonlines.open(inputfile) as reader:
    reader = list(reader)
    total_length = len(reader)
    random_numbers = random.sample(range(total_length), args.eval_nums)
    random_numbers = sorted(random_numbers)

    for i in tqdm(random_numbers, total=len(random_numbers)):
        print(i, "----------------------------------------------------------")
        # check the checkpoints
        with open("/root/autodl-tmp/Results_Loop_medAlpaca-7b/{}/checkpoints.txt".format(args.data_name), 'r') as checkpoints:
            if str(i) in checkpoints.readline().split(' '):
                continue

            else:
                # fusion_knowledge
                line = reader[i]
                context, question, fusion_knowledge = Fusion_Knowledge(args, model, line)
                fusion_knowledge = fusion_knowledge.replace("\n", " ")
                fusion_knowledge = fusion_knowledge.replace("  "," ")
                if len(fusion_knowledge.split("?")) >1:
                    fusion_knowledge = fusion_knowledge.split("?")[1].replace("  ", "")
                # print("fusion_knowledge:", fusion_knowledge)
                # Loop 1:
                prompt_user = prompt_users_list[2]
                prompt_user = prompt_user.split("{}")
                prompt_loop1_1 = f'{question}'

                response_loop1_list = []
                best_response, response_loop_list, _ = response_loop1(args, model, fusion_knowledge, prompt_loop1_1, context, question, response_loop1_list)

                # Loop 2:
                generated_final_response = response_loop2(args, model, fusion_knowledge, context, question, best_response)
                # print(i, "-----------------------------------------------------------------------------------------------")
                # print(generated_final_response)

                #write into outfile
                writer = jsonlines.open(outfile, mode='a')
                writer.write(str(i)+": -------------------------------------------------------")
                writer.write(str(generated_final_response))
                writer.close()

                # load checkpoint to checkpoints
                with open("/root/autodl-tmp/Results_Loop_medAlpaca-7b/{}/checkpoints.txt".format(args.data_name),
                          'a') as checkpoints:
                    checkpoints.write(" " + str(i))