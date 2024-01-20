# coding: utf-8

import argparse
import os
import time
from tqdm import tqdm

from fastchat.serve.inference import generate_stream, load_model1
from fastchat.conversation import conv_templates, SeparatorStyle
from transformers import GenerationConfig
import jsonlines
import re
import sys


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

# define the generation step (generate sentence here)
# def generate_step(args, model, tokenizer, prompt, conv):
#     params = {
#         'prompt': prompt,
#         'temperature': args.temperature,
#         'max_new_tokens': args.max_new_tokens,
#         'stop': conv.sep if con.sep_style == SeparatorStyle.SINGLE else conv.sep2,
#     }
#
#     pre = 0
#
#     for outputs in generate_stream(tokenizer, model, params, args.device):
#         outputs = outputs[len(promt) + 1 :].strip()
#         outputs = outputs.split(" ")
#
#     return " ".join(outputs)


# 2
def generate_step(args, model, tokenizer, prompt, conv):
    stop_str = conv.sep if conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE else conv.sep2

    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_token_ids = input_tokens["input_ids"].to("cuda")

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_token_ids,
            max_new_tokens=args.max_new_tokens,
            early_stopping=False,
            num_beams=args.num_beams,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    output = tokenizer.decode(generation_output[0][len(input_token_ids[0]):])
    result = re.search(stop_str, output)
    if result:
        prompt = prompt[:result.span()[0]]
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--gptscore_model", type=str, default="gpt3.5-turbo-ChatCompletion")
parser.add_argument("--demo_num", type=int, default=0)
parser.add_argument("--weight",type=float, default=0.8)
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str, default="/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/")
parser.add_argument("--data_name", type=str)
parser.add_argument("--model-name", type=str, default="vicuna-7b-v1.5")
parser.add_argument("--undertest_model_name", type=str, default="vicuna-7b-v1.5")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--conv-template", type=str, default="vicuna-7b-v1.5")
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

prompt_users_list = ["Give me the background knowledge to answer the given question: ",
                     "Give me the background knowledge to answer the given question: ",
                     "Refer to the knowledge: {}, please answer the question: {} with one paragraph.",
                     "The answer can get a higher factuality and consistency. Please refine the response you just made to improve its factuality and consistency.",
                     "The answer is not entailed with the knowledge: {} and the question: {}. Please re-explain an answer with one paragraph."]


# Firstly, define Fusion knowledge
def Fusion_Knowledge(args, model, tokenizer, line):
    print("Fusion knowledge-----------")

    subject = line['subject']
    context = line['context']
    question = line['question']
    prompt_user = prompt_users_list[0]

    # retrieved message
    if subject == "" or subject == " " or subject == "  ":
        prompt_arm1 = f'{prompt_user}{question}' # no subject
    else:
        prompt_arm1 = f'{prompt_user}{question} about the subject: {subject}' # there is a subject
    conv = conv_templates[args.conv_template].copy()
    conv.append_message(conv.roles[0], prompt_arm1)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() # system
    retrieved_message = generate_step(args, model, tokenizer, prompt, conv)

    # acquired knowledge
    prompt_user = prompt_users_list[1]
    prompt_arm2 = f'{prompt_user}{context}{question}'
    conv = conv_templates[args.conv_template].copy()
    conv.append_message(conv.roles[0], prompt_arm2)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() # system
    acquired_knowledge = generate_step(args, model, tokenizer, prompt, conv)

    #fusion knowledge
    fusion_knowledge = f'{retrieved_message}{acquired_knowledge}'
    return context, question, fusion_knowledge


'''model, tokenizer = load_model1(args.model_name, args.device, args.num_gpus, args.load_8bit)
model = model.to('cuda')

with jsonlines.open(args.input_file) as reader:
    reader = list(reader)
    for i, line in tqdm(enumerate(reader), total=len(reader)):
        question, fusion_knowledge = Fusion_Knowledge(args, model, tokenizer, line)
        print("fusion_knowledge:", fusion_knowledge)'''


# Loop 1:
def response_loop1(args, model, tokenizer, fusion_knowledge, prompt_loop1_1, context, question, response_loop1_list):
    history = []
    print("Loop 1: answering loop-----------")
    # prompt_user = prompt_users_list[2]
    # prompt_user = prompt_user.split("{}")
    # prompt_loop1_1 = f'{prompt_user[0]}{fusion_knowledge}{prompt_user[1]}{question}{prompt_user[2]}'

    conv = conv_templates[args.conv_template].copy()
    conv.append_message(conv.roles[0], prompt_loop1_1)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() # system
    generated_answer = generate_step(args, model, tokenizer, prompt, conv)

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

        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        generated_answer = generate_step(args, model, tokenizer, prompt, conv)

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
    with open("/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/{}/history_Loop1.txt".format(args.data_name), "a") as log_loop1:
        for item in history:
            for index in range(len(item)-1):
                log_loop1.write(str(item[index])+", ")
            log_loop1.write(str(item[len(item)-1])+"\n")

    return best_response, response_loop1_list, history


# Loop 2:
def response_loop2(args, model, tokenizer, fusion_knowledge, context, question, best_response):
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
        with open("/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
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
        best_response, response_loop1_list, _ = response_loop1(args, model, tokenizer, fusion_knowledge, instruction, context, question, response_loop1_list)

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
                with open("/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
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
                                                                      tokenizer,
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
        with open("/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
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
model, tokenizer = load_model1(args.model_name, args.device, args.num_gpus, args.load_8bit)
model = model.to('cuda')

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
        # check the checkpoints
        with open("/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/{}/checkpoints.txt".format(args.data_name), 'r') as checkpoints:
            if str(i) in checkpoints.readline().split(' '):
                continue

            else:
                # fusion_knowledge
                line = reader[i]
                context, question, fusion_knowledge = Fusion_Knowledge(args, model, tokenizer, line)

                # Loop 1:
                prompt_user = prompt_users_list[2]
                prompt_user = prompt_user.split("{}")
                prompt_loop1_1 = f'{prompt_user[0]}{fusion_knowledge}{prompt_user[1]}{question}{prompt_user[2]}'
                response_loop1_list = []
                best_response, response_loop_list, _ = response_loop1(args, model, tokenizer, fusion_knowledge, prompt_loop1_1, context, question, response_loop1_list)

                # Loop 2:
                generated_final_response = response_loop2(args, model, tokenizer, fusion_knowledge, context, question, best_response)
                # print(i, "-----------------------------------------------------------------------------------------------")
                # print(generated_final_response)

                #write into outfile
                writer = jsonlines.open(outfile, mode='a')
                writer.write(str(i)+": -------------------------------------------------------")
                writer.write(str(generated_final_response))
                writer.close()

                # load checkpoint to checkpoints
                with open("/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/{}/checkpoints.txt".format(args.data_name),
                          'a') as checkpoints:
                    checkpoints.write(" " + str(i))