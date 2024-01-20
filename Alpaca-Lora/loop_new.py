# coding: utf-8

import argparse
import os
import time
from tqdm import tqdm

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import jsonlines
import re
import sys
from peft import PeftModel
from alpaca_lora.utils.callbacks import Iteratorize, Stream
from alpaca_lora.utils.prompter import Prompter
n = os.getcwd().split('/')[2]

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
def generate_step(args, model, tokenizer, prompt):
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_token_ids = input_tokens["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        num_beams=1,
        do_sample=True,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_token_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )
    output = tokenizer.decode(generation_output.sequences[0][len(input_token_ids[0]):])
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--gptscore_model", type=str, default="gpt3.5-turbo-ChatCompletion")
parser.add_argument("--demo_num", type=int, default=0)
parser.add_argument("--weight",type=float, default=0.8)
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str, default="/root/autodl-tmp/Results_Loop_alpaca-lora-7b/")
parser.add_argument("--data_name", type=str)
parser.add_argument("--model-name", type=str, default="alpaca-lora-7b")
parser.add_argument("--undertest_model_name", type=str, default="alpaca-lora-7b")
parser.add_argument("--base_model", type=str, default="/root/autodl-fs/llama-7b-hf")
parser.add_argument("--lora_weights", type=str, default="/root/autodl-fs/alpaca-lora-7b")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--conv-template", type=str, default="alpaca-lora-7b")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true", default=True)
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

prompt_users_list = ["Give me the background knowledge to answer the following question",
                     "Give me the background knowledge to answer the following question",
                     "Refer to the knowledge: {}, please answer the following question with one paragraph.",
                     "The answer can get a higher factuality and consistency. Please refine the answer to improve its factuality and consistency.",
                     "The answer: {} is not entailed with the knowledge: {} and the question: {}. Please re-explain an answer with one paragraph."]


# Firstly, define Fusion knowledge
def Fusion_Knowledge(args, model, tokenizer, line):
    print("Fusion knowledge-----------")

    subject = line['subject']
    context = line['context']
    question = line['question']
    prompt_user = prompt_users_list[0]

    # retrieved message
    if subject == "" or subject == " " or subject == "  ":
        prompt_arm1 = f'{prompt_user}' # no subject
    else:
        prompt_arm1 = f'{prompt_user} about the subject: {subject}' # there is a subject
    # conv = conv_templates[args.conv_template].copy()
    # conv.append_message(conv.roles[0], prompt_arm1)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt() # system

    # alpaca-lora
    prompter = Prompter('')
    instruction = prompter.generate_prompt(prompt_arm1, question)
    # print(instruction)
    retrieved_message = generate_step(args, model, tokenizer, instruction)
    # print(retrieved_message)

    # acquired knowledge
    prompt_user = prompt_users_list[1]
    prompt_arm2 = f'{prompt_user}'
    # conv = conv_templates[args.conv_template].copy()
    # conv.append_message(conv.roles[0], prompt_arm2)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt() # system

    # alpaca-lora
    instruction = prompter.generate_prompt(prompt_arm2,context+question)
    # print(instruction)
    acquired_knowledge = generate_step(args, model, tokenizer, instruction)
    # print(acquired_knowledge)

    #fusion knowledge
    fusion_knowledge = f'{retrieved_message}{acquired_knowledge}'
    return subject, context, question, fusion_knowledge


'''------------------------------------------------------------------------------------------'''
# # tokenizer
# tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
#
# # model
# model = LlamaForCausalLM.from_pretrained(
#     args.base_model,
#     load_in_8bit=args.load_8bit,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# model = PeftModel.from_pretrained(
#     model,
#     args.lora_weights,
#     torch_dtype=torch.float16,
# )
# # unwind broken decapoda-research config
# model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
# model.config.bos_token_id = 1
# model.config.eos_token_id = 2
# model.eval()
# model = model.to('cuda')
#
# with jsonlines.open(args.input_file) as reader:
#     reader = list(reader)
#     for i, line in tqdm(enumerate(reader), total=len(reader)):
#         subject, context, question, fusion_knowledge = Fusion_Knowledge(args, model, tokenizer, line)
#         print("fusion_knowledge:", fusion_knowledge)
'''---------------------------------------------------------------------------------------------'''

# Loop 1:
def response_loop1(args, model, tokenizer, fusion_knowledge, prompt_loop1_1, subject, context, question, response_loop1_list):
    history = []
    print("Loop 1: answering loop-----------")
    # prompt_user = prompt_users_list[2]
    # prompt_user = prompt_user.split("{}")
    # prompt_loop1_1 = f'{prompt_user[0]}{fusion_knowledge}{prompt_user[1]}{question}{prompt_user[2]}'

    # conv = conv_templates[args.conv_template].copy()
    # conv.append_message(conv.roles[0], prompt_loop1_1)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt() # system

    #alpaca-lora
    prompter = Prompter('')
    if subject == "" or subject == " " or subject == "  ":
        instruction = prompter.generate_prompt(f"Answer the question with one paragraph.", prompt_loop1_1+question)
    else:
        instruction = prompter.generate_prompt(f"Answer the question about {subject} with one paragraph. {prompt_loop1_1}", f'About {subject}:{question}?')

    ### print the instruction
    # print(instruction)
    generated_answer = generate_step(args, model, tokenizer, instruction)

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

        # conv.append_message(conv.roles[0], instruction)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # alpaca-lora
        instruction = prompter.generate_prompt(f'Refine the answer: {generated_answer} about {subject} you just made.', instruction)

        ### print instruction
        # print(instruction)
        generated_answer = generate_step(args, model, tokenizer, instruction)

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
            highest_fusion_score = fusion_score
        loop_now += 1

    best_response = response_loop1_list[highest_fusion_index]

    # write history to file
    with open("/root/autodl-tmp/Results_Loop_alpaca-lora-7b/{}/history_Loop1.txt".format(args.data_name), "a") as log_loop1:
        for item in history:
            for index in range(len(item)-1):
                log_loop1.write(str(item[index])+", ")
            log_loop1.write(str(item[len(item)-1])+"\n")

    return best_response, response_loop1_list, history


# Loop 2:
def response_loop2(args, model, tokenizer, fusion_knowledge, subject, context, question, best_response):
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
        with open("/root/autodl-tmp/Results_Loop_alpaca-lora-7b/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
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
        instruction = f'{prompt_user[0]}{alternative_response[1]}{prompt_user[1]}{fusion_knowledge}{prompt_user[2]}{question}{prompt_user[3]}'

        # turn into loop1
        best_response, response_loop1_list, _ = response_loop1(args, model, tokenizer, fusion_knowledge, instruction, subject, context, question, response_loop1_list)

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
                with open("/root/autodl-tmp/Results_Loop_alpaca-lora-7b/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
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
                                                                       subject,
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
        with open("/root/autodl-tmp/Results_Loop_alpaca-lora-7b/{}/history_Loop2.txt".format(args.data_name), "a") as log_loop2:
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
print("here")
entailment_scorer = Sent_Similar()
print("here 1")
# tokenizer, model
# tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

# model
model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    load_in_8bit=args.load_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    args.lora_weights,
    torch_dtype=torch.float16,
)
# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()
model = model.to('cuda')
print("model loaded.")

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
        with open("/root/autodl-tmp/Results_Loop_alpaca-lora-7b/{}/checkpoints.txt".format(args.data_name), 'r') as checkpoints:
            if str(i) in checkpoints.readline().split(' '):
                continue

            else:
                # fusion_knowledge
                line = reader[i]
                subject, context, question, fusion_knowledge = Fusion_Knowledge(args, model, tokenizer, line)

                # Loop 1:
                prompt_user = prompt_users_list[2]
                prompt_user = prompt_user.split("{}")
                prompt_loop1_1 = f'{prompt_user[0]}{fusion_knowledge}{prompt_user[1]}{question}'
                response_loop1_list = []
                best_response, response_loop_list, _ = response_loop1(args, model, tokenizer, fusion_knowledge, prompt_loop1_1, subject, context, question, response_loop1_list)

                # Loop 2:
                generated_final_response = response_loop2(args, model, tokenizer, fusion_knowledge, subject, context, question, best_response)
                # print(i, "-----------------------------------------------------------------------------------------------")
                # print(generated_final_response)

                #write into outfile
                writer = jsonlines.open(outfile, mode='a')
                writer.write(str(i)+": -------------------------------------------------------")
                writer.write(str(generated_final_response))
                writer.close()

                # load checkpoint to checkpoints
                with open("/root/autodl-tmp/Results_Loop_alpaca-lora-7b/{}/checkpoints.txt".format(args.data_name),
                          'a') as checkpoints:
                    checkpoints.write(" " + str(i))