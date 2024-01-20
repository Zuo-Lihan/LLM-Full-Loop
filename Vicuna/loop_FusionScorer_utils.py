from evaluate.GPTScore.gpt3_score import gpt3score
from evaluate.GPTScore.opt_score import OPTScorer
from evaluate.OneTimeChatScore import GPT_onetime_chat_message_score
from evaluate.txtfiles import *
import torch
import time

def evaluate_response(gptscore_model, undertest_model_name, data_name, demo_num, context, question, fusion_knowledge, generated_answer, entailment_scorer, ctrleval_scorer, weight=0.5):
    # score the factuality
    PREFIX = {
        0: f'''Based on the Context and Question, please generate the factual response. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.
    Context: {context}
    Question: {question}
    Response: ''', }
    prefix = PREFIX[demo_num]


    if gptscore_model == 'gpt3':
        time.sleep(3.6)
        gptscore = gpt3score(input=prefix,
                             output=generated_answer,
                             gpt3model='gpt3.5-turbo',
                             # api_key="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",
                             api_key = "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",
                             url_agent="https://api.openai-proxy.com/v1") # load API here, and should use a url-agent in 'gpt3score'
    elif gptscore_model == 'OPT':
        srcs = [prefix]
        tgts = [generated_answer]
        score_list = OPTScorer(device='cuda', max_length=1024, checkpoint=gptscore_model).score(srcs, tgts, prompt_text="", batch_size=1)
        gptscore = score_list[0]
        # factuality_score = gptscore
    else:
        prefix_txt = prefix_file_txt_root
        unrelated_words_txt = unrelated_words_txt_root
        judge_messages_write_txt = judge_messages_write_txt_root
        gptscore = GPT_onetime_chat_message_score(prefix_txt=prefix_txt,
                                                  response=generated_answer,
                                                  unrelated_words_txt=unrelated_words_txt,
                                                  judge_messages_write_txt=judge_messages_write_txt,
                                                  data_name=data_name,
                                                  # api_KEY="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",
                                                  api_KEY = "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",
                                                  base_url="https://api.openai-proxy.com/v1",
                                                  model_name=undertest_model_name)
        time.sleep(1.6)
    factuality_score = gptscore

    # score the entailment and the consistency
    query = f'{context}{question}'
    scores, _ = entailment_scorer.get_scores(query, [generated_answer])
    entailment_score = scores[0] # entailment score

    if fusion_knowledge:
        prefix = [fusion_knowledge]
        data = [fusion_knowledge+'\n'+generated_answer]
        try:
            consistency_result = ctrleval_scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=1)
            consistency_score = consistency_result[0] # consistency score
        except:
            consistency_score = float('-inf')
    else:
        consistency_score = float('-inf')

    # fusion the scorers here
        # use the 'softplus' function to map the scores to a positive 0-1 field
    factuality_score_softplus = torch.log(torch.tensor(1 + (torch.e)**factuality_score, dtype=torch.float64).detach())
    consistency_score_softplus = torch.log(torch.tensor(1 + (torch.e)**consistency_score, dtype=torch.float64).detach())
    fusion_score = weight*factuality_score_softplus + (1 - weight)*consistency_score_softplus
    return factuality_score, entailment_score, consistency_score, fusion_score

