import openai
import time


def GPT_onetime_chat_message_score(prefix_txt, response, unrelated_words_txt, judge_messages_write_txt, data_name, api_KEY, base_url, model_name, temperature=1.0, logprobs=True, top_p=1.0):
    openai.api_key = api_KEY
    openai.api_base = base_url

    # open the unrelated_words_txt
    with open(unrelated_words_txt, 'r') as f:
        items = f.readlines()
        for item in items:
            item = str(item)
            segments_message_data = response.replace(item.split('\n')[0], '')
    segments_message_list = segments_message_data.split(".")  # divide every sentence
    segments_message_list = segments_message_list[:-1]  # abandon the last '.'

    # number the segments
    segments = []
    for i, segment in enumerate(segments_message_list):
        content = f"{i+1}.{segment}"
        segments.append(content)

    # link the prefix with the segments
    with open(prefix_txt, 'r') as ff:
        prefix = ff.read()
    input = prefix
    for item in segments:
        input = input + item+'\n'
    # print(input)

    # put the 'input' to the GPT3.5-turbo
    onechat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user",
                     "content": input}],
        max_tokens=2048,
        temperature=temperature,
        logprobs=logprobs,
        top_p=top_p
    )

    # divide the judgements and the explanations
    output = onechat_completion["choices"][0]["message"]["content"]
    contents = output.split('\n')
    contents_lens = len(contents)
    judgements = contents[1].replace(' ', '').split(',')

    explanations = []
    for expl_index in range(3, contents_lens):
        explanations.append(contents[expl_index])

    # output the judgements and the explanations for evaluation
    # print("Judgements----------------------------")
    # print(judgements)
    # print("Exaplanations-------------------------")
    # print(explanations)

    # write to txt
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    with open("{}/{}/judge_messages_in_Loop1_{}.txt".format(judge_messages_write_txt, data_name, model_name), 'a') as f:
        f.write(formatted_time+'\n')
        f.write("judgements: "+str(judgements)+'\n')
        f.write("explanations:\n"+str(explanations)+'\n')

    # score rules
    # True: 1, False: -1, Neither: 0
    Score_rules_dict = {'True':int(1), 'False': int(-1), 'Neither': int(0)}
    judge_lens = len(judgements)
    # total_score
    total_score = 0
    for i in judgements:
        total_score += Score_rules_dict[i]
    # avg_score
    avg_score = total_score / judge_lens*1.0
    return avg_score
