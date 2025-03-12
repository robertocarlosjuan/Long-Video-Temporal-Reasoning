def get_prompt_template(d, version, media=None, step=0):
    if version == "nvila_responses":
        text = f'Question: {d["question"]}\nOptions:\n1. {d["answer_choice_0"]}\n2. {d["answer_choice_1"]}\n3. {d["answer_choice_2"]}\n4. {d["answer_choice_3"]}\n5. {d["answer_choice_4"]}'
        prompt = [media, text]
        return prompt
    elif version == "nvila_constrained":
        text = f'Pick a correct option to answer the question.\nQuestion: {d["question"]}\nOptions:\n1. {d["answer_choice_0"]}\n2. {d["answer_choice_1"]}\n3. {d["answer_choice_2"]}\n4. {d["answer_choice_3"]}\n5. {d["answer_choice_4"]}'
        prompt = [media, text]
        return prompt
    elif version == "nvila_mallmletters":
        text = f'Question: select the correct option for this task: {d["question"]}\nOptions:\n(a) {d["answer_choice_0"]}\n(b) {d["answer_choice_1"]}\n(c) {d["answer_choice_2"]}\n(d) {d["answer_choice_3"]}\n(e) {d["answer_choice_4"]}\nAnswer:'
        prompt = [media, text]
        return prompt
    elif version == "nvila_reason":
        text = f'Question: select the correct option for this task: {d["question"]}\nOptions:\n(a) {d["answer_choice_0"]}\n(b) {d["answer_choice_1"]}\n(c) {d["answer_choice_2"]}\n(d) {d["answer_choice_3"]}\n(e) {d["answer_choice_4"]}\nOutput format: [OPTION]: [Reason]'
        prompt = [media, text]
        return prompt
    elif version == "nvila_selectexplain":
        text = f'Question: select the correct option for this task and explain your answer: {d["question"]}\nOptions:\n(a) {d["answer_choice_0"]}\n(b) {d["answer_choice_1"]}\n(c) {d["answer_choice_2"]}\n(d) {d["answer_choice_3"]}\n(e) {d["answer_choice_4"]}\nAnswer:'
        prompt = [media, text]
        return prompt
    elif version == "nvila_llavao1":
        texts = [
            f'Briefly explain what steps you\'ll take to answer the question: {d["question"]}',
            f'Describe the contents of the video.',
            f'Follow the steps to answer the question.',
            f'Question: select the correct option for this task: {d["question"]}\nOptions:\n(a) {d["answer_choice_0"]}\n(b) {d["answer_choice_1"]}\n(c) {d["answer_choice_2"]}\n(d) {d["answer_choice_3"]}\n(e) {d["answer_choice_4"]}'
        ]
        prompts = [
            [texts[0]],
            [media, texts[1]],
            [texts[2]],
            [texts[3]],
        ]
        return prompts[step]
    elif version == "nvila_describechoose":
        if 'intermediate_response' not in d:
            d['intermediate_response'] = ["", ""]
        texts = [
            f'T',
            f'The following is a description of the video.\n{d["intermediate_response"][0]}\nQuestion: select the correct option for this task: {d["question"]}\nOptions:\n(a) {d["answer_choice_0"]}\n(b) {d["answer_choice_1"]}\n(c) {d["answer_choice_2"]}\n(d) {d["answer_choice_3"]}\n(e) {d["answer_choice_4"]}\nAnswer:'
        ]
        prompts = [
            [media, texts[0]],
            [media, texts[1]],
        ]
        return prompts[step]
    elif version == "nvila_llavao1edited":
        if 'intermediate_response' not in d:
            d['intermediate_response'] = ["", "", "", ""]
        texts = [
            f'Describe the contents of the video.',
            f'The following is a description of a video.\n{d["intermediate_response"][0]}\nGiven that the video has all the information you need, briefly explain what steps you\'ll take to answer this question: {d["question"]}',
            f'Follow the steps below to answer the question \"{d["question"]}\".\n{d["intermediate_response"][1]}',
            f'{d["intermediate_response"][2]}\nQuestion: select the correct option for this task: {d["question"]}\nOptions:\n(a) {d["answer_choice_0"]}\n(b) {d["answer_choice_1"]}\n(c) {d["answer_choice_2"]}\n(d) {d["answer_choice_3"]}\n(e) {d["answer_choice_4"]}\nAnswer:'
        ]
        prompts = [
            [media, texts[0]],
            [texts[1]],
            [media, texts[2]],
            [media, texts[3]],
        ]
        return prompts[step]
    elif version == 'nvila_tomatoprompt':
        index2ans = {'A':d["answer_choice_0"], 'B':d["answer_choice_1"], 'C':d["answer_choice_2"], 'D':d["answer_choice_3"], 'E':d["answer_choice_4"]}
        text = f"""You will be provided with a video. Analyze the video and provide the answer to the question about the video content. Answer the multiple-choice question about the video content. 

You must use the video to answer the multiple-choice question; do not rely on any externel knowledge or commonsense. 

<question> 
{d["question"]} 
</question>

<options> 
{index2ans} 
</options>

Even if the information in the video is not enough to answer the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION. 

DO NOT GENERATE ANSWER SUCH AS 'NOT POSSIBLE TO DETERMINE.' 
"""
        prompt = [media, text]
        return prompt
    else:
        print("undefined prompt_template")

# d = {
#     "question":1, 
#     "answer_choice_0": 2,
#     "answer_choice_1": 2,
#     "answer_choice_2": 2,
#     "answer_choice_3": 2,
#     "answer_choice_4": 2,
#     }
# print(get_prompt_template(d, "nvila_constrained"))
            
            
