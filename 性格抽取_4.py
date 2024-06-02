import os
import sys

import openai

openai.api_key = 'sk-VvF4' # 在这里输入你的OpenAI API Token

os.environ["OPENAI_API_KEY"] = openai.api_key

smart_system_prompt = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: 2023-03-15"""

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

role_name = '易学习'
role_name_en = 'yixuexi'
world_name = '人民的名义'

max_predict_token = 2500

role_path = '对话抽取/reorganized_story_人民的名义/role/' + role_name + '/text/'
role_text_path = role_path

#@title 定义get_chunks函数
import os
import codecs

import random

def get_chunks(role_text_path, max_predict_token=2500):

    role_texts = []

    for filename in os.listdir(role_text_path):
        if filename.endswith('.txt'):
            with codecs.open(os.path.join(role_text_path, filename), 'r', 'utf-8') as f:
                role_texts.append(f.read())

    random.shuffle(role_texts)

    role_chunk = []
    chunk = ''
    current_len = 0
    for text in role_texts:
        len_text = len(enc.encode(text))
        if current_len + len_text <= max_predict_token:
            chunk += '\n\n' + text
            current_len += (2 + len_text )
        else:
            role_chunk.append(chunk)
            chunk = text
            current_len = len_text

    # for last chunk add more texts from the head of role_texts
    if chunk:
        for text in role_texts:
            len_text = len(enc.encode(text))
            if current_len + len_text <= max_predict_token:
                chunk += '\n\n' + text
                current_len += (2 + len_text )
            else:
                break
        role_chunk.append(chunk)


    for chunk in role_chunk:
        print(len(enc.encode(chunk)), end = ' ')

    return role_chunk

role_chunk = get_chunks(role_text_path, max_predict_token)

prefix_prompt = f'''
你在分析小说{world_name}中的角色{role_name}
结合小说{world_name}中的内容，以及下文中角色{role_name}的对话
判断{role_name}的人物设定、人物特点以及语言风格
{role_name}的对话:
'''

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key="sk-Ogm6fa3eHsEV9KcfLeLjOyzvl3VVsh8acQg1oqvtDs0kbUQc",
    openai_api_base="https://api.chatanywhere.tech/v1"
)

character = role_name
series = world_name
original_prompt = f'''I want you to act like {character} from {series}.
If others‘ questions are related with the novel, please try to reuse the original lines from the novel.
I want you to respond and answer like {character} using the tone, manner and vocabulary {character} would use.
You must know all of the knowledge of {character}.
'''
print(original_prompt)

responses = []

count = 0

n = 4

for chunk in role_chunk:

    print('index = ', count)

    whole_message = prefix_prompt + "```\n" + chunk + "\n```"

    messages = [
        SystemMessage(content=smart_system_prompt),
        HumanMessage(content=whole_message),
    ]

    if count < 1:
        print(whole_message)
    else:
        response = chat(messages)

        responses.append(response)

    count = count + 1
    if count > n + 1:
        break
    # break


for response in responses:
    print(response.content)
    print('\n----------\n')

