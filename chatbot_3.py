import os
import zipfile
import tiktoken
import json

# 参数设置

# 支持跨越多少行寻找目标角色，也即控制段内行间距不超过该值
max_find_lines = 15

max_token_num = 600

# target_role支持 空字符串(默认前三个)或者List of string 如果出错默认保存第一个
target_role = None

# 输入文件路径
input_name = '对话抽取/reorganized_story_人民的名义/reorganized_人民的名义.jsonl'

# 保存路径
savepath = '对话抽取/reorganized_story_人民的名义/role'
# os.system(f"rm -rf {savepath}")
os.makedirs(savepath, exist_ok=True)
# zip_path = '/content/linghuchong_text.zip'

# input_name = '/content/reorganized_xiaoaojianghu.jsonl'
# 用utf8编码为我读取jsonl文件，并且把每一行解析成一个list of json, data
# 请用python为我实现

# # 输入文件路径
# input_name = '/content/shediaoyingxiongzhuan.jsonl'

enc = tiktoken.get_encoding("cl100k_base")

data_in_chunk = []
with open(input_name, encoding='utf8') as f:
    for line in f:
        data_in_chunk.append(json.loads(line))

data = [] # data存储的是所有的dialogue

for chunk in data_in_chunk:
    for d in chunk:
        data.append(d)

print(len(data))
for i,d in enumerate(data):
    if d['role'] == 'scene':
        first_scene_id = i
        break
print(first_scene_id)


# 统计角色出现频率
from collections import Counter

role_counts = Counter()
for line in data:
    role = line['role']
    role_counts[role] += 1

common_roles = role_counts.most_common(30)

status = 0

role_name = [] # role_name存储的是所有的角色

for role, count in common_roles:
    status = status + 1
    if role != 'scene':
        role_name.append(role)
    if status % 3 == 0:
        print(role, count)
    else:
        print(role, count, end=' ')
sorted_roles = sorted(common_roles, key=lambda x: x[1], reverse=True)
sorted_roles_clear = [role[0] for role in sorted_roles if role[0] != "scene"]

# 获取需要提取的人物列表
# 名为role_best的list of str，其按升序记录了所有的角色，
# 请编写一个Python程序，其判断target_role的类型，如果是空字符串或者None，就将role_best前三行提取到一个名为role_extract的list of string；
# 如果target_role为一个list of string，则逐个判断其中的每一个字符串是否在role_best中出现，如果出现，则提取到一个名为role_extract的list of string，否则打印一行错误信息，
# 在遍历完target_role后，如果role_extract不足三个，就用role_best从前往后补充至三个

def extract_roles(role_best, target_role):
    role_extract = []

    if target_role is None or target_role == ""or target_role == [""]:
        role_extract = role_best[:]
    elif isinstance(target_role, list):
        for role in target_role:
            if role in role_best:
                role_extract.append(role)
            else:
                print(f"Error: Role '{role}' not found in role_best!")
    else:
        print("Error: Invalid target_role type!")
        return

    if len(role_extract) < 1:
        additional_roles = role_best[:1 - len(role_extract)]
        role_extract.extend(additional_roles)

    return role_extract
role_extract = extract_roles(sorted_roles_clear, target_role)

from IPython.core.interactiveshell import default_banner
role_in_time = []

dialogue_in_time = []



for d in data:
    role = d['role']
    if role == 'scene':
        continue

    if role in role_name:

        role_in_time.append(role)
        dialogue_in_time.append(d['text'])


import plotly.express as px

# 创建映射字典,角色名字到编号
name_to_num = {name:i for i,name in enumerate(role_name)}

# 将角色名字序列转换为编号序列
role_index = [name_to_num[name] for name in role_in_time]

# 生成不同颜色的颜色序列
colors = px.colors.qualitative.Dark24
while len(colors) < len(role_name):
    colors = colors + colors


# 绘制散点图
fig = px.scatter(
    x=list(range(len(role_in_time))),
    y=role_index,
    color=[colors[i] for i in role_index],
    hover_data={'t': dialogue_in_time}
)

# 设置y轴为角色名字
fig.update_yaxes(ticktext=role_name, tickvals=list(range(len(role_name))))


# 设置坐标轴范围
fig.update_xaxes(range=[0, len(role_in_time)])
# fig.update_yaxes(range=[0, len(role_name)])

# 设置坐标轴标题
fig.update_layout(
    xaxis_title='时间',
    yaxis_title='角色'
)

fig.show()

def output_scene_chat_id(data, target_role_single):

  chat_ids = []

  # 先寻找所有出现角色的节点
  for i,d in enumerate(data):
      if d['role'] == target_role_single:
          chat_ids.append(i)

  previous_scene_ids = []

  # 对于每一个chat_ids，向前寻找scene的节点
  for chat_id in chat_ids:
      ans = first_scene_id
      for j in range(chat_id, first_scene_id,-1):
          if data[j]['role'] == 'scene':
              ans = j
              break
      previous_scene_ids.append(ans)
  # chat_ids为role == target_role的角色对话id，而previous_scene_ids为每一个chat_ids前面的一个chat_id
  return chat_ids, previous_scene_ids

def divide_chats2chunks(chat_ids, previous_scene_ids):
  chat_ids_in_chunk = []
  current_chunk = []

  for chat_id in chat_ids:
      if not current_chunk:
          current_chunk.append(chat_id)
          continue
      # 支持跨越多少行寻找目标角色，也即控制段内行间距不超过该值
      #  max_find_lines = 10
      if chat_id - current_chunk[-1] <= max_find_lines:
          current_chunk.append(chat_id)
      else:
          # 也就是说chat_ids_in_chunk每个角色的行间距都不超过10行
          chat_ids_in_chunk.append(current_chunk)
          current_chunk = [chat_id]

  if current_chunk:
      chat_ids_in_chunk.append(current_chunk)

  # print(chat_ids_in_chunk[0])

  chat_id2previous_scene_id = {}

  for previous, chat_id in zip(previous_scene_ids, chat_ids):
      chat_id2previous_scene_id[chat_id] = previous
      if previous > 0:
          if data[previous-1]['role'] != target_role:
              chat_id2previous_scene_id[chat_id] -= 1
  return chat_ids_in_chunk, chat_id2previous_scene_id

# chat_ids的分块， chat_id对应的旁白id

def count_token( my_str ):
    return len(enc.encode(my_str))
def data2str( data ):
    role = data['role']
    if role in ['旁白', '', 'scene','Scene','narrator' , 'Narrator']:
        return 'scene:' + data['text']
    else:
        return role + ':「' + data['text'] + '」'

def id2texts(data, chat_ids_in_chunk, chat_id2previous_scene_id):
    line_token = [count_token(data2str(d)) for d in data] #每一个chunk的token数量
    from ast import Break
    final_chunks = []

    print_count = 0

    appended_key = []

    for chunk in chat_ids_in_chunk:
        N = len(chunk)

        current_i = 0

        while current_i < N - 1:

            consider_chat_id = chunk[current_i]

            previous_scene_id = chat_id2previous_scene_id[consider_chat_id]

            # 保底
            withdraw_start = previous_scene_id
            withdraw_end = consider_chat_id

            current_count = sum(line_token[previous_scene_id:consider_chat_id + 1])
            # 把当前行内的
            while current_count < max_token_num and current_i < N - 1: # 满足每一段的token数量在一点范围内
                consider_end = chunk[current_i + 1]
                consider_count = sum(line_token[previous_scene_id:consider_end + 1])
                if consider_count < max_token_num:
                    current_count = consider_count
                    withdraw_start = previous_scene_id
                    withdraw_end = consider_end
                    current_i += 1
                else:
                    break

            # print_count += 1

            # print(withdraw_start, end = ' ')
            # if print_count % 5 == 0:
            #     print()

            if withdraw_end + 1 not in appended_key:
                appended_key.append(withdraw_end + 1)
                chunk_str = ''
                for i in range(withdraw_start, withdraw_end + 1):
                    chunk_str += data2str(data[i]) + '\n'

                final_chunks.append(chunk_str)

            current_i += 1
    return appended_key, final_chunks

def save_chunk2zip(savepath, save_title, final_chunks):
    os.makedirs(savepath, exist_ok=True)
    for i in range(0, len(final_chunks)):
        my_str = final_chunks[i]
        with open(savepath + f'/text_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(my_str)
    zip_path = "对话抽取/role/content/" + save_title + "_text.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in os.listdir(savepath):
            zipf.write(savepath + "/" + filename)

    print('Zipped folder saved to', zip_path)

for role_cur_name in role_extract :
  chat_ids, previous_scene_ids = output_scene_chat_id(data, role_cur_name)
  chat_ids_in_chunk, chat_id2previous_scene_id = divide_chats2chunks(chat_ids, previous_scene_ids)
  appended_key, final_chunks = id2texts(data, chat_ids_in_chunk, chat_id2previous_scene_id)
  save_chunk2zip(savepath+"/"+role_cur_name, role_cur_name, final_chunks) #如果你想修改保存的zip名称，请修改本函数的第二个参数save_title

