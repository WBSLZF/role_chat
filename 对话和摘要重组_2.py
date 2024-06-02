import numpy as np
import os
import copy
import json


# 储存txt和jsonl的文件夹路径。如需修改，请与下方自动化循环保持一致

save_folder_path =  "对话抽取/reorganized_story_人民的名义"

# chunk所在文件夹，请以_raw结尾
folder_path = "对话抽取/人民的名义_extract"


# 故事名字，默认为_raw之前的名字
story_name_en = os.path.basename(folder_path).split("_")[0]

# # 测试ID
# id = 1

# 默认的保存路径
save_jsonl_path = f"对话抽取/reorganized_story_{story_name_en}/reorganized_{story_name_en}.jsonl"
save_txt_path = f"对话抽取/reorganized_story_{story_name_en}/reorganized_{story_name_en}.txt"

# 默认抽取出的dialogue和summary文件位置/如果有不同请在此处和底部自动程序中修改
save_folder = f"对话抽取/{story_name_en}_extract"

# dialoge_file = os.path.join(save_folder, f"{id}_dialogue.txt")
# summarzie_file = os.path.join(save_folder, f"{id}_sum.txt")


folder_path = f"对话抽取/人民的名义_extract/"
chunk_text = []

import os

chunk_text = [""] * (sum([1 for file_name in os.listdir(folder_path) if file_name.endswith("_sum.txt")]) + 1)

# 遍历文件夹中的文件
for file_name in os.listdir(folder_path):
    if file_name.endswith("_raw.txt"):
        file_path = os.path.join(folder_path, file_name)

        # 提取文件名中的 i
        i = int(file_name.split('_')[0])

        # 打开文件并读取内容
        with open(file_path, 'r',encoding='utf-8') as file:
            text = file.read()

        # 将文件内容添加到列表中，按 i 的顺序插入到正确位置
        # 一开始chunk_text存储的是raw_text
        chunk_text[i] = text

print(len(chunk_text))
#
#
# raw_text = chunk_text[ id ]


# chunk_sum = []
# unique_chunk_sum = []
# 给定summarzie_file = os.path.join(save_folder, f"{id}_sum.txt")

# 先检查这个文件是否存在

# 然后使用utf-8编码打开，检查每一行，如果strip后，行首是'-'，则把后面的字符串append到一个list chunk_sum中

# 请用python为我实现
# if os.path.exists(summarzie_file):
#     with open(summarzie_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip().startswith('-'):
#                 # - 描述了钱塘江边的景色和村民聚集听老者讲故事的情景去掉 -
#                 chunk_sum.append(line.strip()[1:].strip())
# if os.path.exists(dialoge_file):
#   with open(dialoge_file, encoding='utf-8') as f:
#     dialogues = []
#     for line in f:
#         dialogue = json.loads(line)
#         dialogues.append(dialogue)
#
# unique_dialogue = [] # 不重复的对话
# for item in dialogues:
#     if item not in unique_dialogue:
#         unique_dialogue.append(item)
# dia_texts = [data['dialogue'] for data in unique_dialogue]
#
# for item in chunk_sum:
#     if item not in unique_chunk_sum: # 不重复的总结
#         unique_chunk_sum.append(item)
#
# chunk_sum = unique_chunk_sum
# dialogues = unique_dialogue
# print(dialogues) # 不重复的对话，包含[{"role":"","dialogue":""}]
# print(dia_texts) # 不重复的dialogue文本["dialogue_text"]
# print(chunk_sum) # 不重复的总结["summary"]


# 给定长文本raw_text。

# 使用换行符\n或者。 来对这个字符串进行切割，忽略掉strip之后是空的子字符串

# 将每一段话的起点位置存储在一个list of int , starts中

# 将每一段话的结束位置存储在一个list of int , ends中

# 并且将每一个子字符串的存储在一个list of str, lines中

def divide_raw2lines(raw_text):
  previous_str = ''
  starts = []
  ends = []
  lines = []
  for i in range(len(raw_text)):
      previous_str += raw_text[i]
      if raw_text[i] in ('\n','。'):
          strip_str = previous_str.strip(' "“”\r\n')
          if len(strip_str)>0:
              lines.append(strip_str)
              starts.append(i - len(strip_str))
              ends.append(i)
          previous_str = ''
      else:
          pass
  return lines , starts , ends

#
# lines, starts, ends = divide_raw2lines(raw_text)
#
# print(len(lines))

# 已知if '\u4e00' <= char <= '\u9fa5': 可以判断一个char是否是中文字

# 我希望实现一个函数，这个函数的输入是两个list of string, 长度为M的query 和 长度为N的datas

# 输出是一个M*N的numpy float数组 recalls

# 先计算freqs[m][n] 表示query的第m句中的每一个中文字，是否在datas[n]中是否出现，如果出现，则freqs[m][n]加一

# 然后计算recalls[m][n]是freqs[m][n]除掉 query[m]中所有中文字的个数

# query为总结的summary或者dialogue，有m个字符串
# datas为raw_text的lines，有n个字符串
def compute_char_recall(query, datas):
    M = len(query)
    N = len(datas)

    freqs = np.zeros((M, N), dtype=int)

    # q_chars为chunk_sum中所有的中文字符
    for m in range(M):
        q_chars = set()
        for char in query[m]:
            if '\u4e00' <= char <= '\u9fa5':
                q_chars.add(char)

        for n in range(N):
            for char in q_chars:
                if char in datas[n]:
                    freqs[m][n] += 1

    query_chars_count = [len(set(char for char in sent if '\u4e00'<= char <= '\u9fa5'))
                         for sent in query]

    recalls = freqs / np.array(query_chars_count)[:, None]

    return recalls


# import plotly.express as px
#
# s = compute_char_recall(chunk_sum, lines)
#
# fig = px.imshow(s,
#                 labels=dict(x="Line", y="Summary", color="Recall"),
#                 x=list(range(len(lines))),
#                 y=list(range(len(chunk_sum))),
#                 color_continuous_scale='YlOrRd')
#
# fig.update_xaxes(side="top")
#
# fig.show()
def summary2line(chunk_sum, lines):


  s = compute_char_recall(chunk_sum, lines)

  color_map = {}
  ans_Q = {}

  ans_div = {}

  flags = {}

  M = len( chunk_sum )
  N = len( lines )
  # ans_Q[0,0] = 0
  # print(s[0,0])
  # ans_Q[(0, 0)] = s[0, 0]
  for n in range(0, N):
      if n==0:
          ans_Q[ (0,0) ] = s[0,0]
          ans_div[ (0,0 ) ] = []
      else:
          ans_Q[ (0,n) ] = ans_Q[ (0,n-1) ] + s[0,n]  # s[0,n]是什么东西
          ans_div[ (0,n) ] = []

  for m in range(1,M):
      ans_Q[(m,m)] = ans_Q[(m-1,m-1)] + s[m,m]
      ans_div[ (m,m) ] =  ans_div[ (m-1,m-1) ].copy()
      ans_div[ (m,m) ].append(m)


  def find_Q( m , n ):
      # print(m,n)

      if m < 0 or n < 0:
          print('error out bound', m , ' ' , n )
          return 0, []

      if (m,n) in ans_Q.keys():
          return ans_Q[(m,n)], ans_div[(m,n)]

      if (m,n) in color_map.keys():
          print('error repeated quest ', m , ' ', n )
          return 0, []
      else:
          color_map[(m,n)] = 1

      current_div = []

      left, left_div = find_Q( m, n-1 )
      right, right_div = find_Q( m-1, n-1 )

      if left > right:
          ans = left + s[m][n]
          flags[(m,n)] = False
          current_div = left_div

      else:
          ans = right + s[m][n]
          flags[(m,n)] = True
          current_div = right_div.copy()
          current_div.append(n-1)

      # ans = max(  , ) + s[m][n]

      ans_Q[(m,n)] = ans
      ans_div[(m,n)] = current_div.copy()

      return ans, current_div

  # print(find_Q(0,5))
  # print(find_Q(M-1,N-1))

  score, divs = find_Q(M-1,N-1)
  divs.append(N-1)

  return score, divs


def dialogue2line(dia_texts, lines):
    s_dialogue = compute_char_recall(dia_texts, lines)
    # s_dialogue存储了一个M*N的nparray
    # 我们希望实现一个python程序找到长度为M的顺序子序列a0,a1,...,am-1
    # 使得s_dialogue[i][ai]之和最大
    # 输出a0,..., am-1的值
    M, N = s_dialogue.shape
    if M==0 or N==0:
      return []
    dp = np.zeros((M, N))
    dp[0] = s_dialogue[0] # dp[0]为第0个对话所匹配的召回率
    prev_indices = np.zeros((M, N), dtype=int)
    for i in range(1, M):
        for j in range(N):
            max_prev_index = np.argmax(dp[i-1])
            dp[i][j] = dp[i-1][max_prev_index] + s_dialogue[i][j]
            prev_indices[i][j] = max_prev_index

    max_end_index = np.argmax(dp[-1])
    sequence = []
    for i in range(M-1, -1, -1):
        sequence.append(max_end_index)
        max_end_index = prev_indices[i][max_end_index]
    sequence.reverse()

    return sequence
def jsonl_sorted(chunk_sum, divs, dia_texts, seq):

  combined_data = []
  combined_text = ""
  for index in sorted(seq + divs):
      # print(index)
      if index in seq: # seq中为对话信息

          combined_data.append({
              "role" : dialogues[seq.index(index)]["role"],
              'text': dialogues[seq.index(index)]["dialogue"],
              'if_scene': False
          })
          combined_text = combined_text + dialogues[seq.index(index)]["role"] + ":" + dialogues[seq.index(index)]["dialogue"] +"\n"
          seq[seq.index(index)] = -1
      if index in divs: # divs中为旁白
          combined_data.append({
              "role" : "scene" ,
              'text': chunk_sum[divs.index(index)],
              'if_scene': True
          })
          combined_text = combined_text + "scene" + ":" + chunk_sum[divs.index(index)] +"\n"
          divs[divs.index(index)]=-1

  return combined_data, combined_text
import numpy as np
import os
import copy
import json
from tqdm import tqdm

final_jsonl = []
final_txt = ""

for i in tqdm(range(1,len(chunk_text)), desc="Processing", total=len(chunk_text)-1, unit="item"):

  # try:
    # story_name_en = 'shediaoyingxiongzhuan'
    raw_text = chunk_text[ i ]

    import os

    save_folder = f"对话抽取/人民的名义_extract"

    dialoge_file = os.path.join(save_folder, f"{i}_dialogue.txt")
    summarzie_file = os.path.join(save_folder, f"{i}_sum.txt")

    chunk_sum = []
    unique_chunk_sum = []
    # chunk_summarize的段
    if os.path.exists(summarzie_file):
        with open(summarzie_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('-'):
                    chunk_sum.append(line.strip()[1:].strip())

    # 对话信息的存储
    if os.path.exists(dialoge_file):
      with open(dialoge_file, encoding='utf-8') as f:
        dialogues = []
        for line in f:
            dialogue = json.loads(line)
            dialogues.append(dialogue)

    # 不重复的对话，以及总结
    unique_dialogue = []
    for item in dialogues:
        if item not in unique_dialogue:
            unique_dialogue.append(item)
    dia_texts = [data['dialogue'] for data in unique_dialogue]
    for item in chunk_sum:
        if item not in unique_chunk_sum:
            unique_chunk_sum.append(item)

    chunk_sum = unique_chunk_sum
    dialogues = unique_dialogue
    lines, starts, ends = divide_raw2lines(raw_text)
    # print(chunk_sum)
    # print(lines)

    # chunk_sum: 不重复的chunk的总结
    # lines: 原始raw_text进行的切割
    # dia_texts: 不重复的对话
    try:
        score, divs = summary2line(chunk_sum, lines)  #summary匹配
        seq = dialogue2line(dia_texts, lines) #对话匹配
        combined_data, combined_text = jsonl_sorted(chunk_sum, divs.copy(), dia_texts, seq.copy())
        # 如果需要保存每个chunk的，在此处保存
        final_jsonl.append(combined_data)
        final_txt = final_txt + combined_text + "\n"
    except Exception as e:
        print(e)
        print("第" + str(i) + "个chunk出错")
        print(chunk_sum)
        print('----------------------------------')
        print(lines)
        print('**********************************')
        pass
with open(save_jsonl_path, "w", encoding="utf-8") as file:
  # 遍历数据列表中的每个字典
  for record in final_jsonl:
      # 将字典转换为JSON格式的字符串
      json_record = json.dumps(record, ensure_ascii=False)
      # 将转换后的JSON字符串写入文件，并添加换行符
      file.write(json_record + "\n")
with open(save_txt_path, "w", encoding="utf-8") as file:
  file.write(final_txt)