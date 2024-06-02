# coding=gbk
import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

save_folder = "对话抽取/人民的名义_extract"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    print('注意，文件夹',save_folder,'已经存在')

input_name = '对话抽取/人民的名义.txt'

raw_text = open(input_name, encoding='utf-8').read()


chapters = []
chapter_contents = []

for line in raw_text.split('\n'):
    Flag = False
    if line.strip().startswith('第'):
        # 遇到章节标题,将之前章节内容添加到结果列表

        head = line.strip()
        # print(head)
        head = head[:min(10,len(head))]
        if head.find('章',1)>0:
            # print(head)
            Flag = True

    if Flag:
        if chapter_contents:
            chapters.append('\n'.join(chapter_contents))
            chapter_contents = []
        # 记录当前章节标题
        # chapters.append(line)
    else:
        # 累积章节内容
        chapter_contents.append(line)

# 添加最后一个章节内容
if chapter_contents:
    chapters.append('\n'.join(chapter_contents))

print(len(chapters))


for i,ch in enumerate(chapters):
    print(i,ch[:10])


import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

#@title  定义divide函数，用来切分超长文本
def divide_str(s, sep=['\n', '.', '。']):
    mid_len = len(s) // 2  # 中心点位置
    best_sep_pos = len(s) + 1  # 最接近中心点的分隔符位置
    best_sep = None  # 最接近中心点的分隔符
    for curr_sep in sep:
        sep_pos = s.rfind(curr_sep, 0, mid_len)  # 从中心点往左找分隔符
        if sep_pos > 0 and abs(sep_pos - mid_len) < abs(best_sep_pos -
                                                        mid_len):
            best_sep_pos = sep_pos
            best_sep = curr_sep
    if not best_sep:  # 没有找到分隔符
        return s, ''
    return s[:best_sep_pos + 1], s[best_sep_pos + 1:]


def strong_divide(s):
    left, right = divide_str(s)

    if right != '':
        return left, right

    whole_sep = ['\n', '.', '，', '、', ';', ',', '；',\
                 '：', '！', '？', '(', ')', '”', '“', \
                 '’', '‘', '[', ']', '{', '}', '<', '>', \
                 '/', '''\''', '|', '-', '=', '+', '*', '%', \
               '$', '''#''', '@', '&', '^', '_', '`', '~',\
                 '·', '…']
    left, right = divide_str(s, sep=whole_sep)

    if right != '':
        return left, right

    mid_len = len(s) // 2
    return s[:mid_len], s[mid_len:]


#@title 以1500 token为限，切分chunk，输出总chunk数量

max_token_len = 1500
chunk_text = []


for chapter in chapters:

    split_text = chapter.split('\n')

    curr_len = 0
    curr_chunk = ''

    tmp = []

    for line in split_text:
        line_len = len(enc.encode( line ))

        if line_len <= max_token_len - 5:
            tmp.append(line)
        else:
            # print('divide line with length = ', line_len)
            path = [line]
            tmp_res = []

            while path:
                my_str = path.pop()
                left, right = strong_divide(my_str)

                len_left = len(enc.encode( left ))
                len_right = len(enc.encode( right ))

                if len_left > max_token_len - 15:
                    path.append(left)
                else:
                    tmp_res.append(left)

                if len_right > max_token_len - 15:
                    path.append(right)
                else:
                    tmp_res.append(right)

            for line in tmp_res:
                tmp.append(line)

    split_text = tmp

    for line in split_text:
        line_len = len(enc.encode( line ))

        if line_len > max_token_len:
            print('warning line_len = ', line_len)

        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = line
            curr_len = line_len

    if curr_chunk:
        chunk_text.append(curr_chunk)

    # break

print(len(chunk_text))

from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain.chat_models import ChatOpenAI


from langchain.llms import OpenAI
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0,
    openai_api_key="sk-Ogm6fa3eHsEV9KcfLeLjOyzvl3VVsh8acQg1oqvtDs0kbUQc",
    openai_api_base="https://api.chatanywhere.tech/v1"
)

#@title 仅仅抽取dialogue的部分

schema = Object(
    id="script",
    description="Extract Dialogue in order From Novel, ignore the non-dialogue parts",
    attributes=[
        Text(
            id="role",
            description="The character who is speaking, use context to predict the name of the role.",
        ),
        Text(
            id="dialogue",
            description="The dialogue spoken by the characters in the sentence",
        ),
    ],
    examples=[
        (
            '''``这位滑头学长在蔡成功难题解决后又露面了，说是要给他接风，侯亮平心里不悦，只是推托。直到祁同伟说到要在高小琴的山水度假村接风，侯亮平才不动声色地答应了。这真是想睡觉来了枕头，接近神秘的山水集团，他求之不得。

    也不知祁同伟和高小琴是啥关系，竟派高小琴来接。这样，侯亮平便在检察院大门口第一次见到了高小琴。美女老总艳而不俗，媚而有骨，腰肢袅娜却暗藏刚劲，柳眉凤眼流露须眉之气，果然不同凡响！

    侯亮平在车上谈笑风生，称高小琴是京州的一个神话传说。高小琴明眸婉转，于不经意间瞟了侯亮平一眼：侯局长，当传说变成现实的时候，你是不是有些失望？哟，原来就是这么个半老徐娘啊！侯亮平说得诚恳：哪里，是意外啊！你给我的感觉，既像一位风度翩翩的女学者，又像一位叱咤风云的女企业家。高小琴声音甜甜的：在我的神话传说里也有些不友好的，甚至别有用心的故事吧？侯亮平微笑点头：是啊，众说纷纭啊！不过高总，我是一个检察官，我的职业不允许我相信任何神话传说，我只相信自己的眼睛和证据。高小琴含蓄指出，检察官的眼睛有时也会看错，证据也会有假。侯亮平注意到她的话里有话，追问啥意思。高小琴又是嫣然一笑，主动提到他的发小蔡成功：人最不了解的，往往就是自以为很了解的朋友，比如蔡成功。

    嗣后的话题集中在蔡成功身上。高小琴明知蔡成功是他发小，却仍予以悲情控诉。在高小琴的控诉中，蔡成功毫无诚信，谎话连篇，简直不是东西。侯亮平听着，除简短插话，调节气氛，并不争辩。现在他需要倾听，在倾听中发现疑点，找寻线索。他不怕这位美女老总说话，就怕她不说话。经验证明，最难对付的就是沉默的侦查对象。”``''',
            [
                {"role": "侯亮平", "dialogue": "高小琴是京州的一个神话传说。"},
                {"role": "高小琴","dialogue": "侯局长，当传说变成现实的时候，你是不是有些失望？哟，原来就是这么个半老徐娘啊！"},
                {"role": "侯亮平", "dialogue": "哪里，是意外啊！你给我的感觉，既像一位风度翩翩的女学者，又像一位叱咤风云的女企业家。"},
                {"role": "高小琴", "dialogue": "在我的神话传说里也有些不友好的，甚至别有用心的故事吧？"},
                {"role": "侯亮平", "dialogue": "是啊，众说纷纭啊！不过高总，我是一个检察官，我的职业不允许我相信任何神话传说，我只相信自己的眼睛和证据。"},
                {"role": "高小琴", "dialogue": "检察官的眼睛有时也会看错，证据也会有假。"},
                {"role": "侯亮平", "dialogue": "啥意思"},
                {"role": "高小琴", "dialogue": "人最不了解的，往往就是自以为很了解的朋友，比如蔡成功"},
            ],
        ),
        (
            '''``片刻，警方的广播声在夜空中响了起来——大风厂的员工同志们，厂区内的汽油库随时可能发生爆炸，为了你们的人身安全，警方即将执行清场任务，请听到广播后立即离开现场，立即离开现场……

    广播声没起作用。厂门前的男女员工们手挽着手，组成一道道人墙，把十几个假警察包围在警车里，和警方对峙。火光映红了一张张严峻的面孔，许多手机不断闪光，在录像、拍照。这时天阴了下来，月黑星暗，乌云浓厚，仿佛一口黑锅倒扣苍穹。人们被广播声激怒了，益发不肯放过这场灾难的肇事者！工厂被毁，这么多兄弟姐妹被烧死烧伤，这笔账怎能不清算呢？一旦决心拼命，人们比汽油燃起的大火更可怕！假警察们在颤抖，豆大的汗珠从他们的额头脸庞滚落……

    这时，祁同伟再度发出指示：赵局长，鸣枪示警，武力清场！

    谁也没想到，就在这万分紧急的时刻，一位老人阻止了清场。这一夜，陈岩石先是接到郑西坡的告急电话，后来又看到现场视频，得知大风厂出了大事，不顾老伴劝阻，骑电动自行车赶了过来。

    李书记，千万不能莽撞，我们面对的可是工人群众啊！

    李达康很意外：陈老，您怎么来了？这不是您待的地儿，快回去！

    陈岩石伸出手掌：李书记，你给我一只喇叭，我过去劝劝他们。

    祁同伟说：别劝了，现场太乱，很危险，我们马上要清场了……

    陈岩石火了：清啥场？激化矛盾吗？现在也不知烧死烧伤多少人了，再造成新的伤亡吗？快，去找个喇叭来，我把工人劝回厂……

    这时，警方的广播声再次响了起来。
            ``''',
            [
                {"role": "祁同伟", "dialogue": "赵局长，鸣枪示警，武力清场！"},
                {"role": "陈岩石", "dialogue": "李书记，千万不能莽撞，我们面对的可是工人群众啊！"},
                {"role": "李达康", "dialogue": "陈老，您怎么来了？这不是您待的地儿，快回去！"},
                {"role": "陈岩石", "dialogue": "李书记，你给我一只喇叭，我过去劝劝他们。"},
                {"role": "祁同伟", "dialogue": "别劝了，现场太乱，很危险，我们马上要清场了……"},
                {"role": "陈岩石", "dialogue": "清啥场？激化矛盾吗？现在也不知烧死烧伤多少人了，再造成新的伤亡吗？快，去找个喇叭来，我把工人劝回厂……"}
            ],
        ),
        (
            '''``侯亮平按捺着内心的喜悦，表面平静地在屋里踱步。危险尚未完全解除，刘新建还是紧张，高高站在办公桌上，做出随时朝窗外一跃的姿态。侯亮平装着不在意：刘总啊，我知道你是老革命的后代，你爷爷是打鬼子牺牲的，没错吧？刘新建眼睛瞪得老大：没错，我爷爷是“三八式”干部，前年省电视台有个电视剧，说的就是我爷爷的事！侯亮平说：还有你姥姥呢！她当年是京州民族资本家家的大小姐，生长在金窝银窝里，却视金银钱财如粪土，是吧？刘新建眉飞色舞：这你也知道？一点也不错，老人家经常把家里的金条元宝偷出来，把账上的钱转出来，交给京州地下党做经费。侯亮平说：最困难时，组织经费都是你姥姥提供的嘛！你今天跳下去，看九泉之下你姥姥怎么骂你！说着，侯亮平招了招手：下来谈好吗？你站得那么高，我晕。

    刘新建跳下写字台，在大班椅上坐下。气氛得到很大的缓和。

    侯亮平一声叹息，颇动感情地说：刘总，你家前两代人几乎个个都是共产党员，你刘新建也是共产党员，对比一下，你走到今天这一步，比前辈们究竟差了些什么？是不是差了信仰，丧失了信仰啊？

    刘新建表示自己从没丧失过信仰，道是甚至能把《共产党宣言》背下来！说罢，张口就来——一个幽灵，共产主义的幽灵在欧洲游荡。为了对这个幽灵进行神圣的围剿，旧欧洲的一切势力，教皇和沙皇、梅特涅和基佐、法国的激进派和德国警察都联合起来了……``''',
            [
                {"role": "侯亮平", "dialogue": "刘总啊，我知道你是老革命的后代，你爷爷是打鬼子牺牲的，没错吧？"},
                {"role": "刘新建", "dialogue": "没错，我爷爷是“三八式”干部，前年省电视台有个电视剧，说的就是我爷爷的事！"},
                {"role": "侯亮平", "dialogue": "还有你姥姥呢！她当年是京州民族资本家家的大小姐，生长在金窝银窝里，却视金银钱财如粪土，是吧？"},
                {"role": "刘新建", "dialogue": "这你也知道？一点也不错，老人家经常把家里的金条元宝偷出来，把账上的钱转出来，交给京州地下党做经费。"},
                {"role": "侯亮平", "dialogue": "最困难时，组织经费都是你姥姥提供的嘛！你今天跳下去，看九泉之下你姥姥怎么骂你！"},
                {"role": "侯亮平", "dialogue": "下来谈好吗？你站得那么高，我晕"},
                {"role": "侯亮平", "dialogue": "刘总，你家前两代人几乎个个都是共产党员，你刘新建也是共产党员，对比一下，你走到今天这一步，比前辈们究竟差了些什么？是不是差了信仰，丧失了信仰啊？"},
                {"role": "刘新建", "dialogue": "一个幽灵，共产主义的幽灵在欧洲游荡。为了对这个幽灵进行神圣的围剿，旧欧洲的一切势力，教皇和沙皇、梅特涅和基佐、法国的激进派和德国警察都联合起来了……"},
            ],
        )
    ],
    many=True,
)

chain = create_extraction_chain(llm, schema)
print(chain.prompt.format_prompt(text="[user input]").to_string())
text = '''
 李达康再次强调：育良书记，这可不是小事，一定要慎重啊！

    高育良点了点头，又说：省委书记沙瑞金同志刚刚到任，正在下面各市县考察调研呢，我们总不能冷不丁送上这么一份见面大礼吧？

    陈海没想到这一次老师竟如此剑走偏锋，给李达康送偌大一份人情。高育良老师不是不讲原则的人啊，他葫芦里究竟卖的什么药？

    季昌明的性格外柔内刚，表面上谨慎，关键时刻还是敢于表达意见的。他看了看众人，语气坚定地说：高书记、李书记，现在是讨论问题，那我这检察长也实话实说，不论丁义珍一案会给我省造成多大的影响，我们都不宜和最高检争夺办案权，以免造成将来的被动！

    这话意味深长，比较明确地言明了利害，陈海觉得，应该能给老师某种警示。老师却不像得到警示的样子，两眼茫然四顾，也不知在想些啥。陈海便用行动支持自己的领导，及时地看起手腕上的表。他看手表时的动作幅度非常大，似乎就是要让领导们知道他很着急。

    李达康却一点不急，继续打如意算盘，他不同意季昌明的看法，坚持由省纪委先把丁义珍规起来。理由是，双规可以在查处节奏的掌握上主动一些。祁同伟随声附和，称赞李书记这个考虑比较周到……

    陈海实在听不下去了，在祁同伟论证李书记的考虑如何周到时，“呼”地站了起来。行，行，那就规起来吧，反正得先把人控制住……

    不料，高育良瞪了他一眼：陈海！急啥？这么大的事，就是要充分讨论嘛。高书记不到火候不揭锅，批了学生几句，顺势拐弯，端出自己真正的想法——既然产生了意见分歧，就要慎重，就得请示省委书记沙瑞金同志了！说罢，高育良拿起办公桌上的红色保密电话。

    原来是这样！老师这是要把矛盾上交啊！那么老师前边说的话也只是送了李达康一份空头人情，批他这学生也不过做做样子。陈海感叹，老师就是高明，要不怎么能成为H大学和H省官场的不倒翁呢？

    与会者都是官场中人，见高书记拿起红色保密电话，马上知趣地自动避开。李达康是难以改造的老烟枪，心情又特别压抑，现在正好到对面接待室过把瘾。祁同伟上卫生间。季昌明在办公室与卫生间之间的走廊溜达。陈海挂记着现场情况，趁机走出2号楼打电话……

    转眼间，偌大的办公室里只剩下了高育良一个人。

    高育良一边与沙瑞金书记通着电话，一边于不经意间把这些细节记在了脑海中。在以后的日子里，高育良会经常回忆咀嚼当时的情景和细节，琢磨谁是泄密者。的确，这一环节是后来事件演变的关键。

    陈海来到2号楼院子里，深深吸一口气。他内心沮丧懊恼，对自己十分不满意。关键时刻，修炼的火候还是差远了，说着说着就发急，露出一口小狼牙。这么一个汇报会，顶撞了常委李书记，还挨了老师高书记的批，主要领导都对你有看法，还要不要进步了？陈海刻意训练自己，遇事不急于表态，避免得罪人，要成为与父亲不同的人。可江山易改，本性难移，父亲给予的一腔热血总会在一定时候沸腾起来。

    陈海实在忍受不了这无穷无尽的会议。他心急火燎，一晚上嘴角竟起了一个燎泡。万一弄丢了丁义珍，侯亮平真能撕了他！何况这猴子同学又身置花果山，总局的侦查处处长啊。作为省反贪局局长，陈海对总局多一分敬畏，也就对H省这些领导们的拖拉作风多了一分不满。

    关键是一定要盯住丁义珍！陈海为防泄密，出了2号楼以后，才和手下女将陆亦可通了个电话，问那边情况。陆亦可汇报说，宴会进入了高潮，来宾轮番向丁义珍敬酒，场面宏大。说是倘若能把丁义珍灌倒，今晚就万事大吉了。陈海千叮咛万嘱咐，要他们都瞪起眼来。

    开会时一直关机，现在有必要和猴子深入通个气了。这一通气才知道，侯亮平被困机场，反贪总局已将抓捕丁义珍的手续交给侯亮平——既然有手续了，可以先抓人再汇报，猴子的思路可以实施了。陈海不再迟疑，结束和侯亮平的通话后，做出了一个大胆的决定：不等省委意见了，先以传讯的名义控制丁义珍，北京手续一到立即拘捕！

    用手机向陆亦可发出指令后，陈海站在大院里长长吐了一口气。省委大院草坪刚修剪过，空气中弥漫着浓郁的青草香气，这是陈海最喜欢的气息。甬道两旁的白杨树据说是上世纪五十年代种的，合抱粗，仰脸见不到树梢，树叶哗哗啦啦如小孩拍掌，是陈海最爱听的声音。他希望自己变得更完善，更成熟——或者说是更圆滑，但一味瞻前顾后，他总是做不到。做人要有担当，哪怕付出些代价！这一点，陈海从内心佩服侯亮平同学，这猴子同学有股孙悟空天不怕地不怕的劲儿。

    在院子里站一会儿，陈海的心情好多了。夜空中的云彩越来越浓厚，刚才还挂在天际的月亮，现在全不见踪影。要下雨了吧？夜空湿气重了，黑色如漆渐渐涂抹着苍穹。这样的时刻，来一场雨也好。

    再次走进2号楼时，陈海从容淡定。让这些领导们慢慢研究去吧，早点来个先斩后奏就好了，也不这么遭罪。他敢打赌，省委最终决定会与北京一致。又想，陆亦可应该行动了吧？他在心里计算着时间，想象着在宴会上抓捕丁义珍的场面，不由得一阵激动……

    高育良办公室里，人差不多到齐了。老师干咳两声，开始传达新任省委书记沙瑞金的指示：当前的政治环境，反腐是头等大事，要积极配合北京的行动。具体实施，由育良同志代表省委相机决定！

'''
# i = 250
response = chain.run( text )["data"]
print(response)

exit(-1)
import os
import json
from tqdm import tqdm

# save_folder = "/content/drive/MyDrive/GPTData/weixiaobao_extract"

system_prompt = """
Summarize the key points of the following text in a concise way, using bullet points.
"""

q_example = """###
Text:
这位滑头学长在蔡成功难题解决后又露面了，说是要给他接风，侯亮平心里不悦，只是推托。直到祁同伟说到要在高小琴的山水度假村接风，侯亮平才不动声色地答应了。这真是想睡觉来了枕头，接近神秘的山水集团，他求之不得。

也不知祁同伟和高小琴是啥关系，竟派高小琴来接。这样，侯亮平便在检察院大门口第一次见到了高小琴。美女老总艳而不俗，媚而有骨，腰肢袅娜却暗藏刚劲，柳眉凤眼流露须眉之气，果然不同凡响！

侯亮平在车上谈笑风生，称高小琴是京州的一个神话传说。高小琴明眸婉转，于不经意间瞟了侯亮平一眼：侯局长，当传说变成现实的时候，你是不是有些失望？哟，原来就是这么个半老徐娘啊！侯亮平说得诚恳：哪里，是意外啊！你给我的感觉，既像一位风度翩翩的女学者，又像一位叱咤风云的女企业家。高小琴声音甜甜的：在我的神话传说里也有些不友好的，甚至别有用心的故事吧？侯亮平微笑点头：是啊，众说纷纭啊！不过高总，我是一个检察官，我的职业不允许我相信任何神话传说，我只相信自己的眼睛和证据。高小琴含蓄指出，检察官的眼睛有时也会看错，证据也会有假。侯亮平注意到她的话里有话，追问啥意思。高小琴又是嫣然一笑，主动提到他的发小蔡成功：人最不了解的，往往就是自以为很了解的朋友，比如蔡成功。

嗣后的话题集中在蔡成功身上。高小琴明知蔡成功是他发小，却仍予以悲情控诉。在高小琴的控诉中，蔡成功毫无诚信，谎话连篇，简直不是东西。侯亮平听着，除简短插话，调节气氛，并不争辩。现在他需要倾听，在倾听中发现疑点，找寻线索。他不怕这位美女老总说话，就怕她不说话。经验证明，最难对付的就是沉默的侦查对象。”

Summarize in BULLET POINTS form:
"""

a_example = """
- 侯亮平被祁同伟邀请参加接风活动，起初不太愿意，直到祁同伟说到要在高小琴的山水度假村接风, 为了进一步了解山水集团才答应了
- 祁同伟派高小琴来接他，侯亮平第一次见到了她，觉得她美貌与气质不同凡响
- 高小琴暗示侯亮平可能会失望，侯亮平以诚恳回应，但隐约感觉到她有话不说
- 谈话中聚焦在蔡成功身上，高小琴控诉蔡成功缺乏诚信，侯亮平留意探寻线索，他不怕这位美女老总说话，就怕她不说话。
"""
for i in tqdm(range(len(chunk_text))):
    if i < 229:
        continue
    save_name = os.path.join(save_folder, f"{i}_dialogue.txt")

    if not os.path.exists(save_name) or os.path.getsize(save_name) < 5:
        if os.path.exists(save_name):
            print('re-generate dialogue id = ', i)
        query_text = f"``{chunk_text[i]}``"
        dialogue_response = chain.run( query_text )["data"]

        with open(save_name, 'w', encoding='utf-8') as f:
            if 'script' not in dialogue_response:
                print('Error: response does not contain key "script"')
            else:
                for chat in dialogue_response['script']:
                    json_str = json.dumps(chat, ensure_ascii=False)
                    f.write(json_str+"\n")

    save_name_sum = os.path.join(save_folder, f"{i}_sum.txt")

    if not os.path.exists(save_name_sum) or os.path.getsize(save_name_sum) < 5:
        if os.path.exists(save_name_sum):
            print('re-summarize id = ',i )
        #dealing with summarize
        messages = [SystemMessage( content = system_prompt),
                HumanMessage( content = q_example),
                AIMessage( content = a_example)]

        new_input = f"###\nText:\n{chunk_text[ i ]}\nSummarize in BULLET POINTS form:"

        messages.append( HumanMessage(content = new_input) )

        summarize_response = llm( messages ).content

        with open(save_name_sum, 'w', encoding='utf-8') as f:
            f.write( summarize_response )

    raw_text_save_name = os.path.join(save_folder, f"{i}_raw.txt")

    if not os.path.exists(raw_text_save_name) or os.path.getsize(raw_text_save_name) < 5:
        with open(raw_text_save_name, 'w', encoding='utf-8') as f:
            f.write( chunk_text[i] )

    # if i >5:
    #     break
