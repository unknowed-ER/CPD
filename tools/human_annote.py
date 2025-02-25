import sys
import gradio as gr
import argparse
import warnings
import time
import os
import json
from dataclasses import dataclass,field
from datetime import datetime
import utils

# ，1，chat_cmd.sh
# 2
FORMAT=2
# review
# TODO chatgpt/gpt4 review
if FORMAT==1:
    Files=[
        '/home/lzy/Chinese-Vicuna/outs/outs/7b_sharegpt-a100-1/chat.json',
        '/home/lzy/Chinese-Vicuna/outs/outs/7b-sharegpt-a100-2/chat_llama-7b_with_eos.json',
        '/home/lzy/Chinese-Vicuna/outs/7b-sharegpt-0502/chat_fix.jsonl',
        # '/home/lzy/Chinese-Vicuna/outs/outs/7b-sharegpt-a100-2/chat_yahma_llama_7b.jsonl',
    ]
else:
    DIR='/mnt/wsl/PHYSICALDRIVE2/disentangle/dialogue/LLM/Chinese-Vicuna/outs/test/20230508/'
    file_path = '/mnt/wsl/PHYSICALDRIVE2/disentangle/dialogue/LLM/Chinese-Vicuna/outs/test/20230508/test_20230507-format2.json'
    pre_data = utils.from_json(file_path)
    Files = ['ChatGPT','WenXin','Chatglm','Vicuna','Chinese-LLaMA-Alpaca','7b-sharegpt-4090-2/chat.json','7b-sharegpt-a100-2/chat.json','7b_sharegpt-a100-1/chat_llama_7b.json', '7b_sharegpt-a100-1/chat_yamha.json']
NUM=len(Files)
TIME=datetime.today().strftime("%Y%m%d-%h%m")
print('load:', Files)
print('Num:',NUM)

@dataclass
class ColItem:
    """Class for keeping track of an item in inventory."""
    col_index: int = 0
    name: str = ''
    ori_texts: list = field(default_factory=list) 
    texts: list = field(default_factory=list) 

    def load(self, file_or_key):
        if file_or_key=='':
            return 
        if FORMAT==1:
            datas = utils.from_jsonl(file_or_key)
        elif FORMAT==2:
            datas = pre_data[file_or_key]
        self.texts = []
        self.name = file_or_key
        self.col_index = 0
        
        self.ori_texts = datas
        for data in datas:
            self.texts.append([(d['input'], d['output']) for d in data['history']]) 
        return f'now:\t{self.col_index+1}/total:\t{len(self.texts)}',self.texts[0]
    
    def get(self, index):
        self.col_index = index
        return self.texts[index]

    def conv(self,):
        return self.ori_texts[self.col_index]['history']

    def up(self, ):
        if self.col_index >= len(self.texts) -1 :
            self.col_index = -1
        self.col_index += 1
        return f'now:\t{self.col_index+1}/total:\t{len(self.texts)}', self.texts[self.col_index] 
    
    def down(self,):
        if self.col_index <= 0 :
            self.col_index = len(self.texts)
        self.col_index -= 1
        return f'now:\t{self.col_index+1}/total:\t{len(self.texts)}', self.texts[self.col_index] 

class Viewer:
    cols = []

    def __init__(self, col_num):
        for i in range(col_num):
            self.cols.append(ColItem())

    def next(self,):
        show_texts = [col.up() for col in self.cols]
        return [j for i in show_texts for j in i]

    def prev(self,):
        show_texts = [col.down() for col in self.cols]
        return [j for i in show_texts for j in i]
    
    def load(self, *files):
        ans = []
        for col,file in zip(self.cols, files):
            if file == '':
                ans.append(None, None)
            else:
                ans.append(col.load(file))
        return [j for i in ans for j in i]
    
    def save(self, *args):
        ans = []
        col_num = len(self.cols)
        scores = args[:col_num]
        comments = args[col_num:]
        for i in range(col_num):
            ans.append({
                'index': self.cols[i].col_index,
                'name': self.cols[i].name,
                'score': int(scores[i]),
                # 'comment': comments[i],
                'conv': self.cols[i].conv(),
            })
        utils.to_jsonl([{
            'comment': comments[0],
            'result': ans,
        }], f'{DIR}/score_{TIME}.jsonl')

css = """html *
{
   font-size: 16px !important;
}"""
viewer = Viewer(NUM)
collect = [str(i) for i in range(NUM)]
with gr.Blocks(css=css) as demo:
    title = gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>"
        + "Chinese-Vicuna "
        + "</h1>"
    )
    description = gr.Markdown(
        'used in: https://github.com/Facico/Chinese-Vicuna'
    )
    files = []
    chatbots = []
    infos = []
    scores = []
    comments = []
    # with gr.Row().style(equal_height=False): 
    with gr.Row(variant='panel'):
        global_load_btn = gr.Button('Load').style(full_width=False,size='sm')
        global_save_btn = gr.Button('save').style(full_width=False,size='sm')
        global_prev_btn = gr.Button('\u2B05').style(full_width=False,size='sm')
        global_next_btn = gr.Button('\u27A1').style(full_width=False,size='sm')
    with gr.Row(variant='panel'):
        comments = gr.components.Textbox(
            lines=1, label="comment", placeholder="."
        )    
    with gr.Row().style(equal_height=False):        
        for i in range(NUM):
            with gr.Column(min_width=100): #  gr.Column() has a min_width of 320px. If you reduce this, you should be able to fit more columns in your screen.
                file = gr.Textbox(lines=1, value=Files[i], label='Path', placeholder='').style(container=False,border=False,rounded=False)
                info = gr.Markdown()
                chatbot = gr.Chatbot().style(height=768)
                with gr.Row():
                    prev_btn = gr.Button(' ⮜ ').style(full_width=False,size='sm')
                    next_btn = gr.Button(' ⮞ ').style(full_width=False,size='sm')
                prev_btn.click(viewer.cols[i].down, None, [info,chatbot], show_progress=False)
                next_btn.click(viewer.cols[i].up, None, [info,chatbot], show_progress=False)
                
                score = gr.components.Dropdown(
                    collect, label="Score",
                )
                # comment = gr.components.Textbox(
                #     lines=1, label="comment", placeholder="."
                # )
            scores.append(score)
            # comments.append(comment)
            files.append(file)
            infos.append(info)
            chatbots.append(chatbot)

    global_next_btn.click(
        viewer.next, 
        None,
        [j for i in zip(infos,chatbots) for j in i],
    )
    global_prev_btn.click(
        viewer.prev, 
        None,
        [j for i in zip(infos,chatbots) for j in i],
    )
    # wrong: list(zip(infos, chatbots))
    global_load_btn.click(viewer.load, files, [j for i in zip(infos,chatbots) for j in i])
    global_save_btn.click(
        viewer.save, 
        scores + [comments],
        None,
    )
demo.launch(share=1, server_port=8089)
