import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import json
import gradio as gr
import argparse
import warnings
import os
from datetime import datetime
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM, printf
import utils
import copy
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import prompt
import middleware

tokenizer = None
model = None
args = None

def preload(args):

    # NOTE don't add eos !
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    assert tokenizer.eos_token_id == 2
    assert tokenizer.bos_token_id == 1
    tokenizer.pad_token_id = 0 

    torch_dtype = torch.bfloat16 if args.dtype=='bf16' else torch.float16
    use_int8 = args.dtype=='int8'
    if args.checkpoint_dir:
        print(f'>>> load llama models from {args.checkpoint_dir}')
        model = StreamLlamaForCausalLM.from_pretrained(
            args.checkpoint_dir,
            load_in_8bit=use_int8,
            torch_dtype=torch_dtype,
            device_map="auto",
        )    
    elif args.lora_path:
        print(f'>>> load lora models from {args.lora_path}')
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            load_in_8bit=use_int8,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        model = StreamPeftGenerationMixin.from_pretrained(
            model, args.lora_path, 
            torch_dtype=torch_dtype, 
            load_in_8bit=use_int8,  
            device_map="auto",
        )
    else:
        print(f'>>> load llama models from {args.model_path}')
        model = StreamLlamaForCausalLM.from_pretrained(
            args.model_path,
            load_in_8bit=use_int8,
            torch_dtype=torch_dtype,
            device_map="auto",
        )    
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer
    
# ------------------------------- commandline -------------------------------------
@torch.inference_mode()
def chat_cmd():
    datas = utils.from_jsonl(args.data_path)
    mid = middleware.InferMiddleWare(model, tokenizer)
    ans = mid.infer_chat(datas, args.prompt_type)
    utils.to_jsonl(ans, args.out_path)

# ------------------------------ gradio code --------------------------------------

def save(
    inputs,
    extra,
    history,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    max_memory=1024,
    do_sample=False,
    prompt_type='0',
    **kwargs, 
):
    history = [] if history is None else history
    data_point = {}
    if prompt_type == 'instruct':
        PROMPT = prompt.instruct_prompt(tokenizer,max_memory)
    elif prompt_type == 'chat':
        PROMPT = prompt.chat_prompt(tokenizer,max_memory)
    elif prompt_type == 'chat with knowledge':
        PROMPT = prompt.chat_context_prompt(tokenizer,max_memory)
        data_point['context'] = extra
    elif prompt_type == 'chat with persona':
        PROMPT = prompt.chat_persona_prompt(tokenizer,max_memory)
        data_point['persona'] = extra
    else:
        raise Exception('not support')
    data_point['history'] = history
    data_point['generation_parameter'] = {
        "temperature":temperature,
        "top_p":top_p,
        "top_k":top_k,
        "num_beams":num_beams,
        "bos_token_id":tokenizer.bos_token_id,
        "eos_token_id":tokenizer.eos_token_id,
        "pad_token_id":tokenizer.pad_token_id,
        "max_new_tokens":max_new_tokens,
        "min_new_tokens":min_new_tokens, 
        "do_sample":do_sample,
        "repetition_penalty":repetition_penalty,
        "max_memory":max_memory,
    }
    data_point['info'] = args.__dict__
    print(data_point)
    # save_dir = f"outs/{args.prompt_type.replace(' ','_')}"
    # os.makedirs(save_dir, exist_ok=True)
    # utils.to_jsonl([data_point], f"{save_dir}/{datetime.now()}.jsonl")
    file_name = f"{args.lora_path}/{args.prompt_type.replace(' ','_')}_{args.dtype}.jsonl"
    utils.to_jsonl([data_point], file_name, mode='a')

@torch.inference_mode()
def evaluate(
    inputs,
    extra,
    history,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    max_memory=1024,
    do_sample=False,
    prompt_type='chat',
    **kwargs,
):
    mid = middleware.InferMiddleWare(
        model, 
        tokenizer,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        max_memory=max_memory,
        do_sample=do_sample,
    )   
    for output, history in mid.infer_gradio(
        inputs, 
        history,
        prompt_type=prompt_type,
        use_typewriter=args.use_typewriter,
    ):
        yield output, history

def clear():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return None, None

def chat():

    # gr.Interface has bugs to clear chatbot's history,so we customly implement it based on gr.block
    with gr.Blocks() as demo:
        fn = evaluate
        title = gr.Markdown(
            "<h1 style='text-align: center; margin-bottom: 1rem'>"
            + "Chinese-Vicuna "
            + "</h1>"
        )
        description = gr.Markdown(
            "instruction，Alpaca-lora，llama7B，lora。，llamalora。"
        )
        history = gr.components.State()
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):
                input_component_column = gr.Column()
                with input_component_column:
                    input = gr.components.Textbox(
                        lines=2, label="Input", placeholder="."
                    )
                    extra = gr.components.Textbox(
                        lines=2, label="Extra", placeholder="context or persona"
                    )
                    temperature = gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature")
                    topp = gr.components.Slider(minimum=0, maximum=1, value=0.9, label="Top p")
                    topk = gr.components.Slider(minimum=0, maximum=100, step=1, value=60, label="Top k")
                    beam_number = gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="Beams Number")
                    max_new_token = gr.components.Slider(
                        minimum=1, maximum=2048, step=1, value=256, label="Max New Tokens"
                    )
                    min_new_token = gr.components.Slider(
                        minimum=1, maximum=1024, step=1, value=5, label="Min New Tokens"
                    )
                    repeat_penal = gr.components.Slider(
                        minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
                    )
                    max_memory = gr.components.Slider(
                        minimum=0, maximum=2048, step=1, value=2048, label="Max Memory"
                    )
                    do_sample = gr.components.Checkbox(label="Use sample")
                    # must be str, not number !
                    type_of_prompt = gr.components.Dropdown(
                        ['instruct', 'chat','chat with knowledge','chat with persona'], value='chat', label="Prompt Type", info="select the specific prompt; use after clear history"
                    )
                    input_components = [
                        input, extra, history, temperature, topp, topk, beam_number, max_new_token, min_new_token, repeat_penal, max_memory, do_sample, type_of_prompt
                    ]
                    input_components_except_states = [input,extra, temperature, topp, topk, beam_number, max_new_token, min_new_token, repeat_penal, max_memory, do_sample, type_of_prompt]
                with gr.Row():
                    cancel_btn = gr.Button('Cancel')
                    submit_btn = gr.Button("Submit", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop", visible=False)
                with gr.Row():
                    reset_btn = gr.Button("Reset Parameter")
                    clear_history = gr.Button("Clear History")

            with gr.Column(variant="panel"):
                chatbot = gr.Chatbot().style(height=1024)
                output_components = [ chatbot, history ]  
                with gr.Row():
                    save_btn = gr.Button("Save Chat")
            
            def wrapper(*args):
                # here to support the change between the stop and submit button
                try:
                    for output in fn(*args):
                        output = [o for o in output]
                        # output for output_components, the rest for [button, button]
                        yield output + [
                            gr.Button.update(visible=False),
                            gr.Button.update(visible=True),
                        ]
                finally:
                    yield [{'__type__': 'generic_update'}, {'__type__': 'generic_update'}] + [ gr.Button.update(visible=True), gr.Button.update(visible=False)]

            def cancel(history, chatbot):
                if history == []:
                    return (None, None)
                return history[:-1], chatbot[:-1]

            extra_output = [submit_btn, stop_btn]
            save_btn.click(
                save, 
                input_components, 
                None, 
            )
            pred = submit_btn.click(
                wrapper, 
                input_components, 
                output_components + extra_output, 
                api_name="predict",
                scroll_to_output=True,
                preprocess=True,
                postprocess=True,
                batch=False,
                max_batch_size=4,
            )
            submit_btn.click(
                lambda: (
                    submit_btn.update(visible=False),
                    stop_btn.update(visible=True),
                ),
                inputs=None,
                outputs=[submit_btn, stop_btn],
                queue=False,
            )
            stop_btn.click(
                lambda: (
                    submit_btn.update(visible=True),
                    stop_btn.update(visible=False),
                ),
                inputs=None,
                outputs=[submit_btn, stop_btn],
                cancels=[pred],
                queue=False,
            )
            cancel_btn.click(
                cancel,
                inputs=[history, chatbot],
                outputs=[history, chatbot]
            )
            reset_btn.click(
                None, 
                [],
                (
                    # input_components ; don't work for history...
                    input_components_except_states
                    + [input_component_column]
                ),  # type: ignore
                _js=f"""() => {json.dumps([
                    getattr(component, "cleared_value", None) for component in input_components_except_states ] 
                    + ([gr.Column.update(visible=True)])
                    + ([])
                )}
                """,
            )
            clear_history.click(clear, None, [history, chatbot], queue=False)

    demo.queue().launch(share=args.share_link,server_port=None)

def _start(
    *, 
    model_path: str=f"{utils.MODEL_DIR}/Llama-2-7b-hf", # 'yahma_llama_7b' 
    lora_path: str=None,
    checkpoint_dir: str=None,
    dtype: str='int8', 
    prompt_type: str='chat',
    use_typewriter: bool=True,
    show_beam: bool=False,
    # for webui
    share_link: bool=False, # for gradio share url
    gradio: bool=False,
    # for commandline
    data_path: str=None,
    out_path: str=None,
):
    global model, tokenizer, args
    import inspect
    frame = inspect.currentframe()
    names, _, _, locals = inspect.getargvalues(frame)
    args = utils.HyperParams().from_inspect(names, locals)
    model, tokenizer = preload(args)
    if gradio:
        chat()
    else:
        assert args.data_path is not None and args.out_path is not None
        chat_cmd()

if __name__ == "__main__":
    import defopt
    defopt.run(_start)