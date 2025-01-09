Vicuna model:
1. C++
2. GPTQ2bit, 4bit, 6bit, 8bit.
---
## C++
： [Llama.cpp](https://github.com/ggerganov/llama.cpp) 、 [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp), 

   - lora.
   - checkpoint7B13G，13B37G, 30B65B.  ( 13B64G，swap )
   - ， 7B,13B,30B,65Bcheckpoint1,2,4,8 ( cpp )

1.，lora，`ggml`，cpp。
```
bash prepare_llama_cpp.sh
```
 ( ，hflora`consolidated.0x.pth`，`x`num_shards，`ggml-model-f16.bin`。 )
```bash 
python tools/merge_lora_for_cpp.py --lora_path $lora_path
```

1. ，`vicuna.cpp`，CPUC++ !
```bash
cd tools/vicuna.cpp
make chat 
# we also offer a Makefile.ref, which you can call it with `make -f Makefile.ref `
./chat -m $ggml-path

```
[Optional]ggmlint4（`ggml-model-q4_0.bin`）（）。
```bash
make quantize
./quantize.sh
```

---
## Quantize LLaMA
，4GLLaMA-7B(2bit)。[GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)。
transformers4.29.0.dev0。
### 1. huggingface。，:
```bash 
python convert_llama.py --input_dir /model/llama-7b --model_size 7B --output_dir ./llama-hf
```
### 2. ，8bit、4bit、2bit:
- LLaMA-7B8-bit
```bash
CUDA_VISIBLE_DEVICES=0 python quant_llama.py ./llama-hf/llama-7b wikitext2 --wbits 8 --true-sequential --act-order --groupsize 128 --save llama7b-8bit-128g.pt
```

- LLaMA-7B4-bit（）
```bash
CUDA_VISIBLE_DEVICES=0 python quant_llama.py ./llama-hf/llama-7b wikitext2 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt
```

- LLaMA-7B2-bit
```bash
CUDA_VISIBLE_DEVICES=0 python quant_llama.py ./llama-hf/llama-7b wikitext2 --wbits 2 --true-sequential --act-order --groupsize 128 --save llama7b-2bit-128g.pt
```
### 3.  or gradio：
- text
```bash
python quant_generate.py --model_path ./llama-hf/llama-7b --quant_path llama7b-4bit-128g.pt --wbits 4 --groupsize 128 --text "the mean of life is"
```
- gradio，
```bash
python quant_generate.py --model_path ./llama-hf/llama-7b --quant_path llama7b-4bit-128g.pt --wbits 4 --groupsize 128 --gradio
```

#### LLaMA-7B ：
- 8bit[8.5G] [Download](https://huggingface.co/Chinese-Vicuna/llama7b_8bit_128g).
```text
the mean of life is 70 years.
the median age at death in a population, regardless if it's male or female?
```
- 4bit[5.4G] [Download](https://huggingface.co/Chinese-Vicuna/llama7b_4bit_128g).
```text
the mean of life is 70 years.
the median age at death in africa was about what?
```
- 2bit[4G] [Download](https://huggingface.co/Chinese-Vicuna/llama7b_2bit_128g).
```text
the mean of life is a good., and it’s not to be worth in your own homework for an individual who traveling on my back with me our localities that you can do some work at this point as well known by us online gaming sites are more than 10 years old when i was going out there around here we had been written about his time were over all sited down after being spent from most days while reading between two weeks since I would have gone before its age site;...
```
---


TODO:
- [ ] `merge_lora.py`。
- [ ] `n_ctx'。
- [ ] cpu。