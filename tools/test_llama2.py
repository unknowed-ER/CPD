from transformers import LlamaForCausalLM,LlamaTokenizer
import torch
    
@torch.inference_mode()
def test_model_7b_logits():
    input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
    model = LlamaForCausalLM.from_pretrained("/model/llama-2-7b-hf", device_map="auto")
    out = model(torch.tensor([input_ids]))
    EXPECTED_MEAN = torch.tensor([[-6.6550, -4.1227, -4.9859, -3.2406, 0.8262, -3.0033, 1.2964, -3.3699]])
    torch.testing.assert_close(out.logits.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)

    EXPECTED_SLICE = torch.tensor([-12.8281, -7.4453, -0.4639, -8.0625, -7.2500, -8.0000, -6.4883, -7.7695, -7.8438, -7.0312, -6.2188, -7.1328, -1.8496, 1.9961, -8.6250, -6.7227, -12.8281, -6.9492, -7.0742, -7.7852, -7.5820, -7.9062, -6.9375, -7.9805, -8.3438, -8.1562, -8.0469, -7.6250, -7.7422, -7.3398,])
    torch.testing.assert_close(out.logits[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)
    # 

@torch.inference_mode()
def test_model_13b_logits():
    input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
    model = LlamaForCausalLM.from_pretrained("/model/llama-2-13b-hf", device_map="auto")
    out = model(torch.tensor(input_ids))
    # Expected mean on dim = -1
    EXPECTED_MEAN = torch.tensor([[-2.0622, -1.2794, -1.1638, -0.9788, -1.4603, -1.0238, -1.7893, -1.4411]])
    torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
    # slicing logits[0, 0, 0:30]
    # fmt: off
    EXPECTED_SLICE = torch.tensor([-8.1406, -8.0547, 2.7461, -1.2344, -0.1448, -1.8262, -1.0020, -1.8154, -1.6895, -1.8516, -2.3574, -0.9277, 3.7598, 6.5742, -1.2998, -0.1177, -8.1406, -2.9688, -2.9199, -3.1699, -3.5254, -2.3555, -2.7988, -3.4141, -2.8262, -4.5195, -3.3379, -3.3164, -2.7832, -3.0273])
    # fmt: on
    torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

@torch.inference_mode()
def test_model_13bf_logits():
    input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
    model = LlamaForCausalLM.from_pretrained("/model/llama-2-13b-hf", device_map="auto")
    out = model(torch.tensor(input_ids))
    # Expected mean on dim = -1
    EXPECTED_MEAN = torch.tensor([[-0.8562, -1.8520, -0.7551, -0.4162, -1.5161, -1.2038, -2.4823, -2.3254]])
    torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
    # slicing logits[0, 0, 0:30]
    # fmt: off
    EXPECTED_SLICE = torch.tensor([-2.2227, 4.8828, 0.9023, -0.4578, -0.7871, -0.1033, -0.6221, -0.5786, -0.7803, -1.0674, -1.2920, -0.1570, 0.8008, 2.0723, -0.9497, 0.2771, -2.2227, -0.7612, -1.4346, -1.2061, -1.6426, -0.3000, -0.7139, -1.1934, -1.8691, -1.6973, -1.5947, -1.2705, -0.3523, -0.5513])
    # fmt: on
    torch.testing.assert_close(out.mean(-1), EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

@torch.inference_mode()
def test_model_70b_logits():
    input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", device_map="auto")
    out = model(torch.tensor(input_ids))

    EXPECTED_MEAN = torch.tensor(
        [[-4.2327, -3.3360, -4.6665, -4.7631, -1.8180, -3.4170, -1.4211, -3.1810]], dtype=torch.float32
    )
    torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
    # fmt: off
    EXPECTED_SLICE = torch.tensor([-9.4922, -3.9551, 1.7998, -5.6758, -5.1055, -5.8984, -4.8320, -6.8086, -6.5391, -5.6172, -5.5820, -5.5352, 1.7881, 3.6289, -6.5117, -3.4785, -9.5000, -6.0352, -6.8125, -6.0195, -6.6836, -5.4727, -6.2812, -6.0391, -7.3398, -7.4297, -7.4844, -6.5820, -5.8789, -5.5312])
    # fmt: on
    torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

@torch.inference_mode()
def test_model_greedy_generation():
    EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same everywhere in the universe and 2) the passage of time and the length of objects can vary depending on the observer\'s frame of reference.\n\nThe first part of the theory, that the laws of physics are the same everywhere, is known as the "princi"""
    prompt = "Simply put, the theory of relativity states that "
    tokenizer = LlamaTokenizer.from_pretrained("/model/llama-2-7b-hf")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    model = LlamaForCausalLM.from_pretrained(
        "/model/llama-2-7b-hf", 
        device_map='auto', 
        use_safetensors=False,
        torch_dtype=torch.float16
    )

    # greedy generation outputs
    generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(text)

# test_model_greedy_generation()
test_model_7b_logits()