import torch
from transformers import AutoTokenizer
if __name__ == '__main__':
    #todo!!:trust_remote_code = "true"可用于解决，看似是transformers版本问题ImportError: cannot import name'LlamaForCausalLM'	from	'transformers	(/root/anaconda/envs/dsp2/lib/python3.8	/site-packages/transformers/__init__.py)
    tokenizer = AutoTokenizer.from_pretrained("/public11_data/zs/zs/codellama/THUDM_chatglm2-6b/",trust_remote_code = "true",truncation_side="left",
            padding_side="left")
    text = "Hello, how are you?"
    dic1 = {"Q":"Hello, how are you?","A":"I'm fine!"}
    prompt = "you are jack."
    #todo:encode方法可以将一个文本序列编码为一个token序列。
    # encoded_text = tokenizer.encode(text)
    # print(encoded_text)
    # #todo:将一个token序列解码为一个文本
    # decode_text = tokenizer.decode(encoded_text)
    # print(decode_text)
    # #todo:tokenizer.tokenizer方法，将词语分开
    # token = tokenizer.tokenize(text)
    # print(token)

    src_tokens = tokenizer.tokenize(dic1["Q"])
    src_prompt = tokenizer.tokenize(prompt)
    tgt_tokens = tokenizer.tokenize(dic1["A"])
    tokens = src_prompt + src_tokens + tgt_tokens
    print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(input_ids)
    #print(tokenizer.bos_token_id)
    context_length = input_ids.index(tokenizer.bos_token_id)


# if torch.cuda.is_available():
#     print("CUDA is available on this system.")
#     device = torch.device("cuda")
#     print("CUDA device name:", torch.cuda.get_device_name(device))
#     print("CUDA device count:", torch.cuda.device_count())
#
#     import transformers
#
#     print(transformers.__version__+"\n")
# else:
#     print("CUDA is not available on this system.")
