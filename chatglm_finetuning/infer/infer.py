# -*- coding: utf-8 -*-
# @Time    : 2024/12/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper
from deep_training.zoo.model_zoo.chatglm.llm_model import MyTransformer,ChatGLMTokenizer,setup_model_profile, ChatGLMConfig
from deep_training.zoo.model_zoo.chatglm.llm_model import RotaryNtkScaledArguments,RotaryLinearScaledArguments # aigc-zoo 0.1.21

if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLMTokenizer
    config: ChatGLMConfig
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005
    config.initializer_weight = False


    rope_args = RotaryNtkScaledArguments(model_type='chatglm',max_position_embeddings=config.max_sequence_length,alpha=4) 
    # rope_args = RotaryLinearScaledArguments(model_type='chatglm',name='rotary_pos_emb',max_position_embeddings=2048, scale=4) 
    
    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,rope_args=rope_args)

    model = pl_model.get_llm_model()
    if not model.quantized:
        # modified as needed, currently only supports 4/8 bit quantization, you can save the quantized model
        model.half().quantize(4).cuda()
    else:
        # already quantized model
        model.half().cuda()
    model = model.eval()

    text_list = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "晚上睡不着应该怎么办",
        "写一个诗歌，关于冬天",
    ]
    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[], max_length=2048,
                                       eos_token_id=config.eos_token_id,
                                       do_sample=True, top_p=0.7, temperature=0.95, )
        print("input", input)
        print("response", response)

