# -*- coding: utf-8 -*-
# @Time    : 2024/12/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper, get_deepspeed_config
from deep_training.zoo.model_zoo.chatglm.llm_model import MyTransformer,ChatGLMTokenizer,setup_model_profile, ChatGLMConfig,PetlArguments

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    config_args['seed'] = None
    config_args['model_name_or_path'] = None

    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    ###################### warning ######################
    # choose the newest best_ckpt path
    config = ChatGLMConfig.from_pretrained('../scripts/best_ckpt')
    config.initializer_weight = False
    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,)
    if deep_config is None:
        train_weight = '../scripts/best_ckpt/last.ckpt'
    else:
        # generate ./best_ckpt/last/best.pt weight file
        # cd best_ckpt/last
        # python zero_to_fp32.py . best.pt
        train_weight = '../scripts/best_ckpt/last/best.pt'

    # loadind weight after finetuning
    pl_model.load_sft_weight(train_weight,strict=False)

    model = pl_model.get_llm_model()



    if not model.quantized:
        # modified as needed, currently only supports 4/8 bit quantization, you can save the quantized model
        model.half().quantize(8).cuda()
    else:
        # already quantized model
        model.half().cuda()
    model = model.eval()

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    for input in text_list:
        response, history = model.chat(tokenizer, input, history=[],max_length=2048,
                                            eos_token_id=config.eos_token_id,
                                            do_sample=True, top_p=0.7, temperature=0.95,)
        print("input",input)
        print("response", response)

