# -*- coding: utf-8 -*-
# @Time    : 2024/12/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import os
import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper,global_args
from deep_training.zoo.model_zoo.chatglm.llm_model import MyTransformer,ChatGLMTokenizer,setup_model_profile, ChatGLMConfig,PetlArguments


if __name__ == '__main__':
    config_args['seed'] = None
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    ckpt_dir = '../scripts/best_ckpt/last'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)
    config.initializer_weight = False
    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True and config.pre_seq_len is None

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,new_num_tokens=new_num_tokens,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} if torch.cuda.device_count() > 1 else None,
                             )
    # loading lora weight
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False
    if enable_merge_weight:
        # merege lora weight to llm weight
        pl_model.save_sft_weight(os.path.join(ckpt_dir,'pytorch_model_merge.bin'),merge_lora_weight=True)

    else:
        model = pl_model.get_llm_model()

        text_list = [
            "写一个诗歌，关于冬天",
            "晚上睡不着应该怎么办",
        ]
        for input in text_list:
            response, history = model.chat(tokenizer, input, history=[], max_length=2048,
                                           eos_token_id=config.eos_token_id,
                                           do_sample=True, top_p=0.7, temperature=0.95, )
            print("input", input)
            print("response", response)



