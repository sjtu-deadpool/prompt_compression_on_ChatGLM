# -*- coding: utf-8 -*-
# @Time    : 2024/12/16 21:18
import sys
import os
import json
import torch
from sacrebleu.metrics import BLEU
from rouge import Rouge
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper, global_args
from deep_training.zoo.model_zoo.chatglm.llm_model import MyTransformer, ChatGLMTokenizer, setup_model_profile, ChatGLMConfig, PetlArguments


def evaluate(data):
    bleu_scorer = BLEU()
    rouge_scorer = Rouge()

    bleu_scores = []
    rouge_scores = []

    for d in data:
        bleu_score = bleu_scorer.sentence_score(
            hypothesis=d['text'],
            references=d['ref'],
        ).score
        bleu_scores.append(bleu_score)

        rouge_score = rouge_scorer.get_scores(
            hyps=[d['text']],
            refs=d['ref'],
        )[0]['rouge-l']['f']
        rouge_scores.append(rouge_score)

    return {
        "bleu_score": sum(bleu_scores) / len(bleu_scores),
        "rouge-l_score": sum(rouge_scores) / len(rouge_scores),
    }


def infer_and_evaluate_lora(data_path):
    # Load LoRA fine-tuned model
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)
    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLMTokenizer
    config: ChatGLMConfig
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    ckpt_dir = '../scripts/best_ckpt/last'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)
    config.initializer_weight = False
    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode is True and config.pre_seq_len is None

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(
        config=config, model_args=model_args, lora_args=lora_args,
        torch_dtype=torch.float16, new_num_tokens=new_num_tokens
    )

    # Load LoRA weights
    pl_model.load_sft_weight(ckpt_dir)
    model = pl_model.get_llm_model()
    model.half().cuda()
    model = model.eval()

    # Collect JSON files in the data directory
    json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]

    all_results = []

    for json_file in json_files:
        file_path = os.path.join(data_path, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Perform inference
        infer_results = []
        for entry in dataset:
            input_text = entry['instruction'] if entry['input'] == "" else f"{entry['instruction']} {entry['input']}"
            response, _ = model.chat(
                tokenizer, input_text, history=[], max_length=2048,
                eos_token_id=config.eos_token_id, do_sample=True,
                top_p=0.7, temperature=0.95
            )
            infer_results.append({"text": response, "ref": [entry['output']]})

        # Perform evaluation
        evaluation_result = evaluate(infer_results)

        # Save results
        all_results.append({
            "file": json_file,
            "evaluation": evaluation_result
        })

        print(f"File: {json_file}, BLEU Score: {evaluation_result['bleu_score']}, ROUGE-L Score: {evaluation_result['rouge-l_score']}")

    return all_results


if __name__ == '__main__':
    data_dir = "./prompt_compression_on_ChatGLM/chatglm_finetuning/data"
    infer_and_evaluate_lora(data_dir)
