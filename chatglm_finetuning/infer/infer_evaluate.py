# -*- coding: utf-8 -*-
# @Time    : 2024/12/16 20:30
import os
import json
import torch
from sacrebleu.metrics import BLEU
from rouge import Rouge
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper
from deep_training.zoo.model_zoo.chatglm.llm_model import MyTransformer, ChatGLMTokenizer, setup_model_profile, ChatGLMConfig
from deep_training.zoo.model_zoo.chatglm.llm_model import RotaryNtkScaledArguments


def infer_and_evaluate(data_path):
    # Setup inference model
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)
    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLMTokenizer
    config: ChatGLMConfig
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005
    config.initializer_weight = False

    rope_args = RotaryNtkScaledArguments(model_type='chatglm', max_position_embeddings=config.max_sequence_length, alpha=4)
    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16, rope_args=rope_args)
    model = pl_model.get_llm_model()

    if not model.quantized:
        model.half().quantize(4).cuda()
    else:
        model.half().cuda()
    model = model.eval()

    # Setup evaluation metrics
    bleu_scorer = BLEU()
    rouge_scorer = Rouge()

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
            response, _ = model.chat(tokenizer, input_text, history=[], max_length=2048,
                                     eos_token_id=config.eos_token_id,
                                     do_sample=True, top_p=0.7, temperature=0.95)
            infer_results.append({"text": response, "ref": [entry['output']]})

        # Perform evaluation
        bleu_scores = []
        rouge_scores = []
        for result in infer_results:
            bleu_score = bleu_scorer.sentence_score(
                hypothesis=result['text'],
                references=result['ref']
            )
            bleu_scores.append(bleu_score.score)

            rouge_score = rouge_scorer.get_scores(
                hyps=[result['text']],
                refs=result['ref']
            )
            rouge_scores.append(rouge_score[0]['rouge-l']['f'])

        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)

        # Save results
        all_results.append({
            "file": json_file,
            "bleu_score": avg_bleu,
            "rouge_l_score": avg_rouge
        })

    # Print results
    for result in all_results:
        print(f"File: {result['file']}, BLEU Score: {result['bleu_score']}, ROUGE-L Score: {result['rouge_l_score']}")


if __name__ == '__main__':
    data_dir = "./prompt_compression_on_ChatGLM/chatglm_finetuning/data"
    infer_and_evaluate(data_dir)
