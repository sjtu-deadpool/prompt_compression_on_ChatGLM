# Prompt_compression_on_ChatGLM
# Course Project for 2024FALL CSCI-GA 3033-091 IDLS

# ChatGLM Compression Project Guideline

## Team Members

- **Chenyi Fu** (cf3076)
- **Yipeng Zhang** (yz10077)

---

## Local Test Environment

- **Python**: 3.8-3.10
- **CUDA**: Version 12.7
- **NVCC**: Version 12.4
- **Hardware**: NVIDIA RTX 4080 Super, 16GB VRAM
- **System**: WSL2 Ubuntu 24.04

---

## File Structure

To ensure the project runs correctly, navigate to the folder `/prompt_compression_on_ChatGLM/chatglm_finetuning`. Use `display.py` to display the structure of the folder. The project requires the following files and folders:

```
|- display.py
|- data_utils.py
|- args.md
|- train.py
|- training
  |- train_ac.py
  |- train_cl.py
  |- train_pl.py
  |- train_hf.py
  |- __init__.py
|- infer
  |- evaluate.py
  |- infer_finetuning.py
  |- infer.py
  |- infer_lora_finetuning.py
  |- api_lora_demo.py
  |- __init__.py
|- data_processer.py
|- scripts
  |- best_ckpt_lora
    |- last
      |- adapter_model.bin
      |- adapter_config.json
      |- config.json
    |- epoch=8-step=900
      |- adapter_model.bin
      |- adapter_config.json
      |- config.json
    ...
  |- best_ckpt
    |- epoch=0-step=100.ckpt
    |- epoch=9-step=1000.ckpt
    |- last.ckpt
    |- config.json
  |- best_ckpt_ptv2
    |- last.ckpt
    |- config.json
  |- best_ckpt_adalora
    |- last
      |- adapter_model.bin
      |- adapter_config.json
      |- config.json
    ...
  |- train_lora.sh
  |- train_ptv2.sh
  |- train_full.sh
|- README.md
|- config
  |- train_hf.yaml
  |- colossalai_strategy.yaml
  |- deepspeed_offload.yaml
  |- train_cl.yaml
  |- deepspeed.yaml
  |- global.yaml
  |- train_ac.yaml
  |- petl.yaml
  |- train_pl.yaml
  |- __init__.py
  |- main.py
|- modeldownload.ipynb
|- requirements.txt
|- check.ipynb
|- data
  |- alpaca2qa.py
  |- *.json
|- compression_algorithm
  |- bart_en.py
  |- convert_json.py
  |- convert_json2.py
  |- llmlingua_en.py
  |- mycompression_en.py
```

Note: Only the **last checkpoint files** are required to run `infer_finetuning.py`. Pre-trained and fine-tuned checkpoints are available at the following [Google Drive link](https://drive.google.com/drive/folders/149ojsqKzRi9737y_l8_ETXqGxo5nBuFj?usp=drive_link).

---

## Installation

```bash
pip install -U -r requirements.txt
```

---

## Model Weights

Download the following pre-trained ChatGLM weights:

- [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)
- [ChatGLM-6B-INT8](https://huggingface.co/THUDM/chatglm-6b-int8)
- [ChatGLM-6B-INT4](https://huggingface.co/THUDM/chatglm-6b-int4)

---

## Dataset Preparation

All dataset files are available at the following [Google Drive link](https://drive.google.com/drive/folders/1GPaQ_HdZkbCJF4H3GtxvenKTzGn-p03h?usp=drive_link).

### Alpaca-Style Dataset Examples

```json
{
  "instruction": "Describe the structure of an atom.",
  "input": "",
  "output": "An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom."
}
{
  "instruction": "How can we reduce air pollution?",
  "input": "",
  "output": "There are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy-efficient appliances."
}
{
  "instruction": "Describe a time when you had to make a difficult decision.",
  "input": "",
  "output": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."
}
```

### MyDataset Examples

These examples are derived from **IFLYTEK** and **MiningZhiDaoQA** datasets. The data was curated by selecting texts with rhetorical, metaphorical, idiomatic expressions, and crafting relevant questions.

```json
{
  "instruction": "请详细制定一个游览上海的三天计划，包括吃喝玩乐内容以及每天的具体时间安排和活动内容。确保每个行程马不停蹄但不失轻松，同时包含一些人文历史景点和现代地标。请尽量避免重复行程，同时提供替代选择。",
  "input": "关于游览上海的计划",
  "output": "目标：深入体验上海的历史文化与现代都市风貌，兼顾放松与游玩。\n\n步骤与时间安排：\n\n1. **第一天：城市地标与历史文化**\n   - **上午**：\n     - 参观外滩，欣赏浦江两岸的标志性建筑与景色。\n     - 漫步南京路步行街，体验上海最繁华的商业街区。\n   - **中午**：\n     - 前往黄浦江边的本帮菜馆享用上海经典美食，如生煎包、小笼包和本帮红烧肉。\n   - **下午**：\n     - 游览城隍庙和豫园，体验老上海传统建筑和文化，品尝南翔小笼。\n   - **晚上**：\n     - 登上海中心大厦观光台，俯瞰夜晚的魔都风景，或乘坐黄浦江游船欣赏外滩灯光秀。\n\n2. **第二天：现代都市风貌与休闲放松**\n   - **上午**：\n     - 游览陆家嘴金融区，参观东方明珠塔，体验全透明玻璃栈道。\n     - 选择参观上海科技馆，适合亲子或科技爱好者。\n   - **中午**：\n     - 前往正大广场或滨江步道上的美食广场，品尝多国风味美食。\n   - **下午**：\n     - 参观静安寺，体验现代商业与古寺融合的独特氛围。\n     - 或前往思南路和新天地，感受上海时尚休闲区域的文艺气息。\n   - **晚上**：\n     - 前往新天地酒吧街放松，或者在田子坊购买纪念品和欣赏创意艺术。\n\n3. **第三天：人文艺术与自然风景**\n   - **上午**：\n     - 参观上海博物馆，了解上海及中国的历史文物和艺术。\n   - **中午**：\n     - 前往人民广场附近的餐厅品尝地道上海菜，如蟹粉豆腐和油爆虾。\n   - **下午**：\n     - 前往徐家汇公园或世纪公园，享受自然风光，放松身心。\n     - 可选择参观上海图书馆，体验静谧的书香氛围。\n   - **晚上**：\n     - 前往外滩附近的高级餐厅，边用餐边欣赏夜景，为旅行画上完美的句号。\n\n执行保障：提前预订景点门票和餐厅，确保无需排队；合理安排交通工具，地铁为主要出行方式；预留备用时间，避免时间过于紧凑。"
}
{
  "instruction": "根据下面的故事回答问题。\n故事：小明非常喜欢植物，有一天他在花园里种下了一颗向日葵的种子。每天早晨，他都浇水并观察它的生长情况。几周后，向日葵终于开花了，花朵迎着阳光笑容灿烂，金灿灿的非常漂亮。小明感到非常开心，因为他亲手种下的花终于绽放了。\n问题：小明为什么开心？",
  "input": "",
  "output": "小明开心是因为他亲手种下的向日葵终于开花了。"
}
{
  "instruction": "请撰写一个针对学生的详细学习计划，目标是帮助他们在三个月飞速提升英语写作能力。计划需包括具体的学习方法、资源推荐、每日时间分配和每周总结复盘的步骤。请提供可执行且可量化的方案，并适当安排休息时间。",
  "input": "提高英语写作能力的学习计划",
  "output": "目标：三个月内通过系统训练提高英语写作能力，包括词汇运用、句式结构和段落逻辑。\n\n学习方法与步骤：\n\n1. **第一阶段（第1-4周）：基础夯实**\n   - **词汇积累**：\n     - 每天背诵20个写作高频词汇，使用Anki或Quizlet工具辅助记忆。\n     - 每周总结和复习本周所学词汇，进行小测试。\n   - **句式训练**：\n     - 每天练习5个常见句式，如倒装句、强调句和定语从句。\n     - 使用句型造句，每天写10句，确保熟练运用。\n   - **阅读模仿**：\n     - 每天阅读一篇英文短文，分析结构并摘抄好句子。\n     - 每周末进行总结，模仿写作一篇150字左右的文章。\n\n2. **第二阶段（第5-8周）：段落写作**\n   - **段落训练**：\n     - 每天练习写一个主题段落，确保中心句明确、逻辑清晰。\n     - 使用PEEL结构（Point-Example-Explanation-Link）完成段落构建。\n   - **反馈优化**：\n     - 每周请老师或同学点评自己的写作，找出问题并改进。\n     - 针对反馈，每天修改并重写之前的段落。\n   - **资源推荐**：\n     - 使用Grammarly辅助检查语法错误。\n     - 阅读《剑桥雅思写作高分范文》作为参考。\n\n3. **第三阶段（第9-12周）：完整文章写作**\n   - **主题写作**：\n     - 每周练习写一篇完整的文章，字数控制在300-400字。\n     - 涵盖描述、议论、比较等不同文体，确保全面提升。\n   - **每日任务**：\n     - 计划：30分钟词汇复习 + 30分钟文章写作 + 15分钟修改。\n     - 每日完成后总结遇到的问题和进步点。\n   - **周总结与复盘**：\n     - 每周末分析本周写作情况，记录错误和进步。\n     - 针对不足之处进行专项训练，例如句式多样化或逻辑连贯性。\n\n时间安排：\n- 每日学习时长：1.5小时（周末可适当增加练习量）。\n- 每周固定时间复盘：周日晚上1小时。\n\n执行保障：使用专门的笔记本记录每日任务与反馈结果；设立小目标，如“本周掌握5个新句型”、“减少语法错误20%”，保持学习动力。"
}
```

---

## Training

Before starting, place the required JSON dataset files into the `data` folder. Additionally, modify the `output_weight_dir` in the training script to avoid overwriting previous checkpoints. For example:

```python
output_weight_dir = './best_ckpt_lora'
```


```bash
# Prepare the dataset
cd scripts
bash train_full.sh -m dataset

# Full-parameter fine-tuning
bash train_full.sh -m train

# LoRA / AdaLoRA / IA3 fine-tuning
bash train_lora.sh -m train

# P-tuning v2 fine-tuning
bash train_ptv2.sh -m train
```

---

## Inference

### Available Scripts

- `infer.py`: Run inference with the pre-trained model.
- `infer_finetuning.py`: Run inference with fine-tuned checkpoints.
- `infer_lora_finetuning.py`: Run inference with LoRA fine-tuned checkpoints.

Run inference using the following command:

```bash
python infer.py
```

### GPU Memory Requirements

| **Quantization Level** | **Recommended Minimum GPU Memory** |
| ---------------------- | ---------------------------------- |
| FP16 (No Quantization) | 13 GB                              |
| INT8                   | 10 GB                              |
| INT4                   | 6 GB                               |

---

## Local Fine-Tuning Configurations

Default settings for training with PL backend(with this configuration already take about 15.6 Gpu memory):

```yaml
optimizer: lion
scheduler_type: CAWR
scheduler:
  T_mult: 1
  rewarm_epoch_num: 0.5
  verbose: false
optimizer_betas:
  - 0.9
  - 0.999
train_batch_size: 1
eval_batch_size: 1
test_batch_size: 2
learning_rate: 2.0e-04
adam_epsilon: 1.0e-08
gradient_accumulation_steps: 3
max_grad_norm: 1.0
weight_decay: 0
warmup_steps: 0
output_dir: ./outputs_pl
max_seq_length: 512
do_lower_case: null
max_target_length: 100
use_fast_tokenizer: false
dataloader_drop_last: true
dataloader_pin_memory: true
dataloader_num_workers: 0
```

---

## Compression and Evaluation

### Compression

To compress prompts for evaluation, first ensure the JSON files are properly formatted. Then use the `convert_json.py` script located in the `compression_algorithm` folder to apply one of the four available compression algorithms. Once the prompts have been compressed, save the resulting JSON files into the `data` folder.

### Evaluation

After placing the compressed JSON files into the `data` folder, run one of the three provided scripts (`infer.py`, `infer_finetuning.py`, or `infer_lora_finetuning.py`) to perform inference. This will calculate BLEU and ROUGE scores for the outputs:

- `infer.py`: For pre-trained models.
- `infer_finetuning.py`: For fine-tuned models.
- `infer_lora_finetuning.py`: For LoRA fine-tuned models.

These scripts process each JSON file in the `data` directory sequentially, performing inference and then evaluation to generate BLEU and ROUGE-L scores for each file.

---

## References

- [ChatGLM-6B GitHub Repository](https://github.com/THUDM/ChatGLM-6B)
