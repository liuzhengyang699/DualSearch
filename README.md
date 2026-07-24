# DualSearch

<p align="center">
  <img src="assets/logo.png" width="220" alt="DualSearch Logo">
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

## English

DualSearch is a retrieval-augmented framework for encyclopedic visual question
answering on EVQA. It trains multimodal models to retrieve visually related
entities, acquire fine-grained knowledge through text retrieval, and answer
using the combined evidence.

Tool calls follow Qwen's native `<tool_call>` format with JSON arguments.
Visual retrieval uses Qwen3-VL-Embedding, while text retrieval combines BGE-M3,
BM25, and a reranker. Training is built on veRL and includes supervised
fine-tuning and reinforcement learning .

### Training Data

The RL training set is built from the `automatic` and `2_hop` subsets of EVQA.
It contains 10,000 `automatic` examples and 1,000 `2_hop` examples, covering
Plants, Insects, and Birds. The proportions of examples with 1/2/3/4/5 images
are 30%/25%/25%/10%/10%, and the visual retrieval corpus retains at least
eight candidate images for every species.

Use the following commands to build the RL data, SFT data, and retrieval
indexes:

```bash
conda run -n DualS python data/build_rl.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_sft.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_index.py \
  --config "$(pwd)/data/config.local.json"
```

### Environment and Configuration

Install the dependencies required for training and retrieval:

```bash
conda activate DualS
python -m pip install -r requirements.txt
conda install -y -c pytorch -c nvidia \
  "faiss-gpu=1.8.0=py3.10_h4c7d538_0_cuda12.1.1"
```

### Retrieval Services

Build the indexes first, then keep both retrieval services running during RL
training:

```bash
CUDA_VISIBLE_DEVICES=0 python dual_search/search/retrieval_server.py \
  --index_path /absolute/output/indexes/text/bge_m3_Flat.index \
  --corpus_path /absolute/output/text_corpus.jsonl \
  --retriever_model /absolute/local/path/bge-m3 \
  --reranker_model /absolute/local/path/bge-reranker-v2-m3 \
  --device cuda:0 \
  --faiss_gpu \
  --port 8000

CUDA_VISIBLE_DEVICES=1 python dual_search/search/vision_retrieval_server.py \
  --index_path /absolute/output/indexes/vision/qwen3_vl_embedding_Flat.index \
  --corpus_path /absolute/output/vision_corpus.jsonl \
  --retriever_model /absolute/local/path/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --faiss_gpu \
  --port 8001
```

### RL Training

Both `TRAIN_FILE` and  `TEST_FILE` are required.

```bash
export TRAIN_FILE=/absolute/external/path/dual_search_11k/train.parquet
export TEST_FILE=/absolute/external/path/evaluation/test.parquet
export BASE_MODEL=/absolute/local/path/sft-checkpoint/huggingface
export GENRM_MODEL=/absolute/local/path/GenRM
export TEXT_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export VISION_RETRIEVER_URL=http://127.0.0.1:8001/vision_retrieve
bash train_grpo.sh
```

### Acknowledgements

DualSearch builds on [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
and uses
[Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B),
[BGE-M3](https://huggingface.co/BAAI/bge-m3),
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and
[veRL](https://github.com/volcengine/verl).

## 中文

DualSearch 面向 EVQA 中的检索增强多模态问答，训练模型先通过视觉检索定位相关实体，
再借助文本检索补充细粒度知识，最终基于两类检索证据生成答案。

工具调用遵循 Qwen 原生的 `<tool_call>` 格式，参数使用 JSON 表示。视觉检索采用
Qwen3-VL-Embedding，文本检索结合 BGE-M3、BM25 与重排序模型。训练基于 veRL，
包括监督微调和强化学习。

### 训练数据

RL 训练集由 EVQA 的 `automatic` 和 `2_hop` 子集构造，共包含 10,000 条
`automatic` 样本和 1,000 条 `2_hop` 样本，覆盖植物、昆虫和鸟类。单题包含
1/2/3/4/5 张图的样本占比分别为 30%/25%/25%/10%/10%，视觉检索库为每个物种
保留至少 8 张候选图。

分别使用以下命令构造 RL 数据、SFT 数据和检索索引：

```bash
conda run -n DualS python data/build_rl.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_sft.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_index.py \
  --config "$(pwd)/data/config.local.json"
```

### 环境与配置

安装训练与检索所需依赖：

```bash
conda activate DualS
python -m pip install -r requirements.txt
conda install -y -c pytorch -c nvidia \
  "faiss-gpu=1.8.0=py3.10_h4c7d538_0_cuda12.1.1"
```

### 检索服务

先完成索引构造，再在 RL 训练期间保持两个检索服务运行：

```bash
CUDA_VISIBLE_DEVICES=0 python dual_search/search/retrieval_server.py \
  --index_path /absolute/output/indexes/text/bge_m3_Flat.index \
  --corpus_path /absolute/output/text_corpus.jsonl \
  --retriever_model /absolute/local/path/bge-m3 \
  --reranker_model /absolute/local/path/bge-reranker-v2-m3 \
  --device cuda:0 \
  --faiss_gpu \
  --port 8000

CUDA_VISIBLE_DEVICES=1 python dual_search/search/vision_retrieval_server.py \
  --index_path /absolute/output/indexes/vision/qwen3_vl_embedding_Flat.index \
  --corpus_path /absolute/output/vision_corpus.jsonl \
  --retriever_model /absolute/local/path/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --faiss_gpu \
  --port 8001
```

### RL 训练

训练时提供处理好的 `TRAIN_FILE` 和 `TEST_FILE`。

```bash
export TRAIN_FILE=/absolute/external/path/dual_search_11k/train.parquet
export TEST_FILE=/absolute/external/path/evaluation/test.parquet
export BASE_MODEL=/absolute/local/path/sft-checkpoint/huggingface
export GENRM_MODEL=/absolute/local/path/GenRM
export TEXT_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export VISION_RETRIEVER_URL=http://127.0.0.1:8001/vision_retrieve
bash train_grpo.sh
```

### 致谢

DualSearch 在 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
的基础上开发，并使用
[Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)、
[BGE-M3](https://huggingface.co/BAAI/bge-m3)、
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 和
[veRL](https://github.com/volcengine/verl)。
