# DualSearch

<p align="center">
  <img src="assets/logo.png" width="220" alt="DualSearch Logo">
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

## English

DualSearch is a multimodal RL tool-use project for EVQA. It trains a vision-language agent to use two retrieval tools during rollout: image retrieval through `<vision_search>` and text retrieval through `<search>`.

The current stack uses Qwen3-VL-Embedding for visual search, BGE-M3 hybrid retrieval with reranking for text search, and veRL for RL training.

### Install

```bash
conda activate DualS
python -m pip install -r requirements.txt
conda install -y -c pytorch -c nvidia "faiss-gpu=1.8.0=py3.10_h4c7d538_0_cuda12.1.1"
```


### Launch Retrieval Services

Text retrieval:

```bash
CUDA_VISIBLE_DEVICES=0 python dual_search/search/retrieval_server.py \
  --index_path /path/to/text/index/bge_m3_Flat.index \
  --corpus_path /path/to/evqa_val_text_corpus.jsonl \
  --retriever_model /path/to/bge-m3 \
  --reranker_model /path/to/bge-reranker-v2-m3 \
  --device cuda:0 \
  --faiss_gpu \
  --port 8000
```

Image retrieval:

```bash
CUDA_VISIBLE_DEVICES=1 python dual_search/search/vision_retrieval_server.py \
  --index_path /path/to/vision/index/qwen3_vl_embedding_Flat.index \
  --corpus_path /path/to/evqa_val_vision_corpus.jsonl \
  --retriever_model /path/to/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --faiss_gpu \
  --port 8001
```

### Train

Keep both retrieval services running before training. Set `GENRM_MODEL` to the local path of a small generative reward model; the placeholder `/path/to/GenRM` cannot start training. veRL launches and manages the GenRM with tensor parallel size 1.

```bash
export TEXT_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export VISION_RETRIEVER_URL=http://127.0.0.1:8001/vision_retrieve
export GENRM_MODEL=/actual/path/to/small/GenRM
bash train_grpo.sh
```

The reward is modeled as:

```text
R = 0.8 * R_answer + 0.2 * R_format - 0.02 * max(0, N - 2)^2
```

`R_answer` and `R_format` are binary. `N` is the total number of successfully executed text and vision retrieval calls. The first two calls are free, the maximum tool-call budget is 8, and negative rewards are retained.

### Acknowledgements

DualSearch is developed based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1), and uses related works and models including [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and [veRL](https://github.com/volcengine/verl).

## 中文

DualSearch 是一个面向 EVQA 的多模态强化学习工具调用项目。训练过程中，视觉语言模型可以通过 `<vision_search>` 调用图像检索工具，也可以通过 `<search>` 调用文本检索工具。

当前链路使用 Qwen3-VL 图像向量做视觉检索，使用 BGE-M3 混合检索和 reranker 做文本检索，并基于 veRL 进行强化学习训练。

### 安装

```bash
conda activate DualS
python -m pip install -r requirements.txt
conda install -y -c pytorch -c nvidia "faiss-gpu=1.8.0=py3.10_h4c7d538_0_cuda12.1.1"
```

请在仓库根目录下运行命令。

### 启动检索服务

文本检索：

```bash
CUDA_VISIBLE_DEVICES=0 python dual_search/search/retrieval_server.py \
  --index_path /path/to/text/index/bge_m3_Flat.index \
  --corpus_path /path/to/evqa_val_text_corpus.jsonl \
  --retriever_model /path/to/bge-m3 \
  --reranker_model /path/to/bge-reranker-v2-m3 \
  --device cuda:0 \
  --faiss_gpu \
  --port 8000
```

图像检索：

```bash
CUDA_VISIBLE_DEVICES=1 python dual_search/search/vision_retrieval_server.py \
  --index_path /path/to/vision/index/qwen3_vl_embedding_Flat.index \
  --corpus_path /path/to/evqa_val_vision_corpus.jsonl \
  --retriever_model /path/to/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --faiss_gpu \
  --port 8001
```

### 训练

训练前需要保持两个检索服务运行。必须将 `GENRM_MODEL` 指定为本地小模型 GenRM 的真实路径；占位路径 `/path/to/GenRM` 无法启动训练。GenRM 由 veRL 启动和管理，使用 TP=1。

```bash
export TEXT_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export VISION_RETRIEVER_URL=http://127.0.0.1:8001/vision_retrieve
export GENRM_MODEL=/actual/path/to/small/GenRM
bash train_grpo.sh
```

当前奖励建模为：

```text
R = 0.8 * R_answer + 0.2 * R_format - 0.02 * max(0, N - 2)^2
```

其中 `R_answer` 和 `R_format` 为二值奖励，`N` 为实际成功执行的文本检索与视觉检索总次数。前两次检索不惩罚，最大工具调用次数为 8，最终奖励允许为负数。

### 致谢

DualSearch 基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 进行改进，使用了 [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)、[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)、[Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 和 [veRL](https://github.com/volcengine/verl) 等相关工作与模型。
