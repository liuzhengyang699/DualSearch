# DualSearch

<p align="center">
  <img src="assets/logo.png" width="220" alt="DualSearch Logo">
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

## English

DualSearch is a project for training multimodal models to use visual and text
retrieval tools on EVQA. Tool calls use Qwen's native `<tool_call>` boundary
and JSON arguments.

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
  --corpus_path /path/to/output/text_corpus.jsonl \
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
  --corpus_path /path/to/output/vision_corpus.jsonl \
  --retriever_model /path/to/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --faiss_gpu \
  --port 8001
```

### Train

Keep both retrieval services running before training, and set `GENRM_MODEL` to
the actual GenRM path.

```bash
export TEXT_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export VISION_RETRIEVER_URL=http://127.0.0.1:8001/vision_retrieve
export GENRM_MODEL=/actual/path/to/small/GenRM
bash train_grpo.sh
```

### Acknowledgements

DualSearch is developed based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1), and uses related works and models including [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and [veRL](https://github.com/volcengine/verl).

## 中文

DualSearch 是一个面向 EVQA 的训练多模态模型使用图文检索工具的项目。工具格式统一使用 Qwen 原生
`<tool_call>` 边界和 JSON 参数。

当前链路使用 Qwen3-VL-Embedding 做视觉检索，使用 BGE-M3 混合检索和 reranker 做文本检索，并基于 veRL 进行强化学习训练。


### 安装

```bash
conda activate DualS
python -m pip install -r requirements.txt
conda install -y -c pytorch -c nvidia "faiss-gpu=1.8.0=py3.10_h4c7d538_0_cuda12.1.1"
```


### 启动检索服务

文本检索：

```bash
CUDA_VISIBLE_DEVICES=0 python dual_search/search/retrieval_server.py \
  --index_path /path/to/text/index/bge_m3_Flat.index \
  --corpus_path /path/to/output/text_corpus.jsonl \
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
  --corpus_path /path/to/output/vision_corpus.jsonl \
  --retriever_model /path/to/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --faiss_gpu \
  --port 8001
```

### 训练

训练前需要保持两个检索服务运行。并将 `GENRM_MODEL` 指定为 GenRM 的真实路径。

```bash
export TEXT_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export VISION_RETRIEVER_URL=http://127.0.0.1:8001/vision_retrieve
export GENRM_MODEL=/actual/path/to/small/GenRM
bash train_grpo.sh
```

### 致谢

DualSearch 基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 进行改进，使用了 [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)、[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)、[Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 和 [veRL](https://github.com/volcengine/verl) 等相关工作与模型。
