# DualSearch

<p align="center">
  <img src="assets/logo.png" width="220" alt="DualSearch Logo">
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

## English

DualSearch trains multimodal models to use visual and text retrieval tools on
EVQA. Tool calls use Qwen's native `<tool_call>` boundary with JSON arguments.
The retrieval stack uses Qwen3-VL-Embedding for images, BGE-M3 plus BM25 and a
reranker for text, and veRL for SFT and reinforcement learning.

### Fixed 11K recipe

The public data builder has one fixed RL recipe:

- 10,000 `automatic` and 1,000 `2_hop` questions.
- Plants, Insects, and Birds only.
- The 1/2/3/4/5-image distribution is exactly 30%/25%/25%/10%/10%.
- A 25% reserve pool is used internally to replace invalid samples.
- Every final row is retrieval-resolvable and has at least eight visual
  candidates.

These values are intentionally not configurable. The builder writes
`train.parquet`, `vision_corpus.jsonl`, `text_corpus.jsonl`, reports, and
manifests. It does **not** write `test.parquet`; RL validation must use a
separately prepared, leak-free evaluation Parquet supplied through
`TEST_FILE`.

All EVQA, iNaturalist, Wikipedia, teacher, embedding, reranker, base-model, and
GenRM paths must refer to resources already available locally. The builders do
not download datasets or model weights. `output_dir` must be an absolute
directory outside the repository checkout.

### Environment and configuration

The CPU-only RL-data build needs the lightweight dependencies:

```bash
conda create -n DualS python=3.10 pip -y
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  conda run -n DualS python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-data-cpu.txt
```

SFT, indexing, retrieval services, and RL training require the full GPU
environment:

```bash
conda activate DualS
python -m pip install -r requirements.txt
conda install -y -c pytorch -c nvidia \
  "faiss-gpu=1.8.0=py3.10_h4c7d538_0_cuda12.1.1"
```

Copy the tracked example and edit every source/model path. Keep the local copy
untracked:

```bash
cp data/config.example.json data/config.local.json
```

The supported build interface consists of exactly three entry points:

```bash
conda run -n DualS python data/build_rl.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_sft.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_index.py \
  --config "$(pwd)/data/config.local.json"
```

Each command reuses matching fingerprints from
`<output_dir>/.build_cache`; add `--force` to rebuild its stage.
`build_rl.py` is the CPU stage and stops at the 11K RL Parquet and corpora.
`build_sft.py` requires the configured OpenAI-compatible multimodal teacher to
already be running. `build_index.py` requires the configured local
Qwen3-VL-Embedding and BGE-M3 checkpoints.

To train the cold-start checkpoint after `build_sft.py`:

```bash
MODEL_PATH=/absolute/local/path/Qwen3-VL-4B-Instruct \
SFT_DATA_DIR=/absolute/external/path/dual_search_11k \
bash train_sft.sh
```

### Retrieval services

Build the indexes first, then keep both services running during RL:

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

### RL training

Both `TRAIN_FILE` and the externally produced `TEST_FILE` are required. For
GRPO, `BASE_MODEL` should normally point to an exported SFT checkpoint.

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

DualSearch is developed from
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) and uses
[Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B),
[BGE-M3](https://huggingface.co/BAAI/bge-m3),
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and
[veRL](https://github.com/volcengine/verl).

## 中文

DualSearch 用于训练多模态模型在 EVQA 上调用视觉检索与文本检索工具。工具调用统一采用
Qwen 原生 `<tool_call>` 边界和 JSON 参数。检索链路使用 Qwen3-VL-Embedding、
BGE-M3、BM25 和 reranker，SFT 与强化学习基于 veRL。

### 固定 11K 配方

公开的数据构造入口只实现一套固定 RL 配方：

- 10,000 条 `automatic` 和 1,000 条 `2_hop`。
- 只包含植物、昆虫和鸟类。
- 1/2/3/4/5 图比例严格为 30%/25%/25%/10%/10%。
- 内部使用 25% 候补池替换无效样本。
- 最终所有样本均可检索，并且每个物种至少有 8 张视觉候选图。

以上参数不对外开放配置。构造器会生成 `train.parquet`、
`vision_corpus.jsonl`、`text_corpus.jsonl`、报告和 manifest，但**不会**
生成 `test.parquet`。RL 验证必须通过 `TEST_FILE` 指向另一条流水线生成的、
无泄漏的评测 Parquet。

EVQA、iNaturalist、Wikipedia、教师模型、embedding 模型、reranker、基础模型
和 GenRM 都必须事先存放在本地。构造脚本不会自动下载数据集或模型权重。
`output_dir` 必须是仓库目录之外的绝对路径。

### 环境与配置

CPU-only 的 RL 数据构造只需轻量依赖：

```bash
conda create -n DualS python=3.10 pip -y
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  conda run -n DualS python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-data-cpu.txt
```

SFT、索引、检索服务和 RL 训练需要完整 GPU 环境：

```bash
conda activate DualS
python -m pip install -r requirements.txt
conda install -y -c pytorch -c nvidia \
  "faiss-gpu=1.8.0=py3.10_h4c7d538_0_cuda12.1.1"
```

复制示例配置，并将其中所有数据与模型路径改为本地绝对路径。本地配置不会被 Git
跟踪：

```bash
cp data/config.example.json data/config.local.json
```

项目只提供以下三个数据构造入口：

```bash
conda run -n DualS python data/build_rl.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_sft.py \
  --config "$(pwd)/data/config.local.json"

conda run -n DualS python data/build_index.py \
  --config "$(pwd)/data/config.local.json"
```

三个命令都会复用 `<output_dir>/.build_cache` 中指纹一致的结果；需要强制重建时
追加 `--force`。
`build_rl.py` 是 CPU 阶段，止于 11K RL Parquet 和图文 corpus。
`build_sft.py` 要求配置中的 OpenAI 兼容多模态教师服务已经启动。
`build_index.py` 要求配置中的 Qwen3-VL-Embedding 与 BGE-M3 权重已在本地。

运行 `build_sft.py` 后，可训练冷启动 checkpoint：

```bash
MODEL_PATH=/absolute/local/path/Qwen3-VL-4B-Instruct \
SFT_DATA_DIR=/absolute/external/path/dual_search_11k \
bash train_sft.sh
```

### 检索服务

先完成索引构造，再在 RL 训练期间保持两个服务运行：

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

训练时必须同时提供 `TRAIN_FILE` 和外部生成的 `TEST_FILE`。运行 GRPO 时，
`BASE_MODEL` 通常应指向 SFT 导出的 checkpoint。

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

DualSearch 基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 改进，并使用
[Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)、
[BGE-M3](https://huggingface.co/BAAI/bge-m3)、
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 和
[veRL](https://github.com/volcengine/verl)。
