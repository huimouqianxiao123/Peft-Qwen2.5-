# Peft-Qwen2.5 电商客服意图识别

本项目使用 `Qwen2.5-7B-Instruct` 做电商客服意图识别，并通过 QLoRA/PEFT 微调提升分类效果。训练、评估和服务封装主要面向 Google Colab 环境，部署 notebook 使用 FastAPI 提供 OpenAI Chat Completions 兼容接口。

## 项目结构

| 文件 | 说明 |
| --- | --- |
| `QLora.ipynb` | QLoRA 微调主流程：读取多轮对话数据、构造 chat template、训练 adapter、评估微调后效果。 |
| `test_UnQLora.ipynb` | 未加载微调 adapter 的基座模型评估。 |
| `fastapi部署模型.ipynb` | Colab 内启动 FastAPI 接口服务，服务内部加载基座模型和微调 adapter，提供 OpenAI 兼容接口，并内置 30 条手写并发测试请求。 |
| `data/intent_train_multiturn_qwen.jsonl` | 人工标注后的多轮意图识别训练数据，格式为 Qwen chat messages。 |
| `tests/` | notebook 结构测试，检查关键训练、评估、部署代码是否存在。 |

## 数据来源与人工标注

原始数据来自京东电商客服对话数据集 `JDDC-Baseline-Seq2Seq`：

```bash
git clone https://github.com/SimonJYang/JDDC-Baseline-Seq2Seq.git
```

在原始京东客服对话数据的基础上，本项目进行了人工意图标注与格式整理，最终构造了 5000 条模拟真实业务场景的多轮电商客服意图识别样本。标注目标不是生成客服回复，而是识别用户当前话语的业务意图。

数据处理后的训练格式为 Qwen chat messages：

```json
{
  "messages": [
    {"role": "system", "content": "你是电商客服意图识别助手..."},
    {"role": "user", "content": "用户当前对话内容..."},
    {"role": "assistant", "content": "查询物流"}
  ]
}
```

标注后的样本覆盖物流、订单、售后、支付、发票、优惠、商品咨询、投诉、转人工、问候确认等常见京东电商客服业务场景，更贴近真实线上客服请求分布。

## 任务标签

模型输出限定为以下 18 个意图标签：

`查询物流`、`催促配送`、`修改订单`、`取消订单`、`申请退款`、`退换货`、`发票咨询`、`支付咨询`、`价格价保`、`优惠活动`、`商品咨询`、`催促处理`、`投诉反馈`、`转人工`、`问候确认`、`业务咨询`、`查询订单`、`其他`

## 训练环境

推荐训练环境：

| 配置 | 值 |
| --- | --- |
| 平台 | Google Colab |
| GPU | NVIDIA A100 |
| 显存 | 40GB |
| 基座模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 微调方式 | QLoRA / PEFT |
| 量化 | 4bit NF4 |
| 精度 | `fp16` |
| 优化器 | `paged_adamw_8bit` |

当前 notebook 中的主要训练配置：

| 参数 | 当前值 |
| --- | --- |
| `model_path` | `/content/drive/MyDrive/models/qwen2.5-7b-instruct` |
| `adapter_output_dir` | `/content/drive/MyDrive/models/qwen2.5-7b-intent-qlora` |
| `max_seq_length` | `1024` |
| `train_sample_size` | `2500` |
| `per_device_train_batch_size` | `10` |
| `gradient_accumulation_steps` | `8` |
| 有效 batch size | `10 * 8 = 80` |
| `learning_rate` | `2e-4` |
| `num_train_epochs` | 当前代码为 `3` |

说明：`QLora.ipynb` 的历史评估输出中打印了“训练轮数: 10”，但当前 `TrainingArguments` 代码配置为 `num_train_epochs=3`。复现实验时建议以实际运行时的 `TrainingArguments` 为准，并同步更新评估打印信息。

## 显存利用率优化建议

A100 40GB 显存适合把 batch size 调大，以减少 GPU 空转。当前配置是：

```python
per_device_train_batch_size=10
gradient_accumulation_steps=8
```

如果训练时显存占用没有接近上限，可以优先调大：

```python
per_device_train_batch_size=12
```

如果仍然稳定，再尝试：

```python
per_device_train_batch_size=16
```

建议调参顺序：

1. 保持 `max_seq_length=1024` 不变，逐步提高 `per_device_train_batch_size`。
2. 如果显存不足，先回退 batch size，而不是关闭 QLoRA 或 4bit。
3. 如果想保持有效 batch size 接近 80，可以在增大单卡 batch 后降低 `gradient_accumulation_steps`，例如 `16 * 5 = 80`。
4. 训练时保留 `gradient_checkpointing=True`，它能降低显存压力，但会增加一些计算开销。

## 微调前后效果对比

评估方式：随机抽取 500 条样本，使用相同意图标签集合做分类评估。

| 模型 | 评估文件 | 有效评估样本数 | Overall Accuracy |
| --- | --- | ---: | ---: |
| 未微调 Qwen2.5-7B-Instruct | `test_UnQLora.ipynb` | 500 | 19.80% |
| QLoRA 微调后模型 | `QLora.ipynb` | 500 | 97.60% |

提升：

| 指标 | 数值 |
| --- | ---: |
| 绝对准确率提升 | `97.60% - 19.80% = 77.80 个百分点` |
| 相对提升倍数 | `97.60 / 19.80 ≈ 4.93 倍` |

未微调模型的主要问题是输出不稳定，容易生成解释性文本或偏向少数标签；微调后模型更倾向于直接输出标签名，在 `查询物流`、`退换货`、`申请退款`、`问候确认`、`其他` 等高频类别上表现明显提升。

微调后评估摘要：

```text
有效评估样本数: 500
总体准确率 (Overall Accuracy): 97.60%
weighted avg precision: 0.97
weighted avg recall: 0.98
weighted avg f1-score: 0.97
```

未微调评估摘要：

```text
有效评估样本数: 500
总体准确率 (Overall Accuracy): 19.80%
weighted avg precision: 0.52
weighted avg recall: 0.20
weighted avg f1-score: 0.21
```

## 训练流程

在 Google Colab 中运行：

1. 打开 `QLora.ipynb`。
2. 挂载 Google Drive。
3. 确认模型路径和数据路径：

```python
model_path = "/content/drive/MyDrive/models/qwen2.5-7b-instruct"
file_path = "/content/drive/MyDrive/data/intent_train_multiturn_qwen.jsonl"
adapter_output_dir = "/content/drive/MyDrive/models/qwen2.5-7b-intent-qlora"
```

4. 运行训练 cell。
5. adapter 会保存到：

```text
/content/drive/MyDrive/models/qwen2.5-7b-intent-qlora
```

## FastAPI 接口服务

打开 `fastapi部署模型.ipynb`，按顺序运行 cell。

说明：FastAPI 不是模型本身，它只是模型服务的接口封装层。实际推理模型是 `Qwen2.5-7B-Instruct` 加载 QLoRA/PEFT 微调 adapter 后得到的意图识别模型。

部署 notebook 默认加载：

```python
MODEL_PATH = "/content/drive/MyDrive/models/qwen2.5-7b-instruct"
ADAPTER_PATH = "/content/drive/MyDrive/models/qwen2.5-7b-intent-qlora"
```

服务接口：

| 接口 | 说明 |
| --- | --- |
| `GET /health` | 查看服务状态、队列长度、batch 配置。 |
| `GET /v1/models` | OpenAI 兼容模型列表。 |
| `POST /v1/chat/completions` | OpenAI 兼容聊天补全接口。 |

支持的推理参数：

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 50,
  "max_tokens": 12,
  "repetition_penalty": 1.0,
  "stop": null,
  "stream": false
}
```

## 推理吞吐优化

`fastapi部署模型.ipynb` 已加入有界队列和动态 batch：

```python
MAX_QUEUE_SIZE = 32
MAX_BATCH_SIZE = 4
BATCH_TIMEOUT_SECONDS = 0.03
REQUEST_TIMEOUT_SECONDS = 180
MAX_INPUT_TOKENS = 2048
```

设计目标：

1. 请求先进入有界队列，队列满时直接返回 `503`，提示用户稍后重试。
2. 后台 worker 将短时间内的请求聚合成 batch，再统一送入 GPU。
3. 对 decoder-only 模型使用左 padding：

```python
tokenizer.padding_side = "left"
```

4. 显式启用 KV cache：

```python
model.config.use_cache = True
generate(..., use_cache=True)
```

由于推理瓶颈主要在 GPU 显存和矩阵计算，而不是 CPU，建议根据 Colab A100 40GB 的实际显存占用调大 `MAX_BATCH_SIZE`。可以从 `4` 开始，逐步尝试 `6`、`8`，如果出现 CUDA OOM 再回退。

## 30 条手写并发测试

`fastapi部署模型.ipynb` 最后一个测试 cell 会：

1. 手写生成 30 条不同意图问题，不从训练数据 JSONL 读取。
2. 使用 `ThreadPoolExecutor(max_workers=30)` 同时发送 30 个请求。
3. 对比 `expected_intent` 和模型输出。
4. 打印每条请求的预测结果、是否正确、耗时和原始输出。
5. 汇总正确数和准确率。

示例输出格式：

```text
01 | 正确 | 预期: 查询物流 | 预测: 查询物流 | 耗时: 1.234s | 问题: ...
...
正确数: 28/30
准确率: 93.33%
```

## 本地结构测试

本仓库包含一些 notebook 结构测试，可在本地运行：

```bash
python -m unittest tests.test_fastapi_deploy_notebook -v
```

该测试不加载大模型，只检查部署 notebook 是否包含关键接口、adapter 路径、动态 batch、KV cache 和 30 条手写测试用例。
