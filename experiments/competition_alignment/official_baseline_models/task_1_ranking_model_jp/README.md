---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:171392
- loss:CosineSimilarityLoss
base_model: sentence-transformers/multi-qa-mpnet-base-dot-v1
widget:
- source_sentence: ダブ 化粧落とし
  sentences:
  - マイクロソフト Surface Pro タイプカバー プラチナ FFP-00159
  - サンタマルシェ ディープクレンジング
  - シャープ[SHARP]　シャープファクシミリ用インクリボン（1本50m）2本入 【UXNR4A4W】
- source_sentence: oxbow チモシー
  sentences:
  - 愛らびっと 2021年産新刈 スーパープレミアムチモシー 1番刈り 500g シングルプレス 牧草 チモシー
  - ロジテック 音楽CD取り込みドライブ WiFi 2.4Ghz対応 11n iOS/Android対応 USB2.0 ホワイト LDR-PS24GWU3RWH
  - PERLESMITH テレビ壁掛け金具 中型 32-55インチ対応 アーム式 耐荷重45kg LCD LED 液晶テレビ用 前後＆左右&上下多角度調節可能
    VESA400x400mm (テレビ壁掛け)
- source_sentence: オールブラックスズボン
  sentences:
  - ラストギアス　（１） (角川コミックス・エース)
  - '[Make 2 Be] ポロシャツ メンズ カジュアル 半袖 襟元 チェック柄 バイカラー チームカラー 卓球 スポーツ ゴルフウェア ゴルフ ワンポイントロゴ
    夏 クールビズ 通気性 速乾 MF19 (31.Green_L)'
  - サンワサプライ 電源タップ 6個口 2P 2m 雷ガード・ホコリ防止シャッター・裏面マグネット付き 配線しやすいスイングプラグ ホワイト TAP-SP2116MG-2W
- source_sentence: 和風どれっしんぐ
  sentences:
  - BETONES/ビトーンズ レディースボクサーパンツ ANIMAL4 ／ グリーン（パンダ）D004L
  - ドーバー パストリーゼ77 5Ｌ詰替スプレーヘッド無し【4本入】
  - キユーピー エルドレッシング フレンチ(赤) 1000ml
- source_sentence: おたま ステンレス
  sentences:
  - 伝える準備
  - ホットアイマスク ギフト グラフェン加熱フイルム 磁力接続タイプ 純シルク製 usb給電 安眠 遮光 圧迫感なし 3段階温度調節 オートオフタイマー 3D立体構造
    睡眠アイマスク 繰り返し使用 睡眠改善 旅行 出張 目の疲れ アイマスク水洗可能 プレゼント ブラック (Black)
  - ナガオ 燕三条 プロフェッショナル お玉 29.5cm 18-8ステンレス 日本製 58002
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_dot
- spearman_dot
model-index:
- name: SentenceTransformer based on sentence-transformers/multi-qa-mpnet-base-dot-v1
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: pearson_dot
      value: 0.29976738061330976
      name: Pearson Dot
    - type: spearman_dot
      value: 0.3155903325609965
      name: Spearman Dot
---

# SentenceTransformer based on sentence-transformers/multi-qa-mpnet-base-dot-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) <!-- at revision 17997f24dca0df1a4fed68894fb0e1e133e60482 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Dot Product
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'おたま ステンレス',
    'ナガオ 燕三条 プロフェッショナル お玉 29.5cm 18-8ステンレス 日本製 58002',
    '伝える準備',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[40.3164, 20.7959, 21.6029],
#         [20.7959, 41.5458, 21.8360],
#         [21.6029, 21.8360, 35.9944]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric           | Value      |
|:-----------------|:-----------|
| pearson_dot      | 0.2998     |
| **spearman_dot** | **0.3156** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 171,392 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                         | label                                                          |
  |:--------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                           | string                                                                             | float                                                          |
  | details | <ul><li>min: 3 tokens</li><li>mean: 10.2 tokens</li><li>max: 47 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 48.34 tokens</li><li>max: 222 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.49</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0              | sentence_1                                                                                                                                                                            | label            |
  |:------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>楽譜の読み方</code>     | <code>やさしくたのしく 楽譜の読み方</code>                                                                                                                                                          | <code>1.0</code> |
  | <code>磁石なし手帳型ケース</code> | <code>iPhone13Pro ケース 6.1インチ 対応 FYY 軽量 薄型 手帳型ケース ハンドメイド 高級PUレザー カード収納 スタンド機能 サイドマグネット ストラップ付き ワイヤレス充電対応 耐衝撃 スマホケース 2021新型 アイフォン13プロケース アイフォン13Pro ケース iPhone13プロケース カバー (ゴールド)</code> | <code>1.0</code> |
  | <code>ブラ 肩紐なし</code>    | <code>[ウイング/ワコール] ノンワイヤーブラジャー 脇高パッドでシルエットをキープ 【フィットトップ】 フルカップ Date. MB1010 レディース ブラック M</code>                                                                                        | <code>0.1</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 1
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | spearman_dot |
|:------:|:----:|:-------------:|:------------:|
| 0.0934 | 500  | 0.2140        | -            |
| 0.1867 | 1000 | 0.2089        | 0.2560       |
| 0.2801 | 1500 | 0.2060        | -            |
| 0.3734 | 2000 | 0.2032        | 0.2875       |
| 0.4668 | 2500 | 0.2017        | -            |
| 0.5601 | 3000 | 0.2013        | 0.2930       |
| 0.6535 | 3500 | 0.1992        | -            |
| 0.7468 | 4000 | 0.1991        | 0.3066       |
| 0.8402 | 4500 | 0.1998        | -            |
| 0.9335 | 5000 | 0.1966        | 0.3156       |


### Framework Versions
- Python: 3.11.15
- Sentence Transformers: 5.2.3
- Transformers: 5.2.0
- PyTorch: 2.10.0
- Accelerate: 1.14.0
- Datasets: 4.6.1
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->