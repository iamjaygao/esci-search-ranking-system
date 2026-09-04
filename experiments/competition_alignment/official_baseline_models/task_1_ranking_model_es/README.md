---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:123968
- loss:CosineSimilarityLoss
base_model: sentence-transformers/multi-qa-mpnet-base-dot-v1
widget:
- source_sentence: hp all in one 27
  sentences:
  - Oris Oris Aquis Date 01 733 7732 4157-07 8 21 05PEB.
  - Botellas de cola de goma La Asturiana - Clásica gomitas en forma de botella, con
    delicioso sabor cola, con recubrimiento de azúcar, en bolsas de 1 kilo, sin gluten
  - HP All-in-One 22-df0007ns - Ordenador de Sobremesa de 21.5" FHD (Procesador AMD
    Ryzen 3, 8GB RAM, 256GB SSD, AMD Radeon Graphics, Windows 10) Blanco
- source_sentence: papel infantil pared
  sentences:
  - 'Black: Military History, Vol. III: Volume 3 (Critical Concepts in Military, Strategic,
    and Security Studies)'
  - Pegatinas y Vinilos para Decoración de Pared | Puntos Círculos Imperfectos | Adhesivos
    decorativos Nórdico Infantil | 75uds | Verdes
  - HAPPERS Pack 4 Relleno de Cojín 45x45 Fibra Hueca Siliconada de Gran Densidad
    para Cojines de Sofá o Cama
- source_sentence: nariz roja luz
  sentences:
  - 90 Piezas Colgantes del Encanto de JoyeríA, Colgantes Diy Mezclados Dijes Para
    BisuteríA，Encantos Colgantes Mixtos, Accesorios de DecoracióN, Laveros, Pulseras,
    Collares, Pendientes
  - JOYOOY Halloween LED Luminoso Nariz Payaso Accesorios de iluminación Brillante
    Suministros Accesorios para niños Broma
  - Antiinflamatorio para perros y gatos | Con colágeno + cúrcuma + condroitina y
    magnesio para recuperar su energía y movilidad | Combate el dolor y la inflamación
    en tu mascota | 50 gominolas sin azúcar
- source_sentence: puros habanos para fumar
  sentences:
  - Puck, el de la colina (Pulgarcito 6)
  - Colorante Jabón - 12 Colores Colorante de Bomba de Baño Líquido para Fabricación
    de Jabón - Tinte de Jabón para Kit de Suministros de Elaboración Jabon DIY, Bomba
    de Baño, Manualidades
  - 'Cómo construir la autodisciplina: Resiste tentaciones y alcanza tus metas a largo
    plazo'
- source_sentence: cordones blancos sin nudo
  sentences:
  - Amd A4-Series 3400 - Microprocesador
  - Diealles Shine Cordones Elásticos Sin Nudo, 6 Pares 105CM Cordones Elásticos Sin
    Nudo con Hebilla Metal, Ajustables Cordones de Zapatos Sin Nudos para Zapatillas
    Deportivas
  - Naturseed Psyllium Husk Cascara Ecológico Polvo - Pureza 99% - Orgánico - Sin
    gluten - Alto en fibra - Panificable - Especial para Pan, Reposteria - Mejora
    el tracto intestinal - - Saciante (200Gr)
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
      value: 0.33929885989635294
      name: Pearson Dot
    - type: spearman_dot
      value: 0.34893475330782464
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
    'cordones blancos sin nudo',
    'Diealles Shine Cordones Elásticos Sin Nudo, 6 Pares 105CM Cordones Elásticos Sin Nudo con Hebilla Metal, Ajustables Cordones de Zapatos Sin Nudos para Zapatillas Deportivas',
    'Naturseed Psyllium Husk Cascara Ecológico Polvo - Pureza 99% - Orgánico - Sin gluten - Alto en fibra - Panificable - Especial para Pan, Reposteria - Mejora el tracto intestinal - - Saciante (200Gr)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[44.9756, 31.6434, 18.9250],
#         [31.6434, 45.6345, 18.0975],
#         [18.9250, 18.0975, 45.6883]])
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
| pearson_dot      | 0.3393     |
| **spearman_dot** | **0.3489** |

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

* Size: 123,968 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                        | label                                                          |
  |:--------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                           | string                                                                            | float                                                          |
  | details | <ul><li>min: 3 tokens</li><li>mean: 8.88 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 42.41 tokens</li><li>max: 97 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.46</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                         | sentence_1                                                                                                                      | label            |
  |:---------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>pirelli zapatillas</code>                    | <code>PUMA Viz Runner, Zapatillas de Running Hombre, Negro Black White, 44 EU</code>                                            | <code>0.1</code> |
  | <code>anillo de oro para medio dedo letra a</code> | <code>KnSam Anillo Oro de 18K, Marca de Amor Anillo Solitario, Mujer Talla 13 y Hombre Talla 23,5 (Precio por 2 Anillos)</code> | <code>1.0</code> |
  | <code>el terror rojo en españa</code>              | <code>El triunfo de la democracia en España: De Franco a Felipe González pasando por Juan Carlos (Historia)</code>              | <code>0.0</code> |
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
| 0.1291 | 500  | 0.2075        | -            |
| 0.2581 | 1000 | 0.2020        | 0.2399       |
| 0.3872 | 1500 | 0.1984        | -            |
| 0.5163 | 2000 | 0.1950        | 0.2888       |
| 0.6453 | 2500 | 0.1899        | -            |
| 0.7744 | 3000 | 0.1889        | 0.3304       |
| 0.9035 | 3500 | 0.1838        | -            |
| 1.0    | 3874 | -             | 0.3489       |


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