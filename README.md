# Sentence-Level Text Simplification for Dutch

**Play with the [demo](https://huggingface.co/spaces/BramVanroy/mai-simplification-nl-2023-demo)!**

This repository contains training code and detailed commands for three models for Dutch text simplification:

- [`BramVanroy/ul2-small-dutch-simplification-mai-2023`](https://huggingface.co/BramVanroy/ul2-small-dutch-simplification-mai-2023)
- [`BramVanroy/ul2-base-dutch-simplification-mai-2023`](https://huggingface.co/BramVanroy/ul2-base-dutch-simplification-mai-2023)
- [`BramVanroy/ul2-large-dutch-simplification-mai-2023`](https://huggingface.co/BramVanroy/ul2-large-dutch-simplification-mai-2023)


## 1. Split/get the data

This is only needed if you have access to the original, un-split data. You will probably just want to use
the pre-split data that is a result of this process. You can find this dataset on
[the hub](https://huggingface.co/datasets/BramVanroy/chatgpt-dutch-simplification) or use it as a dataset_name in
your scripts as `BramVanroy/chatgpt-dutch-simplification`.

If you still want to process your own data, you can for instance make 80% train, 10% dev, 10% test splits.

```shell
python make_split.py data/chatgpt_corpus_extended.csv 0.8 0.1 0.1 --sep ";" --shuffle_seed 42
```

## 2. Train

We finetune small, base, and large variants of `yhavinga/ul2-*-dutch`. A series of Dutch T5 models trained on the UL2
objective. 

For T5 finetuning tips, see [here](https://discuss.huggingface.co/t/t5-finetuning-tips/684/2).


### 2.1 Naive prefix testing

See information about source prefixes for the Dutch UL2 models 
[here](https://huggingface.co/yhavinga/ul2-small-dutch#intended-uses--limitations).
Possible prefixes are: `[NLU]`, `[NLG]`, or `[S2S]`. I ran a few default training runs with all possible prefixes
and did not find a major difference between the three, so for the remainder we'll stick with the `[NLG] ` prefix.
Adding a trailing space seems to work best.


### 2.2 Hyperparameter sweep

I ran a [wandb sweep](https://docs.wandb.ai/ref/python/sweep) for all three model sizes of `yhavinga/ul2-*-dutch`. We
use `adafactor` as an optimizer in all cases, as that is the suggested optimizer for T5 models and was also used in the
pretraining stage of the models. The optimization objective is to maximize sari+rougeLsum evaluation scores.
(You can also optimize for a minimal loss with `--hparam_optimize_for_loss`.)

For each model I ran 16 trials, with the following spectrum (the defaults when running the `train.py` script):

```python
{
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
    "per_device_train_batch_size": {"values": [8, 12, 16, 20, 24, 28, 32]},
    "num_train_epochs": {"min": 8, "max": 40}
}
```

You can modify these with the following parameters:

- hparam_lr_min
- hparam_lr_max
- hparam_bs_min
- hparam_bs_max
- hparam_epoch_min
- hparam_epoch_max


Note the flag `--include_inputs_for_metrics`, which is needed to calculate the sari metric.

#### ul2-small-dutch
```shell
CUDA_VISIBLE_DEVICES=2 WANDB_PROJECT="mai-simplification-nl-small-2023" python train.py \
    --no_use_fast_tokenizer \
    --dataset_name BramVanroy/chatgpt-dutch-simplification \
    --overwrite_output_dir \
    --adafactor \
    --text_column source \
    --simple_column target \
    --predict_with_generate \
    --save_strategy epoch \
    --report_to wandb \
    --log_level info \
    --source_prefix "[NLG] " \
    --include_inputs_for_metrics \
    \
    --do_hparams_search \
    \
    --model_name_or_path yhavinga/ul2-small-dutch \
    --output_dir models/ul2-small--hparam-search
```

**Best hyperparameters**

```json
{
    "learning_rate": 0.0006370158604635734,
    "num_train_epochs": 37,
    "per_device_train_batch_size": 20
}
```

The best run in the hyperparameter search was `run-t4jeq9fg` so for the evaluation, we use its last checkpoint
`checkpoint-1887`.

Important note: the sweep showed that higher number of epochs yielded better results. It is possible that even more 
than 40 epochs would give you even a better model.

**Evaluate** 

```shell
CUDA_VISIBLE_DEVICES=2 WANDB_PROJECT="mai-simplification-nl-small-2023" python train.py \
    --no_use_fast_tokenizer \
    --dataset_name BramVanroy/chatgpt-dutch-simplification \
    --overwrite_output_dir \
    --text_column source \
    --simple_column target \
    --log_level info \
    --source_prefix "[NLG] " \
    --include_inputs_for_metrics \
    \
    --predict_with_generate \
    --generation_num_beams 3 \
    --do_eval \
    --do_predict \
    \
    --model_name_or_path /home/local/vanroy/mai-simplification-nl-2023/models/ul2-small--hparam-search/run-t4jeq9fg/checkpoint-1887 \
    --output_dir /home/local/vanroy/mai-simplification-nl-2023/models/ul2-small--hparam-search/run-t4jeq9fg/
```

**Final model**: [BramVanroy/ul2-small-dutch-simplification-mai-2023](https://huggingface.co/BramVanroy/ul2-small-dutch-simplification-mai-2023)


#### ul2-base-dutch
```shell
CUDA_VISIBLE_DEVICES=1 python train.py \
    --no_use_fast_tokenizer \
    --dataset_name BramVanroy/chatgpt-dutch-simplification \
    --overwrite_output_dir \
    --adafactor \
    --text_column source \
    --simple_column target \
    --predict_with_generate \
    --save_strategy epoch \
    --report_to wandb \
    --log_level info \
    --source_prefix "[NLG] " \
    --include_inputs_for_metrics \
    \
    --do_hparams_search \
    \
    --model_name_or_path yhavinga/ul2-base-dutch \
    --output_dir models/ul2-base--hparam-search
```


**Best hyperparameters**

```json
{
    "learning_rate": 0.00026885245616406115,
    "num_train_epochs": 26,
    "per_device_train_batch_size": 12
}
```

The best run in the hyperparameter search was `dkgwv7w4` so for the evaluation, we use its last checkpoint
`checkpoint-2210`.

**Evaluate** 

```shell
CUDA_VISIBLE_DEVICES=2 python train.py \
    --no_use_fast_tokenizer \
    --dataset_name BramVanroy/chatgpt-dutch-simplification \
    --overwrite_output_dir \
    --text_column source \
    --simple_column target \
    --log_level info \
    --source_prefix "[NLG] " \
    --include_inputs_for_metrics \
    \
    --predict_with_generate \
    --generation_num_beams 3 \
    --do_eval \
    --do_predict \
    \
    --model_name_or_path /home/local/vanroy/mai-simplification-nl-2023/models/ul2-base--hparam-search/run-dkgwv7w4/checkpoint-2210 \
    --output_dir /home/local/vanroy/mai-simplification-nl-2023/models/ul2-base--hparam-search/run-dkgwv7w4/
```

**Final model**: [BramVanroy/ul2-base-dutch-simplification-mai-2023](https://huggingface.co/BramVanroy/ul2-base-dutch-simplification-mai-2023)


#### ul2-large-dutch
```shell
CUDA_VISIBLE_DEVICES=3 WANDB_PROJECT="mai-simplification-nl-large-2023" python train.py \
    --no_use_fast_tokenizer \
    --dataset_name BramVanroy/chatgpt-dutch-simplification \
    --overwrite_output_dir \
    --adafactor \
    --text_column source \
    --simple_column target \
    --predict_with_generate \
    --save_strategy epoch \
    --report_to wandb \
    --log_level info \
    --source_prefix "[NLG] " \
    --include_inputs_for_metrics \
    \
    --do_hparams_search \
    \
    --model_name_or_path yhavinga/ul2-large-dutch \
    --output_dir models/ul2-large--hparam-search
```

Note: in retrospect, I probably should have allowed a smaller learning rate for this large model. `1e-4` is still
quite large. You may achieve better results when training with a smaller learning rate.

**Best hyperparameters**

```json
{
    "learning_rate": 0.0002927210895006501,
    "num_train_epochs": 27,
    "per_device_train_batch_size": 32
}
```

The best run in the hyperparameter search was `zu8po8md` so for the evaluation, we use its last checkpoint
`checkpoint-864`.

**Evaluate** 

```shell
CUDA_VISIBLE_DEVICES=3 python train.py \
    --no_use_fast_tokenizer \
    --dataset_name BramVanroy/chatgpt-dutch-simplification \
    --overwrite_output_dir \
    --text_column source \
    --simple_column target \
    --log_level info \
    --source_prefix "[NLG] " \
    --include_inputs_for_metrics \
    \
    --predict_with_generate \
    --generation_num_beams 3 \
    --do_eval \
    --do_predict \
    \
    --model_name_or_path /home/local/vanroy/mai-simplification-nl-2023/models/ul2-large--hparam-search/run-zu8po8md/checkpoint-864 \
    --output_dir /home/local/vanroy/mai-simplification-nl-2023/models/ul2-large--hparam-search/run-zu8po8md/
```

**Final model**: [BramVanroy/ul2-large-dutch-simplification-mai-2023](https://huggingface.co/BramVanroy/ul2-large-dutch-simplification-mai-2023)
