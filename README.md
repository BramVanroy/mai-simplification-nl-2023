## 1. Split the data

Making 80% train, 10% dev, 10% test splits.

```shell
python make_split.py data/chatgpt_corpus_extended.csv 0.8 0.1 0.1 --sep ";" --shuffle_seed 42
```


## 2. Train

See information about prefixes [here](https://huggingface.co/yhavinga/ul2-small-dutch#intended-uses--limitations) and
T5 finetuning tips [here](https://discuss.huggingface.co/t/t5-finetuning-tips/684/2).

Possible prefixes for the ul2 Dutch models: `[NLU]`, `[NLG]`, or `[S2S]`. (We end up using `[NLG]`, see below.)


### 2.1 Naive prefix testing

I ran a few default training runs with all possible prefixes (above) and did not find a major difference between the
three, so for the remainder we'll stick with the `[NLG] ` prefix. Adding a trailing space seems to work best.


### 2.2 Hyperparameter sweep

I ran a [wandb sweep](https://docs.wandb.ai/ref/python/sweep) for all three model sizes of `yhavinga/ul2-*-dutch`. We
use adafactor as an optimizer in all cases, as that is the suggested optimizer for T5 models and was also used in the
pretraining stage of the models. The optimization objective is to maximize sari+rougeLsum evaluation scores.
(You can also optimize for a minimal loss with `--hparam_optimize_for_loss`.)

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

#### ul2-base-dutch
```shell
CUDA_VISIBLE_DEVICES=2 WANDB_PROJECT="mai-simplification-nl-base-2023" python train.py \
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
