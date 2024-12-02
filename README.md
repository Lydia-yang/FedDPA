# Dual-Personalizing Adapter for Federated Foundation Models

Implementation of the paper accepted by NeurIPS 2024: [Dual-Personalizing Adapter for Federated Foundation Models](https://arxiv.org/abs/2403.19211).

## Requirments
The code requires some dependencies (Python=3.8)  as specified in `requirements.txt`. Please follow the relevant libraries to install or run:
```bash
pip install -r requirements.txt
```
If `transformers` doesn't work, install it from source as:
```bash
pip install -U git+https://github.com/huggingface/transformers@de9255de27abfcae4a1f816b904915f0b1e23cd9
```

## Data Preparation
We construct two federated datasets from [FLAN](https://github.com/google-research/FLAN). Each dataset comprises eight tasks, with each client assigned a specific task. Both datasets are under the data folder.
  
## Running
The architecture of our code is based on [FederatedGPT](https://github.com/JayZhang42/FederatedGPT-Shepherd). Specifically, we adapt Hugging Face's [PEFT](https://github.com/huggingface/peft) to implement the joint tuning of global and local adapters. 

Example usage:
```bash
python main.py --global_model 'meta-llama/Llama-2-7b-hf'\
      --data_path  "./data/dataset1" \
      --output_dir  './lora-7b/'\
      --num_communication_rounds 20 \
      --num_clients  8 \
      --prompt_template_name 'alpaca_short' \
      --client_selection_frac 1 \
      --local_model True # if true, learn global and local adapters iteratively (FedDPA-T)
```

We can also tweak the hyperparameters:
```bash
python main.py --global_model 'meta-llama/Llama-2-7b-hf'\
      --data_path  "./data/dataset1" \
      --output_dir  './lora-7b/'\
      --num_communication_rounds 20 \
      --num_clients  8 \
      --client_selection_frac 1 \
      --local_num_epochs  10 \
      --local_batch_size  64 \
      --local_micro_batch_size 32 \
      --local_learning_rate 0.0003 \
      --lora_r 8 \
      --lora_target_modules='[q_proj,v_proj]' \
      --group_by_length \
      --local_model True \ # if true, learn global and local adapters iteratively (FedDPA-T)
      --local_weight 0.5 # if FedDPA-T, select weights of joint learning of global and local adapters
```

## Inference 

Inference only on the global model: 

```bash
python GlobalModel_generate.py \
      --load_8bit \
      --base_model 'meta-llama/Llama-2-7b-hf' \
      --lora_weights_path /output/path/to/lora_weights  \
      --lora_config_path /output/path/to/lora_config   \
      --prompt_template 'alpaca_short' \
      --output_file 'out/result.jsonl' \
      --test_file './data/dataset1/flan_test_200_selected_nstrict_1.jsonl'
      
```

Inference on both global and local adapters with Instance-wise Dynamic Weighting Mechanism: 

```bash
python GlobalModel_generate.py \
      --load_8bit \
      --base_model 'meta-llama/Llama-2-7b-hf' \
      --lora_weights_path /output/path/to/global_lora  \
      --lora_config_path /output/path/to/lora_config   \
      --prompt_template 'alpaca_short' \
      --output_file 'out/result.jsonl' \
      --test_file './data/dataset1/flan_test_200_selected_nstrict_1.jsonl' \
      --local True \
      --local_model_path /output/path/to/local_lora  \
      --input_file /data/path/to/trainng_instances \
      --auto True \
      -max_weight 0.5
      
```


## Citation
If you find this project helpful, please consider to cite the following paper:
```
@article{yang2024dual,
  title={Dual-Personalizing Adapter for Federated Foundation Models},
  author={Yang, Yiyuan and Long, Guodong and Shen, Tao and Jiang, Jing and Blumenstein, Michael},
  journal={arXiv preprint arXiv:2403.19211},
  year={2024}
}
```
