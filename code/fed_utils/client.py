import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

   
    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                  )
                                                  )

    def set_glocal(self, local_path=None, epoch=0):
        if epoch <1:
            return
        if local_path is not None:
            single_output_dir = local_path
        else:
            single_output_dir = os.path.join(self.output_dir,str(epoch), "local_output_{}".format(self.client_id), 'local')
        tmp_path = single_output_dir+'/pytorch_model.bin'
        self.model.add_local_model('local', tmp_path)
        self.model.set_adapter('default')
        self.model.set_local(['local'], [0.5,0.5])

    def set_local(self, local_path=None, epoch=0, a=0.5, prv=None):
        if epoch==0:
            tmp_path = local_path
        else:
            if epoch == 1:
                single_output_dir = os.path.join(self.output_dir, str(epoch-1), "local_output_{}".format(self.client_id))
                tmp_path = None
            else:
                if prv is None:
                    single_output_dir = os.path.join(self.output_dir,str(epoch-1), "local_output_{}".format(self.client_id), 'local')
                else:
                    tmp_epoch = 0
                    if self.client_id in prv.keys():
                        tmp_epoch = prv[self.client_id]
                    single_output_dir = os.path.join(self.output_dir,str(tmp_epoch), "local_output_{}".format(self.client_id), 'local')
                tmp_path = single_output_dir+'/pytorch_model.bin'
                if not os.path.exists(tmp_path):
                    tmp_path = None
                print("******************",tmp_path,"*************************")
        self.model.add_local_model('local', tmp_path)
        self.model.set_adapter('local')
        self.model.set_local(['default'], [a,1-a])
        # tmp = [param[0] for name, param in self.model.named_parameters() if 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight' in name]
        # print('combined1', tmp[0])
        # tmp = [param[0] for name, param in self.model.named_parameters() if 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.local.weight' in name]
        # print('combined2', tmp_path, tmp[0])
        # print('model', self.model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.lora_A.local.weight'][0])

    def initiate_local_training(self, weight=0.5, local=False, local_path=None, epoch=0):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        
        if local:
            if local_path is not None:
                single_output_dir = local_path
                print("************************load local path******************************\n")
                #print(single_output_dir)
            else:
                single_output_dir = os.path.join(self.output_dir,str(epoch), "local_output_{}".format(self.client_id), 'local')
            global_adapter_weight = self.params_dict_new
            #print('global', global_adapter_weight['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'][0])
            if os.path.exists(single_output_dir+"/pytorch_model.bin"):
                local_adapters_weights = torch.load(single_output_dir+"/pytorch_model.bin")
                #print('local', local_adapters_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'][0])
                for k in global_adapter_weight.keys():
                    if "default" in k:
                        tmpk = k.split('default.')
                        k_local = tmpk[0]+tmpk[-1]
                        self.params_dict_new[k] = (1-weight)*local_adapters_weights[k_local] + weight*global_adapter_weight[k]
                #print('combined', self.params_dict_new['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'][0])
                adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_new, "default")
                set_peft_model_state_dict(self.model, adapter_weight, "default")

        # self.model.state_dict = (
        #     lambda instance, *_, **__: get_peft_model_state_dict(
        #         instance, self.params_dict_new, "default"
        #     )
        # ).__get__(self.model, type(self.model))
        new_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_new, "default")
        set_peft_model_state_dict(self.model, new_adapter_weight, "default")

        # tmp = [param[0] for name, param in self.model.named_parameters() if 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight' in name]
        # print('combined_default', tmp[0])

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set, config,local=False,glocal=False):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        #new_adapter_weight = self.model.state_dict()
        if local:
            single_output_dir = os.path.join(self.output_dir,str(epoch), "local_output_{}".format(self.client_id), 'local')
            adapter_name = 'local'
        else:
            single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
            adapter_name = 'default'
        os.makedirs(single_output_dir, exist_ok=True)
        new_adapter_weight = get_peft_model_state_dict(
                self.model, adapter_name=adapter_name
            )
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")
        config.save_pretrained(single_output_dir)
        
        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id
        
        if local or (glocal and epoch>0):
            self.model.set_adapter("default")
            self.model.unset_local()
            self.model.delete_adapter('local')

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
