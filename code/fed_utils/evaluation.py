from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def global_evaluation(model, data_files, generate_and_tokenize_prompt, batch_size, device):
    data = load_dataset("json", data_files=data_files)
    val_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = val_data.with_format('torch')
    data_loader = DataLoader(val_data, batch_size=batch_size)
    loss = []
    for inputs in tqdm(data_loader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'labels']}
            #inputs = inputs.to(device)
            #print(batch)
            output = model(**batch)
            loss.append(output[0].cpu())
    loss = sum(loss) / len(loss)
    return loss


