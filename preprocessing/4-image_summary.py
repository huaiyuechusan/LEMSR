import torch
from multiprocess import set_start_method
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from datasets import load_dataset
from torchvision import transforms
from PIL import ImageOps
from torch.cuda.amp import autocast
import os
import gc  
import pandas as pd


os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" 

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(model_id,cache_dir='/llm_models/.HF_cache/',attn_implementation="flash_attention_2", torch_dtype=torch.float16,
                                                          ).eval()


prompt_video = "[INST] <image>\nPlease describe this image, which is a cover about Sports_and_Outdoors product" \
         " Provide a detailed description in one continuous paragraph, including content information and visual features such as colors, objects, text," \
         " and any notable elements present in the image.[/INST]"

def add_image_file_path(example):
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    example['item_id'] = filename
    return example

dataset_name = "Sports_and_Outdoors"
img_dir = f"./origin_image/{dataset_name}"  
dataset = load_dataset("imagefolder", data_dir=img_dir)
dataset = dataset.map(lambda x: add_image_file_path(x))
print(dataset)

processor = AutoProcessor.from_pretrained(model_id, return_tensors=torch.float16)


def gpu_computation(batch, rank):
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)

    batch_images = batch['image']

    max_width = max(img.width for img in batch_images)
    max_height = max(img.height for img in batch_images)

    padded_images = []
    for img in batch_images:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height

            padding = (
            delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

            new_img = ImageOps.expand(img, border=padding, fill='black')
            padded_images.append(new_img)

    batch['image'] = padded_images

    model_inputs = processor([prompt_video for i in range(len(batch['image']))], batch['image'], return_tensors="pt",padding=True).to(device)

    with torch.no_grad() and autocast():
        outputs = model.generate(**model_inputs, max_new_tokens=200)

    ans = processor.batch_decode(outputs, skip_special_tokens=True)
    ans = [a.split("[/INST]")[1] for a in ans]

    del model_inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return {"summary": ans}


if __name__ == "__main__":
    set_start_method("spawn")
    datasets_name = dataset_name

    total_samples = len(dataset['train'])
    chunk_size = total_samples // 5
    all_dfs = []
    
    for i in range(5):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < 5 else total_samples
        
        current_subset = dataset['train'].select(range(start_idx, end_idx))
        
        updated_subset = current_subset.map(
            gpu_computation,
            batched=True,
            batch_size=32,   
            with_rank=True,
            num_proc=2  
        )
        
        item_id = updated_subset['item_id']
        summary = updated_subset['summary']
        df = pd.DataFrame({'item_id': item_id, 'summary': summary})
        
        chunk_filename = f'./processed/{datasets_name}/image_summary_part_{datasets_name}_{i+1}.csv'
        df.to_csv(chunk_filename, index=False)

        torch.cuda.empty_cache()
        gc.collect()
    

    all_dfs = []
    for i in range(5):
        chunk_filename = f'./processed/{datasets_name}/image_summary_part_{datasets_name}_{i+1}.csv'
        chunk_df = pd.read_csv(chunk_filename)
        all_dfs.append(chunk_df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(f'./processed/{datasets_name}/image_summary_{datasets_name}.csv', index=False)
