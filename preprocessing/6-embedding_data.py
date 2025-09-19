import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertModel, BertTokenizer, ViTForImageClassification, ViTFeatureExtractor
import numpy as np
from tqdm import tqdm
import warnings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings('ignore')

# 配置参数
DATASET = 'Video_Games'  
DATA_PATH = f'./processed/{DATASET}'
OUTPUT_PATH = f'./processed/{DATASET}'
IMAGE_SIZE = 512
BATCH_SIZE = 32
RESNET_FEATURE_DIM = 768  
VIT_FEATURE_DIM = 768     
TEXT_FEATURE_DIM = 768    # BERT特征维度
COMMON_FEATURE_DIM = 768  # 统一后的特征维度
IMAGE_FOLDER = f'./origin_image/{DATASET}'

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data():
    item_desc_file = os.path.join(DATA_PATH, f'{DATASET}_item_desc.tsv')
    item_map_file = os.path.join(DATA_PATH, f'{DATASET}_i_map.tsv')

    summary_file = os.path.join(f'./processed/{DATASET}/', f'image_summary_description_{DATASET}.csv')

    item_desc = pd.read_csv(item_desc_file, sep='\t')
    item_map = pd.read_csv(item_map_file, sep='\t')
    summary_data = pd.read_csv(summary_file)
    
    return item_desc, item_map, summary_data

class ImageDataset(Dataset):
    def __init__(self, item_desc, image_folder, model_type='resnet'):
        self.item_desc = item_desc
        self.image_folder = image_folder
        self.model_type = model_type
        
        if model_type == 'vit':
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(
                'google/vit-base-patch16-224', 
                cache_dir="/llm_models/.HF_cache/"
            )
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.item_desc)
    
    def __getitem__(self, idx):
        item_id = self.item_desc.iloc[idx]['item_id']
        image_filename = str(self.item_desc.iloc[idx]['item_id']) + ".jpg" 
        
        try:
            image_path = os.path.join(self.image_folder, image_filename)
            
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
            else:
                img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
        except Exception as e:
            print(f"Error loading image for item {item_id}: {e}")
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
        
        if self.model_type == 'vit':
            img_processed = self.feature_extractor(images=img, return_tensors="pt")
            img_processed = {k: v.squeeze(0) for k, v in img_processed.items()}
            return item_id, img_processed
        else:
            img_tensor = self.transform(img)
            return item_id, img_tensor


def extract_vit_features(item_desc):    
    model = ViTForImageClassification.from_pretrained('openai/clip-vit-large-patch14', cache_dir="/llm_models/.HF_cache/")
    model = model.vit
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    dataset = ImageDataset(item_desc, IMAGE_FOLDER, model_type='vit')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lambda x: (
        [item[0] for item in x],
        {k: torch.stack([item[1][k] for item in x]) for k in x[0][1].keys()}
    ))
    
    features = {}
    with torch.no_grad():
        for item_ids, inputs in tqdm(dataloader, desc="ViT"):
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs).last_hidden_state[:, 0, :] 
            outputs = outputs.cpu()
            
            for i, item_id in enumerate(item_ids):
                features[int(item_id)] = outputs[i].numpy()
    
    return features

def extract_text_features(summary_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/llm_models/.HF_cache/")
    model = BertModel.from_pretrained('bert-base-uncased', cache_dir="/llm_models/.HF_cache/")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    features = {}
    with torch.no_grad():
        for _, row in tqdm(summary_data.iterrows(), total=len(summary_data), desc="bert"):
            item_id = int(row['item_id'])
            text = row['summary'] if not pd.isna(row['summary']) else ""
            
            if not text:
                text = "No description available"
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs)
            text_feature = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            features[item_id] = text_feature
    
    return features

def align_feature_dimensions(image_features, text_features, feature_dim, common_dim=COMMON_FEATURE_DIM):
    img_dim = next(iter(image_features.values())).shape[0]
    text_dim = next(iter(text_features.values())).shape[0]
    
    img_transform = nn.Linear(img_dim, common_dim)
    text_transform = nn.Linear(text_dim, common_dim)
    
    all_item_ids = sorted(set(image_features.keys()) | set(text_features.keys()))
    
    img_ids = []
    img_feats = []
    for item_id in all_item_ids:
        if item_id in image_features:
            img_ids.append(item_id)
            img_feats.append(image_features[item_id])
        else:
            img_ids.append(item_id)
            img_feats.append(np.zeros(img_dim))
    
    text_ids = []
    text_feats = []
    for item_id in all_item_ids:
        if item_id in text_features:
            text_ids.append(item_id)
            text_feats.append(text_features[item_id])
        else:  
            text_ids.append(item_id)
            text_feats.append(np.zeros(text_dim))
    
    img_feats_tensor = torch.tensor(np.array(img_feats), dtype=torch.float32)
    text_feats_tensor = torch.tensor(np.array(text_feats), dtype=torch.float32)
    
    img_feats_aligned = img_transform(img_feats_tensor)
    text_feats_aligned = text_transform(text_feats_tensor)
    
    aligned_img_features = {item_id: feat for item_id, feat in zip(img_ids, img_feats_aligned)}
    aligned_text_features = {item_id: feat for item_id, feat in zip(text_ids, text_feats_aligned)}
        
    return aligned_img_features, aligned_text_features


def save_features(clip_vit_features, text_features, item_map):
    
    max_item_id = item_map['item_id'].max()
    
    clip_vit_matrix = torch.zeros(max_item_id + 1, COMMON_FEATURE_DIM)
    text_matrix = torch.zeros(max_item_id + 1, COMMON_FEATURE_DIM)

    
    for item_id, feature in clip_vit_features.items():
        if 0 <= item_id <= max_item_id:
            clip_vit_matrix[item_id] = feature
    
    for item_id, feature in text_features.items():
        if 0 <= item_id <= max_item_id:
            text_matrix[item_id] = feature
    
    torch.save(clip_vit_matrix, os.path.join(OUTPUT_PATH, 'clip_img_feat.pt')) 
    torch.save(text_matrix, os.path.join(OUTPUT_PATH, 'image_summary_description_text_feat.pt'))



def main():

    item_desc, item_map, summary_data = load_data()
    
    text_features = extract_text_features(summary_data)
    
    vit_features = extract_vit_features(item_desc)
    aligned_vit, aligned_text_vit = align_feature_dimensions(vit_features, text_features, VIT_FEATURE_DIM)
        
    save_features(aligned_vit, aligned_text_vit, item_map)


if __name__ == "__main__":
    main()