import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional
import pandas as pd
import numpy as np
from typing import Tuple

class Consistency_Layer(nn.Module):
    
    def __init__(self, feat_dim, output_dim, eps=1e-5):
        super(Consistency_Layer, self).__init__()
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.eps = eps
        self.projection_x = nn.Linear(feat_dim, output_dim)
        self.projection_y = nn.Linear(feat_dim, output_dim)
        self.debug = False
        self.update_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.max_failures = 5
        self.consistency_enabled = True
    
    def compute_consistency_projections(self, X, Y):
       
        batch_size, seq_len, feat_dim = X.shape
        
        X_flat = X.reshape(-1, self.feat_dim)
        Y_flat = Y.reshape(-1, self.feat_dim)
        
        X_mean = torch.mean(X_flat, dim=0, keepdim=True)
        Y_mean = torch.mean(Y_flat, dim=0, keepdim=True)
        X_centered = X_flat - X_mean
        Y_centered = Y_flat - Y_mean


        n_samples = X_flat.size(0)
        cov_xy = torch.matmul(X_centered.t(), Y_centered) / (n_samples - 1)
        cov_xx = torch.matmul(X_centered.t(), X_centered) / (n_samples - 1) + torch.eye(self.feat_dim, device=X.device) * self.eps
        cov_yy = torch.matmul(Y_centered.t(), Y_centered) / (n_samples - 1) + torch.eye(self.feat_dim, device=Y.device) * self.eps
        
        cov_xx_sqrt_inv = torch.inverse(torch.linalg.cholesky(cov_xx))
        cov_yy_sqrt_inv = torch.inverse(torch.linalg.cholesky(cov_yy))
        
        C = torch.matmul(torch.matmul(cov_xx_sqrt_inv, cov_xy), cov_yy_sqrt_inv)
        U, _, V = torch.svd(C)
        
        U = U[:, :self.output_dim]
        V = V[:, :self.output_dim]
        
        proj_X = torch.matmul(cov_xx_sqrt_inv, U)
        proj_Y = torch.matmul(cov_yy_sqrt_inv, V)

        X_proj = torch.matmul(X_centered, proj_X)
        Y_proj = torch.matmul(Y_centered, proj_Y)

        X_proj = X_proj.reshape(batch_size, seq_len, self.output_dim)
        Y_proj = Y_proj.reshape(batch_size, seq_len, self.output_dim)
        
        return X_proj, Y_proj
        

    
    def forward(self, X, Y):
        
        X_proj = self.projection_x(X)
        Y_proj = self.projection_y(Y)
        
        if not self.consistency_enabled:
            return X_proj, Y_proj
        
        if self.training: 
            self.update_count += 1
            if self.update_count % 100 == 0: 
                try:
                    with torch.no_grad(): 
                        X_consistency, Y_consistency = self.compute_consistency_projections(X, Y)
                        
                        if X_consistency is None or Y_consistency is None:
                            self.consecutive_failures += 1
                            return X_proj, Y_proj
                        
                        batch_size, seq_len, feat_dim = X.shape
                        X_flat = X.reshape(-1, feat_dim)
                        Y_flat = Y.reshape(-1, feat_dim)
                        X_consistency_flat = X_consistency.reshape(-1, self.output_dim)
                        Y_consistency_flat = Y_consistency.reshape(-1, self.output_dim)
                        
                        try:
                            X_weight = torch.matmul(torch.pinverse(X_flat), X_consistency_flat)
                            Y_weight = torch.matmul(torch.pinverse(Y_flat), Y_consistency_flat)

                            if X_weight.shape == (feat_dim, self.output_dim) and Y_weight.shape == (feat_dim, self.output_dim):
                                if torch.isnan(X_weight).any() or torch.isnan(Y_weight).any() or \
                                torch.isinf(X_weight).any() or torch.isinf(Y_weight).any():
                                    self.consecutive_failures += 1
                                else:
                                    self.projection_x.weight.data = X_weight.t()
                                    self.projection_y.weight.data = Y_weight.t()
                                    self.success_count += 1
                                    self.consecutive_failures = 0  
                            else:
                                self.consecutive_failures += 1
                        except Exception as e:
                            self.consecutive_failures += 1
                            
                except Exception as e:
                    self.consecutive_failures += 1
        
        return X_proj, Y_proj


class HypridAttentionTransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=8,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        consistency_ratio=0.5, 
    ):
        super(HypridAttentionTransformerEncoder, self).__init__()
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.consistency_ratio = consistency_ratio

        self.n_orig_heads = int(n_heads * (1 - consistency_ratio))
        self.n_consistency_heads = n_heads - self.n_orig_heads

        self.consistency_layer = Consistency_Layer(hidden_size, hidden_size)

        self.orig_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=self.n_orig_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        
        self.consistency_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=self.n_consistency_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
    def compute_consistency_loss(self, orig_features, consistency_features):

        orig_norm = torch.nn.functional.normalize(orig_features, p=2, dim=-1)
        consistency_norm = torch.nn.functional.normalize(consistency_features, p=2, dim=-1)

        cosine_sim = torch.sum(orig_norm * consistency_norm, dim=-1)
        
        consistency_loss = 1.0 - cosine_sim.mean()

        batch_size, seq_len, hidden_size = orig_features.shape
        orig_flat = orig_features.reshape(-1, hidden_size)
        consistency_flat = consistency_features.reshape(-1, hidden_size)
        
        orig_mean = torch.mean(orig_flat, dim=0, keepdim=True)
        consistency_mean = torch.mean(consistency_flat, dim=0, keepdim=True)
        orig_centered = orig_flat - orig_mean
        consistency_centered = consistency_flat - consistency_mean

        n_samples = orig_flat.size(0)
        cov_orig_consistency = torch.matmul(orig_centered.t(), consistency_centered) / (n_samples - 1)

        diag_sum = torch.trace(cov_orig_consistency)
        alignment_loss = 1.0 - (diag_sum / hidden_size)

        combined_loss = torch.tensor(consistency_loss + alignment_loss, device=orig_features.device)
        
        return combined_loss

            
    def forward(self, hidden_states, attention_mask, cross_modal_states=None, output_all_encoded_layers=False):
    
        batch_size, seq_len, hidden_size = hidden_states.shape

        if cross_modal_states is None:
            cross_modal_states = hidden_states
        
        hidden_proj, cross_modal_proj = self.consistency_layer(hidden_states, cross_modal_states)
        
        orig_all_encoder_layers = self.orig_encoder(
            hidden_states, 
            attention_mask, 
            output_all_encoded_layers=True
        )
        
        consistency_all_encoder_layers = self.consistency_encoder(
            hidden_proj, 
            attention_mask, 
            output_all_encoded_layers=True
        )
        
        all_encoder_layers = []
        consistency_losses = []
        
        for layer_idx, (orig_layer, consistency_layer) in enumerate(zip(orig_all_encoder_layers, consistency_all_encoder_layers)):

            gate = self.fusion_gate(torch.cat([orig_layer, consistency_layer], dim=-1))
            fused_layer = gate * orig_layer + (1 - gate) * consistency_layer
            all_encoder_layers.append(fused_layer)
            
            layer_consistency_loss = self.compute_consistency_loss(orig_layer, consistency_layer)
            consistency_losses.append(layer_consistency_loss)
        
        if consistency_losses:
            tensor_losses = [loss for loss in consistency_losses if isinstance(loss, torch.Tensor)]
            if tensor_losses:
                total_consistency_loss = sum(tensor_losses) / len(tensor_losses)
        
        if not output_all_encoded_layers:
            all_encoder_layers = all_encoder_layers[-1]
        
        return all_encoder_layers, total_consistency_loss


class LEMSR(SequentialRecommender):

    def __init__(self, config, dataset):
        super(LEMSR, self).__init__(config, dataset)

        dataset=config["dataset"]
        feature_path=f"./dataset/{dataset}/"
        
        img_feat_name = "vit_img_feat.pt"  
        img_feat = nn.Embedding.from_pretrained(torch.load(feature_path + img_feat_name),freeze=True)

        text_feat_name = "image_summary_description_text_feat.pt"
        text_feat = nn.Embedding.from_pretrained(torch.load(feature_path + text_feat_name),freeze=True)

        self.img_feat=nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,img_feat.weight.shape[-1]),img_feat.weight),dim=0
        ),freeze=True)


        self.text_feat = nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,text_feat.weight.shape[-1]),text_feat.weight),dim=0
        ),freeze=True)

        self.img_alpha=torch.nn.Parameter(torch.tensor([1.]))
        self.text_beta=torch.nn.Parameter(torch.tensor([1.]))

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        
        self.consistency_ratio = config["consistency_ratio"]
        
        self.consistency_loss_weight = config["consistency_loss_weight"]
        
        self.use_temporal_weight = config["use_temporal_weight"]
        self.temporal_decay_rate = config["temporal_decay_rate"]
        self.order_aware_loss_weight = config["order_aware_loss_weight"]
 
        self.inner_size = config[
            "inner_size"
        ]  
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        text_item_feat = torch.load(feature_path + text_feat_name)
        img_item_feat = torch.load(feature_path + img_feat_name)

        img_linear = nn.Linear(img_item_feat.shape[1], text_item_feat.shape[1])
        img_item_feat_transformed = img_linear(img_item_feat)

        text_norm = torch.nn.functional.normalize(text_item_feat, p=2, dim=1)
        img_norm = torch.nn.functional.normalize(img_item_feat_transformed, p=2, dim=1)
        modal_scores = torch.sigmoid(torch.sum(text_norm * img_norm, dim=1))

        alpha_weights = modal_scores.unsqueeze(1)
        beta_weights = 1 - alpha_weights

        self.init_alpha = nn.Parameter(torch.tensor(0.5))
        self.init_beta = nn.Parameter(torch.tensor(0.5))

        fusion_feat = (self.init_alpha * alpha_weights) * text_item_feat + (self.init_beta * beta_weights) * img_item_feat_transformed
        

        self.hidden_size = fusion_feat.shape[1]    
        self.item_embedding = nn.Embedding.from_pretrained(
            torch.cat((torch.zeros(1, self.hidden_size), fusion_feat), dim=0),
            freeze=False
        )

        self.use_enhanced_position = config["use_enhanced_position"]
        if self.use_enhanced_position:
            
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
            self.relative_position_embedding = nn.Embedding(2 * self.max_seq_length - 1, self.hidden_size)

            self.position_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
        
        self.img_trans = nn.Sequential(
            nn.Linear(self.img_feat.weight.shape[-1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.text_trans = nn.Sequential(
            nn.Linear(self.text_feat.weight.shape[-1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        self.trm_encoder = HypridAttentionTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            consistency_ratio=self.consistency_ratio,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.img_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.text_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        self.global_pooling = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

        self.id_interest_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        
        self.img_interest_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        
        self.text_interest_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.n_heads,
            dropout=self.attn_dropout_prob
        )

        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def compute_temporal_weights(self, seq_len, attention_mask=None):
        positions = torch.arange(seq_len, dtype=torch.float, device=self.device)
        
        reversed_positions = seq_len - 1 - positions

        temporal_weights = torch.exp(-self.temporal_decay_rate * reversed_positions)
        
        temporal_weights = temporal_weights.unsqueeze(0)
        
        if attention_mask is not None:
            if len(attention_mask.shape) == 2: 
                temporal_weights = temporal_weights.expand(attention_mask.size(0), -1) 
                temporal_weights = temporal_weights * attention_mask
            
        temporal_weights = temporal_weights.unsqueeze(-1)
        
        weights_sum = torch.sum(temporal_weights, dim=1, keepdim=True)
        weights_sum = torch.clamp(weights_sum, min=1e-12)
        normalized_weights = temporal_weights / weights_sum
        
        return normalized_weights
        
    def compute_order_aware_loss(self, sequence_output, item_seq, item_seq_len):
    
        try:
            batch_size, seq_len, hidden_size = sequence_output.shape
            
            attention_mask = (item_seq > 0).float()  
            
            temporal_weights = self.compute_temporal_weights(seq_len, attention_mask)  
            

            prev_items = sequence_output[:, :-1, :] 
            next_items = sequence_output[:, 1:, :]  
            
            prev_norm = torch.nn.functional.normalize(prev_items, p=2, dim=-1)
            next_norm = torch.nn.functional.normalize(next_items, p=2, dim=-1)
            similarity = torch.sum(prev_norm * next_norm, dim=-1)  
            
            valid_mask = attention_mask[:, :-1] * attention_mask[:, 1:]  

            pair_weights = temporal_weights[:, :-1, 0]
            
            weighted_similarity = similarity * valid_mask * pair_weights 
            
            valid_pairs = torch.sum(valid_mask, dim=1)  
            valid_pairs = torch.clamp(valid_pairs, min=1)  
            
            batch_avg_sim = torch.sum(weighted_similarity, dim=1) / valid_pairs 

            target_similarity = 0.5
            order_loss = torch.mean((batch_avg_sim - target_similarity) ** 2)
            
            return order_loss
            
        except Exception as e:
            return torch.tensor(0.0, device=sequence_output.device)

    def extract_long_term_interest(self, sequence_output, attention_mask):

        attn_weights = self.global_pooling(sequence_output)  
        expanded_attention_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0  
        attn_weights = attn_weights + expanded_attention_mask
        attn_weights = torch.softmax(attn_weights, dim=1) 
    
        if self.use_temporal_weight:
            batch_size, seq_len, _ = sequence_output.shape
            temporal_weights = self.compute_temporal_weights(seq_len, attention_mask)  
            
            combined_weights = 0.5 * attn_weights + 0.5 * temporal_weights
            
            combined_weights_sum = torch.sum(combined_weights, dim=1, keepdim=True)
            combined_weights_sum = torch.clamp(combined_weights_sum, min=1e-12)
            combined_weights = combined_weights / combined_weights_sum
            
            long_term_interest = torch.sum(sequence_output * combined_weights, dim=1)  
            
        
        return long_term_interest

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        if self.use_enhanced_position:
            batch_size, seq_len = item_seq.shape
            
            pos_idx = torch.arange(seq_len, device=item_seq.device).unsqueeze(1)
            rel_pos_idx = pos_idx - pos_idx.transpose(0, 1)

            rel_pos_idx = rel_pos_idx + (seq_len - 1)

            rel_pos_emb = self.relative_position_embedding(rel_pos_idx)

            avg_rel_pos_emb = torch.mean(rel_pos_emb, dim=1)
            
            avg_rel_pos_emb = avg_rel_pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
            
            pos_gate = self.position_gate(torch.cat([position_embedding, avg_rel_pos_emb], dim=-1))
            enhanced_pos_emb = pos_gate * position_embedding + (1 - pos_gate) * avg_rel_pos_emb

            position_embedding = enhanced_pos_emb


        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        img_emb = self.img_feat(item_seq)
        img_emb = self.img_trans(img_emb)
        img_emb = self.img_LayerNorm(img_emb)
        img_emb = self.dropout(img_emb)
        img_input = img_emb + position_embedding

        text_emb = self.text_feat(item_seq)
        text_emb = self.text_trans(text_emb)
        text_emb = self.text_LayerNorm(text_emb)
        text_emb = self.dropout(text_emb)
        text_input = text_emb + position_embedding

        extended_attention_mask = self.get_attention_mask(item_seq)
        
        attention_mask = (item_seq > 0).float() 
        

        batch_size, seq_len, hidden_size = input_emb.shape
        
        img_input_reshaped = img_input.transpose(0, 1)  
        text_input_reshaped = text_input.transpose(0, 1)  
        input_emb_reshaped = input_emb.transpose(0, 1)  
        
        cross_modal_mask = (item_seq == 0).float() 
        cross_modal_mask = cross_modal_mask.masked_fill(cross_modal_mask == 1, -float('inf'))
        cross_modal_mask = cross_modal_mask.masked_fill(cross_modal_mask == 0, 0)
        
        img_text_attn_output, _ = self.cross_modal_attn(
            query=img_input_reshaped,
            key=text_input_reshaped,
            value=text_input_reshaped,
            key_padding_mask=cross_modal_mask
        )
        
        text_img_attn_output, _ = self.cross_modal_attn(
            query=text_input_reshaped,
            key=img_input_reshaped,
            value=img_input_reshaped,
            key_padding_mask=cross_modal_mask
        )
        
        img_text_attn_output = img_text_attn_output.transpose(0, 1)  
        text_img_attn_output = text_img_attn_output.transpose(0, 1)  
    
        cross_modal_fusion = torch.mul(img_text_attn_output, text_img_attn_output)

        id_output_all_layers, id_consistency_loss = self.trm_encoder(
            input_emb, 
            extended_attention_mask, 
            cross_modal_states=cross_modal_fusion,
            output_all_encoded_layers=True
        )
        
        id_output = id_output_all_layers[-1]
        
        self.consistency_losses = {"id": id_consistency_loss}

        self.id_output = id_output
        
        with torch.no_grad():
            img_output_all_layers, img_consistency_loss = self.trm_encoder(
                img_input, 
                extended_attention_mask, 
                cross_modal_states=text_input, 
                output_all_encoded_layers=True
            )
            img_output = img_output_all_layers[-1]
            
            text_output_all_layers, text_consistency_loss = self.trm_encoder(
                text_input, 
                extended_attention_mask, 
                cross_modal_states=img_input,  
                output_all_encoded_layers=True
            )
            text_output = text_output_all_layers[-1]
            
            self.consistency_losses["img"] = img_consistency_loss
            self.consistency_losses["text"] = text_consistency_loss

        short_term_id = self.gather_indexes(id_output, item_seq_len - 1) 
        short_term_img = self.gather_indexes(img_output, item_seq_len - 1)  
        short_term_text = self.gather_indexes(text_output, item_seq_len - 1)  
        
        long_term_id = self.extract_long_term_interest(id_output, attention_mask)  
        long_term_img = self.extract_long_term_interest(img_output, attention_mask) 
        long_term_text = self.extract_long_term_interest(text_output, attention_mask)  
        
        id_gate = self.id_interest_gate(torch.cat([short_term_id, long_term_id], dim=-1)) 
        id_interest = id_gate * short_term_id + (1 - id_gate) * long_term_id
        
        img_gate = self.img_interest_gate(torch.cat([short_term_img, long_term_img], dim=-1))  
        img_interest = img_gate * short_term_img + (1 - img_gate) * long_term_img  

        text_gate = self.text_interest_gate(torch.cat([short_term_text, long_term_text], dim=-1)) 
        text_interest = text_gate * short_term_text + (1 - text_gate) * long_term_text  

        mm_output = torch.cat((self.img_alpha * img_interest, self.text_beta * text_interest, id_interest), dim=-1)

        return mm_output  


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'CE':
            test_item_id_emb = self.item_embedding.weight
            test_item_img_emb = self.img_alpha * self.img_trans(self.img_feat.weight)
            test_item_text_emb = self.text_beta * self.text_trans(self.text_feat.weight)
            test_item_mm_emb = torch.cat((test_item_img_emb, test_item_text_emb, test_item_id_emb), dim=-1)
            
            logits = torch.matmul(seq_output, test_item_mm_emb.transpose(0, 1))
            task_loss = self.loss_fct(logits, pos_items)

            id_consistency_loss = self.consistency_losses.get("id", torch.tensor(0.0, device=task_loss.device))

            order_loss = self.compute_order_aware_loss(self.id_output, item_seq, item_seq_len)

            total_loss = task_loss + self.consistency_loss_weight * id_consistency_loss + self.order_aware_loss_weight * order_loss

            return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        
        test_item_id_emb = self.item_embedding(test_item)
        test_item_img_emb = self.img_alpha * self.img_trans(self.img_feat(test_item))
        test_item_text_emb = self.text_beta * self.text_trans(self.text_feat(test_item))
        test_item_mm_emb = torch.cat((test_item_img_emb, test_item_text_emb, test_item_id_emb), dim=-1)
        
        scores = torch.mul(seq_output, test_item_mm_emb).sum(dim=1)  
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_id_emb = self.item_embedding.weight
        test_item_img_emb = self.img_alpha * self.img_trans(self.img_feat.weight)
        test_item_text_emb = self.text_beta * self.text_trans(self.text_feat.weight)
        test_item_mm_emb = torch.cat((test_item_img_emb, test_item_text_emb, test_item_id_emb), dim=-1)
        
        scores = torch.matmul(seq_output, test_item_mm_emb.transpose(0, 1))  
        return scores