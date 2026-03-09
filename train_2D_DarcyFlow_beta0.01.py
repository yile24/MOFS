import torch.fft
import torch.nn.functional as F
import math
import os
import torch
import numpy as np
import torchvision

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import torch.nn as nn
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemoryBuffer(nn.Module):
    def __init__(self, max_size, feature_dim, device, temperature=0.02, quality_threshold=0.6):
        super().__init__()
        self.max_size = max_size
        self.device = device
        self.quality_threshold = quality_threshold
        self.keys = torch.empty((0, feature_dim), device=device)
        self.values = torch.empty((0, feature_dim), device=device)
        self.quality_scores = torch.empty((0,), device=device)  
        self.usage_counts = torch.empty((0,), device=device)    
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.query_transform = nn.Linear(feature_dim, feature_dim)
        
    def add(self, key, value, quality_score=0.8):
        quality_tensor = torch.tensor([quality_score], device=self.device)
        usage_tensor = torch.tensor([0.0], device=self.device)
        self.keys = torch.cat([self.keys, key], dim=0)
        self.values = torch.cat([self.values, value], dim=0)
        self.quality_scores = torch.cat([self.quality_scores, quality_tensor], dim=0)
        self.usage_counts = torch.cat([self.usage_counts, usage_tensor], dim=0)
        if self.keys.size(0) > self.max_size:
            combined_score = self.quality_scores * 0.7 + self.usage_counts * 0.3
            _, keep_indices = torch.topk(combined_score, self.max_size)
            self.keys = self.keys[keep_indices]
            self.values = self.values[keep_indices]
            self.quality_scores = self.quality_scores[keep_indices]
            self.usage_counts = self.usage_counts[keep_indices]
    
    def retrieve(self, query, top_k, diversity_weight=0.1):
        """Enhanced retrieval with diversity consideration"""
        if self.keys.size(0) == 0:
            B, d = query.shape
            dummy = torch.zeros(B, 1, d, device=query.device)
            return dummy, dummy, torch.zeros(B, 1, device=query.device)
        query_transformed = self.query_transform(query)
        k = min(top_k, self.keys.size(0))
        query_norm = F.normalize(query_transformed, dim=-1)
        keys_norm = F.normalize(self.keys, dim=-1)
        sim = torch.matmul(query_norm, keys_norm.T) / self.temperature
        quality_boost = self.quality_scores.unsqueeze(0) * 0.1
        sim = sim + quality_boost
        topk = torch.topk(sim, k=k, dim=-1)
        indices = topk.indices
        scores = F.softmax(topk.values, dim=-1)
        with torch.no_grad():
            for idx_batch in indices:
                self.usage_counts[idx_batch] += 1.0
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, self.keys.size(-1))
        retrieved_keys = torch.gather(
            self.keys.unsqueeze(0).expand(query.size(0), -1, -1),
            1, expanded_indices
        )
        retrieved_values = torch.gather(
            self.values.unsqueeze(0).expand(query.size(0), -1, -1),
            1, expanded_indices
        )
        weighted_keys = torch.sum(retrieved_keys * scores.unsqueeze(-1), dim=1, keepdim=True)
        weighted_values = torch.sum(retrieved_values * scores.unsqueeze(-1), dim=1, keepdim=True)
        return weighted_keys, weighted_values, scores


class ProgressiveContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.03, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, z1, z2, labels, epoch=0, max_epochs=300):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  
        difficulty_factor = min(1.0, epoch / (max_epochs * 0.3))
        neg_mask = 1 - pos_mask
        hard_neg_threshold = 0.7 - 0.3 * difficulty_factor
        exp_sim = torch.exp(sim_matrix)
        pos_sim = exp_sim * pos_mask
        pos_sum = pos_sim.sum(dim=1)
        hard_neg_mask = (sim_matrix > hard_neg_threshold) * neg_mask
        neg_sim = exp_sim * (neg_mask + hard_neg_mask * difficulty_factor)
        neg_sum = neg_sim.sum(dim=1)
        loss = -torch.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))
        
        return loss.mean()


class OurOperatorNet(nn.Module):

    def __init__(self, encoder, decoder, target_size, width=128, prompt_len=4, 
                 soft_prompt_len=4, memory_size=128, device='cuda'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.width = width
        self.prompt_len = prompt_len
        self.soft_prompt_len = soft_prompt_len
        self.memory_bank = MemoryBuffer(
            max_size=memory_size, 
            feature_dim=width, 
            device=device,
            temperature=0.02,
            quality_threshold=0.6
        )

        self.contrastive_loss = ProgressiveContrastiveLoss()
        self.pos_encoding = Learnable2DPosEncoding(target_size[0], target_size[1], width)
        self.vision_encoder = VisionEncoder(dim=width)
        self.fusion_module = MultimodalFusion(dim=width)
        self.concat = Concat(z_dim=width, fno_channels=width)

        self.cross_attention_grad = CrossAttentionOGD(d_model=width, nhead=4, step_size=0.1)
        self.self_attn = nn.TransformerEncoderLayer(d_model=width, nhead=4, dropout=0.1)

        self.adaptive_gates = nn.ModuleDict({
            'visual_fusion': nn.Sequential(nn.Linear(width*2, 1), nn.Sigmoid()),
            'memory_fusion': nn.Sequential(nn.Linear(width*2, 1), nn.Sigmoid()),
            'modality_fusion': nn.Sequential(nn.Linear(width*2, 1), nn.Sigmoid())
        })

        self.operator_embeddings = nn.Embedding(14, width)
        self.quality_predictor = nn.Sequential(
            nn.Linear(width, width//2),
            nn.ReLU(),
            nn.Linear(width//2, 1),
            nn.Sigmoid()
        )
    
    def encode_field_enhanced(self, x):
        B, C, H, W = x.shape

        feat_fno = self.pos_encoding(self.encoder(x))  

        feat_vision = self.vision_encoder(x)

        combined_feat = torch.cat([feat_fno, feat_vision], dim=1) 
        gate_weight = self.adaptive_gates['visual_fusion'](
            combined_feat.permute(0, 2, 3, 1).reshape(B, H*W, -1) 
        ).mean(dim=1, keepdim=True).unsqueeze(-1)

        fused_feat = gate_weight * feat_fno + (1 - gate_weight) * feat_vision  
        fused_feat = fused_feat.permute(0, 2, 3, 1).reshape(B, H * W, self.width) 
        
        return fused_feat
    
    def forward(self, prompt_a, prompt_u, query_a, text_embedding=None, op_id=None, epoch=0):
        B, J, _, H, W = prompt_a.shape

        prompt_a_flat = prompt_a.reshape(B * J, 1, H, W)
        prompt_u_flat = prompt_u.reshape(B * J, 1, H, W)
        
        feat_a = self.encode_field_enhanced(prompt_a_flat)
        feat_u = self.encode_field_enhanced(prompt_u_flat)

        if text_embedding is not None:
            text_embed_expanded = text_embedding.unsqueeze(1).expand(-1, feat_a.shape[1], -1)
            fused_text = self.fusion_module(feat_a, text_embed_expanded)
            feat_a = self.concat(feat_a, fused_text)
            feat_a = self.self_attn(feat_a)

        fused = self.cross_attention_grad(query=feat_a, key_value=feat_u)

        with torch.no_grad():
            query_repr = self.encode_field_enhanced(query_a).mean(dim=1)
            mem_keys, mem_values, mem_scores = self.memory_bank.retrieve(query_repr, top_k=5)

        mem_summary = mem_keys.mean(dim=1).unsqueeze(1)
        mem_summary = mem_summary.repeat_interleave(J, dim=0)
        mem_summary = mem_summary.expand(-1, fused.shape[1], -1)

        memory_gate_input = torch.cat([fused, mem_summary], dim=-1) 
        memory_gate = self.adaptive_gates['memory_fusion'](memory_gate_input) 
        fused = fused + memory_gate * mem_summary  

        query_feat = self.encode_field_enhanced(query_a)

        e_op = self.operator_embeddings(op_id)
        query_feat = query_feat + e_op.unsqueeze(1)

        memory_attn = nn.MultiheadAttention(
            embed_dim=self.width, num_heads=4, batch_first=True
        ).to(query_feat.device)
        
        mem_attn_out, _ = memory_attn(query_feat, mem_values, mem_values)
        
        query_feat = query_feat + mem_attn_out

        query_feat = query_feat.reshape(B, H, W, self.width).permute(0, 3, 1, 2)
        query_feat = self.concat(query_feat, mem_values.squeeze(1))

        query_feat = query_feat.reshape(B, H, W, self.width).permute(0, 3, 1, 2)
        if fused.ndim == 3:
            fused = fused.reshape(B, J, H, W, self.width).permute(0, 1, 4, 2, 3)
        
        out = self.decoder(fused, query_feat, op_id)
        return out, mem_values, mem_scores
    def add_to_memory_with_quality(self, key_repr, value_repr, prediction_error):
        quality_score = math.exp(-prediction_error)  # Higher quality for lower error
        self.memory_bank.add(key_repr.detach(), value_repr.detach(), quality_score)



def compute_multi_task_loss(a, u, pred_a, pred_u, pred_freq, mask_a, mask_u, myloss, alpha=1.0):
    eps = 1e-8

    diff_a = (a - pred_a) * (1 - mask_a)
    denom_a = a * (1 - mask_a)
    loss_patch_a = diff_a.pow(2).sum() / (denom_a.pow(2).sum() + eps)

    diff_u = (u - pred_u) * (1 - mask_u)
    denom_u = u * (1 - mask_u)
    loss_patch_u = diff_u.pow(2).sum() / (denom_u.pow(2).sum() + eps)

    fft_gt = torch.fft.fft2(a.squeeze(1))
    fft_mag_gt = torch.abs(fft_gt).unsqueeze(1)
    fft_pred = torch.fft.fft2(pred_freq.squeeze(1))
    fft_mag_pred = torch.abs(fft_pred).unsqueeze(1)
    loss_freq = myloss(fft_mag_pred, fft_mag_gt)

    return loss_patch_a + loss_patch_u + alpha * loss_freq


def train_multitask_stage1(model, train_dataloader, test_dataloader, optimizer,scheduler, test_operator_files,num_epochs,n_samples_tr,n_samples_te):


    myloss = LpLoss(size_average=False)
    

    best_model_state_encoder=None
    best_model_state=None


    
    for ep in range(num_epochs):
        model.train()
        total_loss = 0

    

        for x in train_dataloader:
            a = x['a'].squeeze(1).to(device)  

            pred_a, pred_u, mask_a, mask_u, pred_freq = model(a, u)

            loss = compute_multi_task_loss(a, u, pred_a, pred_u, pred_freq, mask_a, mask_u, myloss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()


        model.eval()
        num2=0
        test_err = 0.0


        for x in test_dataloader:
            a = x['a'].squeeze(1).to(device)  
            u = x['u'].squeeze(1).to(device)
            batch_size=a.shape[0]

            pred_a, pred_u, mask_a, mask_u, pred_freq = model(a, u)

            loss = compute_multi_task_loss(a, u, pred_a, pred_u, pred_freq, mask_a, mask_u, myloss)

            test_err += loss.item()
            num2+=1

        test_err=test_err/(batch_size*num2)


def train_stage1(epochs,n_samples_tr,n_samples_te,train_dataset,test_dataset,target_size,terminal,patience,rep,J,test_operator_files):

    dataset = FewShotDataset(train_dataset, J=0,target_resolution=target_size)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True)

    test_dataset = FewShotDataset(test_dataset, J=0,target_resolution=target_size)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)



    width=128
    encoder = FNOEncoder(width=width).to(device)

    model = MultiTaskPretrainModel(encoder, patch_size=target_size[0], embed_dim=width, mask_ratio=0.25).to(device)

    learning_rate=1e-3
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-3)


    train_multitask_stage1(model, train_loader, test_loader, optimizer,save_pre_train_path=save_pre_train_path, save_pre_train_path_stage1=save_pre_train_path_stage1,test_operator_files=test_operator_files,num_epochs=epochs,n_samples_tr=n_samples_tr,n_samples_te=n_samples_te,patience=patience)



def train_stage2(epochs, rep, n_samples_tr, n_samples_te, train_dataset, test_dataset, 
                         target_size, J, text_embedding_map, test_operator_files):

    myloss = LpLoss(size_average=False)
    
    dataset = FewShotDataset(train_dataset, J=J, target_resolution=target_size)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
    
    test_dataset_fs = FewShotDataset(test_dataset, J=J, target_resolution=target_size)
    test_loader = DataLoader(test_dataset_fs, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)
    
    myloss = LpLoss(size_average=False)

    
    width = 128
    encoder = FNOEncoder(width=width).to(device)
    decoder = Decoder(width=width, operator_ids=list(range(14)), prompt_len=4, soft_prompt_len=8)
    
    operator_model = OurOperatorNet(
        encoder, decoder, target_size=target_size, width=width
    ).to(device)

    optimizer = torch.optim.AdamW(
        operator_model.parameters(), 
        lr=1e-3, 
    )
    
    for ep in range(epochs):
        operator_model.train()
        
        train_l2 = 0.0
        num_batches = 0
        
        for batch in train_loader:
            a = batch['a'].to(device)
            u = batch['u'].to(device)
            op_id = batch['operator_id'].to(device)
            batch_size1 = a.shape[0]
            
            prompt_a = a[:, :J]
            prompt_u = u[:, :J]
            query_a = a[:, J]
            query_u = u[:, J]
            
            # Text embedding preparation
            op_id_flat = op_id.view(-1)
            text_emb_batch = torch.stack([text_embedding_map[int(op)] for op in op_id_flat], dim=0)
            text_emb_repeated = text_emb_batch.unsqueeze(1).repeat(1, J, 1)
            text_emb = text_emb_repeated.view(-1, text_emb_batch.shape[-1]).to(device)

            
            pred_u, mem_values, mem_scores = operator_model(
                prompt_a, prompt_u, query_a, 
                text_embedding=text_emb, op_id=op_id, epoch=ep
            )
            
            l2 = 0.0
            prediction_errors = []
            
            for i in range(batch_size1):
                a_norm, u_norm = train_dataset.normalizer_map[int(op_id[i])]
                pred_decoded = u_norm.decode(pred_u[i].reshape(1, -1))
                query_decoded = u_norm.decode(query_u[i].reshape(1, -1))
                error = myloss(pred_decoded, query_decoded)
                l2 += error
                prediction_errors.append(error.item())
            
            # Enhanced memory update with quality assessment
            with torch.no_grad():
                z_a = operator_model.encode_field_enhanced(query_a).mean(dim=1)
                z_u = operator_model.encode_field_enhanced(query_u).mean(dim=1)
                
                for i in range(batch_size1):
                    operator_model.add_to_memory_with_quality(
                        z_a[i:i+1], z_u[i:i+1], prediction_errors[i]
                    )
            
            optimizer.zero_grad()
            l2.backward()
            optimizer.step()
            train_l2 += l2.item()
            num_batches += 1
        

        operator_model.eval()
        test_l2 = 0.0
        test_batches = 0


        test_predictions = []
        test_ground_truth = []
        test_data_list = []


        with torch.no_grad():
            for batch in test_loader:
                a = batch['a'].to(device)
                u = batch['u'].to(device)
                op_id = batch['operator_id'].to(device)
                batch_size2 = a.shape[0]
                
                prompt_a = a[:, :J]
                prompt_u = u[:, :J]
                query_a = a[:, J]
                query_u = u[:, J]
                
                op_id_flat = op_id.view(-1)
                text_emb_batch = torch.stack([text_embedding_map[int(op)] for op in op_id_flat], dim=0)
                text_emb_repeated = text_emb_batch.unsqueeze(1).repeat(1, J, 1)
                text_emb = text_emb_repeated.view(-1, text_emb_batch.shape[-1]).to(device)
                
                pred_u, _, _ = operator_model(
                    prompt_a, prompt_u, query_a, 
                    text_embedding=text_emb, op_id=op_id, epoch=ep
                )
                
                pred_decoded_batch = []
                l2 = 0.0
                for i in range(batch_size2):
                    a_norm, u_norm = test_dataset.normalizer_map[int(op_id[i])]
                    pred_decoded = u_norm.decode(pred_u[i].reshape(1, -1))
                    pred_decoded_batch.append(pred_decoded)
                    query_decoded = u_norm.decode(query_u[i].reshape(1, -1))
                    l2 += myloss(pred_decoded, query_decoded)
                
                test_l2 += l2.item()
                test_batches += 1

                pred_decoded_batch = torch.cat(pred_decoded_batch, dim=0)

                test_predictions.append(pred_decoded_batch.cpu().numpy())
                test_ground_truth.append(query_u.cpu().numpy())
                test_data_list.append(a.cpu().numpy())
        
        train_err = train_l2 / (batch_size1 * num_batches)
        test_err = test_l2 / (batch_size2 * test_batches)
        




def train_stage3(epochs, rep, n_samples_tr, n_samples_te, train_dataset, test_dataset, 
                         target_size, J, text_embedding_map, test_operator_files):

    myloss = LpLoss(size_average=False)
    
    dataset = FewShotDataset(train_dataset, J=J, target_resolution=target_size)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
    
    test_dataset_fs = FewShotDataset(test_dataset, J=J, target_resolution=target_size)
    test_loader = DataLoader(test_dataset_fs, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)
    
    width = 128
    encoder = FNOEncoder(width=width).to(device)
    decoder = Decoder(width=width, operator_ids=list(range(14)), prompt_len=4, soft_prompt_len=8)
    
    operator_model = OurOperatorNet(
        encoder, decoder, target_size=target_size, width=width
    ).to(device)

    for param in operator_model.parameters():
        param.requires_grad = True
    
 
    optimizer = torch.optim.AdamW(
        operator_model.parameters(), 
        lr=5e-4 , 
    )
    

    contrastive_loss_fn = ProgressiveContrastiveLoss(temperature=0.02, margin=0.3)

    contrastive_weights = {
        'au': 0.003,    
        'ut': 0.001,    
        'am': 0.002,    
        'diversity': 0.002  
    }


    for ep in range(epochs):
        operator_model.train()
        
        total_l2_tr = 0.0
        total_contrastive = 0.0
        num_batches = 0

        progress = ep / epochs
        contrastive_scale = min(1.0, progress * 2) 
        
        for batch in train_loader:
            a, u = batch['a'].to(device), batch['u'].to(device)
            op_id = batch['operator_id'].to(device)
            batch_size1 = a.shape[0]
            
            prompt_a = a[:, :J]
            prompt_u = u[:, :J]
            query_a = a[:, J]
            query_u = u[:, J]
            
            op_id_flat = op_id.view(-1)
            text_emb_batch = torch.stack([text_embedding_map[int(op)] for op in op_id_flat], dim=0)
            text_emb_repeated = text_emb_batch.unsqueeze(1).repeat(1, J, 1)
            text_emb = text_emb_repeated.view(-1, text_emb_batch.shape[-1]).to(device)
            

            pred_u, mem_values, mem_scores = operator_model(
                prompt_a, prompt_u, query_a, 
                text_embedding=text_emb, op_id=op_id, epoch=ep
            )
            
            l2 = 0.0
            prediction_errors = []
            
            for i in range(batch_size1):
                a_norm, u_norm = train_dataset.normalizer_map[int(op_id[i])]
                pred_decoded = u_norm.decode(pred_u[i].reshape(1, -1))
                query_decoded = u_norm.decode(query_u[i].reshape(1, -1))
                error = myloss(pred_decoded, query_decoded)
                l2 += error
                prediction_errors.append(error.item())
            
            with torch.no_grad():
                z_a = operator_model.encode_field_enhanced(query_a).mean(dim=1)
                z_u = operator_model.encode_field_enhanced(query_u).mean(dim=1)
                
                for i in range(batch_size1):
                    operator_model.add_to_memory_with_quality(
                        z_a[i:i+1], z_u[i:i+1], prediction_errors[i]
                    )

            contrastive_au = contrastive_loss_fn(z_a, z_u, op_id, ep, epochs)
            contrastive_ut = contrastive_loss_fn(z_u, text_emb_batch.to(device), op_id, ep, epochs)
            
            z_memory = mem_values.squeeze(1)
            contrastive_am = contrastive_loss_fn(z_a, z_memory, op_id, ep, epochs)
            
            if operator_model.memory_bank.keys.size(0) > 1:
                memory_keys = operator_model.memory_bank.keys
                memory_sim = torch.matmul(F.normalize(memory_keys), F.normalize(memory_keys).T)
                diversity_loss = torch.mean(torch.triu(memory_sim, diagonal=1) ** 2)
            else:
                diversity_loss = torch.tensor(0.0, device=device)
            
            total_contrastive_loss = (
                contrastive_weights['au'] * contrastive_au +
                contrastive_weights['ut'] * contrastive_ut +
                contrastive_weights['am'] * contrastive_am +
                contrastive_weights['diversity'] * diversity_loss
            ) * contrastive_scale
            
            loss = l2 + total_contrastive_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(operator_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_l2_tr += l2.item()
            total_contrastive += total_contrastive_loss.item()
            num_batches += 1
        

        operator_model.eval()
        total_l2_test = 0.0
        test_batches = 0

        test_predictions = []
        test_ground_truth = []
        test_data_list = []



        with torch.no_grad():
            for batch in test_loader:
                a, u = batch['a'].to(device), batch['u'].to(device)
                op_id = batch['operator_id'].to(device)
                batch_size2 = a.shape[0]
                
                prompt_a = a[:, :J]
                prompt_u = u[:, :J]
                query_a = a[:, J]
                query_u = u[:, J]
                
                op_id_flat = op_id.view(-1)
                text_emb_batch = torch.stack([text_embedding_map[int(op)] for op in op_id_flat], dim=0)
                text_emb_repeated = text_emb_batch.unsqueeze(1).repeat(1, J, 1)
                text_emb = text_emb_repeated.view(-1, text_emb_batch.shape[-1]).to(device)
                
                pred_u, _, _ = operator_model(
                    prompt_a, prompt_u, query_a, 
                    text_embedding=text_emb, op_id=op_id, epoch=ep
                )
                

                pred_decoded_batch = []
                l2 = 0.0
                for i in range(batch_size2):
                    a_norm, u_norm = test_dataset.normalizer_map[int(op_id[i])]
                    pred_decoded = u_norm.decode(pred_u[i].reshape(1, -1))
                    pred_decoded_batch.append(pred_decoded)
                    query_decoded = u_norm.decode(query_u[i].reshape(1, -1))
                    l2 += myloss(pred_decoded, query_decoded)

                pred_decoded_batch = torch.cat(pred_decoded_batch, dim=0)
                test_predictions.append(pred_decoded_batch.cpu().numpy())
                test_ground_truth.append(query_u.cpu().numpy())
                test_data_list.append(a.cpu().numpy())  
               
                total_l2_test += l2.item()
                test_batches += 1
        
        train_err = total_l2_tr / (batch_size1 * num_batches)
        test_err = total_l2_test / (batch_size2 * test_batches)


if __name__ == "__main__":

    n_samples_tr=10
    n_samples_te=10

    J=4

    operator_files = [
        "2D_DarcyFlow_beta100.0_Train.npy",
        "2D_DarcyFlow_beta10.0_Train.npy",
        "2D_DarcyFlow_beta0.1_Train.npy",
        "ns_incom_inhom_2d_512-100.npy",
        "ns_incom_inhom_2d_512-101.npy",
        "ns_incom_inhom_2d_512-1.npy",
        "ns_incom_inhom_2d_512-0.npy",  
        "ns_incom_inhom_2d_512-10.npy",
        "2D_DarcyFlow_beta1.0_Train.npy",
        "2D_DarcyFlow_beta0.01_Train.npy",
        "ns_incom_inhom_2d_512-102.npy",
    ]

    operator_ids = list(range(len(operator_files)))

    test_idx = operator_files.index("2D_DarcyFlow_beta0.01_Train.npy")

    test_operator_files = [operator_files[test_idx]]
    test_operator_ids = [operator_ids[test_idx]]

    train_operator_files = [f for i, f in enumerate(operator_files) if i != test_idx]
    train_operator_ids = [i for i in operator_ids if i != test_idx]

    target_size = (128, 128)

    operator_data_tr = preload_operator_data(data_dir, train_operator_files, train_operator_ids, target_size, n_samples_tr)
    operator_data_te = preload_operator_data(data_dir, test_operator_files, test_operator_ids, target_size, n_samples_te)

    train_dataset = MultiOperatorDataset(operator_data_tr, train_operator_ids)
    test_dataset = MultiOperatorDataset(operator_data_te, test_operator_ids)

    
    train_stage1(epochs=100, n_samples_tr=n_samples_tr, n_samples_te=n_samples_te, 
                   train_dataset=train_dataset, test_dataset=test_dataset, 
                   target_size=target_size, 
                   rep=rep, J=J, test_operator_files=test_operator_files)

    train_stage2(epochs=100, rep=rep, n_samples_tr=n_samples_tr, n_samples_te=n_samples_te, 
                         train_dataset=train_dataset, test_dataset=test_dataset, 
                         target_size=target_size, J=J, text_embedding_map=text_embedding_map, 
                         test_operator_files=test_operator_files)

    train_stage3(epochs=100, rep=rep, n_samples_tr=n_samples_tr, n_samples_te=n_samples_te, 
                         train_dataset=train_dataset, test_dataset=test_dataset, 
                         target_size=target_size, J=J, text_embedding_map=text_embedding_map, 
                         test_operator_files=test_operator_files) 
