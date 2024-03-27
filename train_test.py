
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from otfrm import OTLoss


class Trainer(object):
    def __init__(self, model, tokenizer, lr, weight_decay, batch_size, gradient_accumulation,
                 return_emb=False, freeze_seq_encoder=False, guide=False, pt_embeddings=None):
        self.model = model
        self.tokenizer = tokenizer
        
        # 冻结预训练序列编码器
        if freeze_seq_encoder:
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)

        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.return_emb = return_emb

        self.guide = guide
        p, entreg, cost_metric = 2, .1, 'cosine'
        self.otloss = OTLoss(loss='sinkhorn', p=p, entreg=entreg, cost_metric=cost_metric)
        self.pt_embeddings = pt_embeddings
        self.alpha = .1

    def train(self, dataset, device, task):
        self.model.train()
        datasampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.batch_size, shuffle=False)
        if task == 'PDBBind':
            Loss = nn.MSELoss()
        elif task in ['Kinase', 'DUDE']:
            Loss = nn.BCELoss()
        loss_total = 0
        self.optimizer.zero_grad()
        current_count = 0
        all_count = len(dataloader)
        spent_time_accumulation = 0
        all_predict_labels, all_real_labels = [], []
        for step, batch in enumerate(dataloader):
            start_time_batch = time.time()
            labels, pro_seqs, pro_ids, (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask) = batch
            drug_data = x_atom.float(), x_bonds.float(), x_atom_index.long(), x_bond_index.long(), x_mask.float() # change data type
            drug_data = [t.to(device) for t in drug_data] # move
            drug_data = [t.reshape(-1, *t.shape[2:]) for t in drug_data] # reshape: (batch_size, 1) -> batch_size (merge first two dims togerther)
            labels = labels.float().to(device)

            # 获取序列长度
            proteins_num = torch.tensor([len(pro_seq) for pro_seq in pro_seqs], dtype=torch.long, device=device)
            max_protein_len_batch = torch.max(proteins_num)
            # 构建序列输入特征
            # 序列padding
            seq_tokens_pad = []
            for pro_id, seq_token in zip(pro_ids, pro_seqs):
                seq_tokens_pad.append((pro_id, list(seq_token) + ['<pad>' for _ in range(max_protein_len_batch - len(seq_token))]))
            # 序列tokenization
            batch_labels, batch_strs, seq_feat = self.tokenizer(seq_tokens_pad)
            seq_feat = seq_feat[:, : , 1:]
            seq_feat = seq_feat.to(device)
            protein_data = (pro_ids, seq_feat, proteins_num)

            if self.guide:
                predict_labels, pro_embs = self.model(protein_data, drug_data)
                # OT loss
                pt_bs = len(self.pt_embeddings) // len(dataloader)
                pt_embs_patch = self.pt_embeddings[step*pt_bs: (step+1)*pt_bs].to(device)
                ot_loss = 1 / self.otloss.cotfrm(pt_embs_patch, pro_embs)
            elif self.return_emb:
                predict_labels, _ = self.model(protein_data, drug_data)
            else:
                predict_labels = self.model(protein_data, drug_data)

            if task == 'PDBBind':
                predict_labels = predict_labels.squeeze(1)
            elif task in ['Kinase', 'DUDE']:
                predict_labels = F.softmax(predict_labels, dim=1)
                predict_labels = predict_labels[:, 1]
                
            loss = Loss(predict_labels, labels)  # mark

            if self.guide:
                loss = loss * (1-self.alpha) + ot_loss * self.alpha

            loss_total += loss.item()  # mark
            loss /= self.gradient_accumulation  # mark
            loss.backward()

            all_predict_labels += predict_labels.detach().cpu().numpy().tolist()
            all_real_labels += labels.detach().cpu().numpy().tolist()

            if (step+1) % self.gradient_accumulation == 0 or (step+1) == len(dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            end_time_batch = time.time()
            seconds = end_time_batch-start_time_batch
            spent_time_accumulation += seconds
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_batch = "%02d:%02d:%02d" % (h, m, s)
            m, s = divmod(spent_time_accumulation, 60)
            h, m = divmod(m, 60)
            have_spent_time = "%02d:%02d:%02d" % (h, m, s)

            current_count += 1
            if current_count == all_count:
                print("Finish batch: %d/%d---batch time: %s, have spent time: %s" % (current_count, all_count, spend_time_batch, have_spent_time))
            else:
                print("Finish batch: %d/%d---batch time: %s, have spent time: %s" % (current_count, all_count, spend_time_batch, have_spent_time), end='\r')

        return loss_total/(step+1), all_predict_labels, all_real_labels

class Tester(object):
    def __init__(self, model, tokenizer, batch_size, return_emb=False, training=True, guide=False, pt_embeddings=None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.return_emb = return_emb
        self.test_return_emb = return_emb & (not training)

        self.guide = guide
        p, entreg, cost_metric = 2, .1, 'cosine'
        self.otloss = OTLoss(loss='sinkhorn', p=p, entreg=entreg, cost_metric=cost_metric)
        self.pt_embeddings = pt_embeddings
        self.alpha = .1

    def test(self, dataset, device, task):
        self.model.eval()
        datasampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.batch_size, shuffle=False)
        if task == 'PDBBind':
            Loss = nn.MSELoss()
        elif task in ['Kinase', 'DUDE']:
            Loss = nn.BCELoss()
        loss_total = 0
        all_predict_labels, all_real_labels = [], []
        if self.test_return_emb:
            all_pro_ids, all_pro_seqs, all_pro_embs = [], [], []  # mark

        for step, batch in enumerate(dataloader):
            labels, pro_seqs, pro_ids, (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask) = batch
            drug_data = x_atom.float(), x_bonds.float(), x_atom_index.long(), x_bond_index.long(), x_mask.float() # change data type
            drug_data = [t.to(device) for t in drug_data] # move
            drug_data = [t.reshape(-1, *t.shape[2:]) for t in drug_data] # reshape: (batch_size, 1) -> batch_size (merge first two dims togerther)
            labels = labels.float().to(device)
            if self.test_return_emb:
                all_pro_ids += list(pro_ids)
                all_pro_seqs += list(pro_seqs)

            # 获取序列长度
            proteins_num = torch.tensor([len(pro_seq) for pro_seq in pro_seqs], dtype=torch.long, device=device)
            max_protein_len_batch = torch.max(proteins_num)
            # 构建序列输入特征
            # 序列padding
            seq_tokens_pad = []
            for pro_id, seq_token in zip(pro_ids, pro_seqs):
                seq_tokens_pad.append((pro_id, list(seq_token) + ['<pad>' for _ in range(max_protein_len_batch - len(seq_token))]))
            # 序列tokenization
            batch_labels, batch_strs, seq_feat = self.tokenizer(seq_tokens_pad)
            seq_feat = seq_feat[:, : , 1:]
            seq_feat = seq_feat.to(device)
            protein_data = (pro_ids, seq_feat, proteins_num)

            with torch.no_grad():
                if self.guide:
                    predict_labels, pro_embs = self.model(protein_data, drug_data)
                    # OT loss
                    pt_bs = len(self.pt_embeddings) // len(dataloader)
                    pt_embs_patch = self.pt_embeddings[step*pt_bs: (step+1)*pt_bs].to(device)
                    ot_loss = 1 / self.otloss.cotfrm(pt_embs_patch, pro_embs)
                    # ot_loss = self.otloss.cpwdist(pt_embs_patch, pro_embs)
                elif self.return_emb:
                    predict_labels, pro_embs = self.model(protein_data, drug_data)
                else:
                    predict_labels = self.model(protein_data, drug_data)
            if self.test_return_emb:
                all_pro_embs.append(pro_embs.cpu().numpy())
            
            if task == 'PDBBind':
                predict_labels = predict_labels.squeeze(1)
            elif task in ['Kinase', 'DUDE']:
                predict_labels = F.softmax(predict_labels, dim=1)
                predict_labels = predict_labels[:, 1]

            loss = Loss(predict_labels, labels)
            if self.guide:
                loss = loss * (1-self.alpha) + ot_loss * self.alpha

            all_predict_labels += predict_labels.detach().cpu().numpy().tolist()
            all_real_labels += labels.detach().cpu().numpy().tolist()
            loss_total += loss.item()

        if self.test_return_emb:
            all_pro_embs = np.concatenate(all_pro_embs, 0).tolist()
            return loss_total/(step+1), all_predict_labels, all_real_labels, all_pro_ids, all_pro_seqs, all_pro_embs
        else:
            return loss_total/(step+1), all_predict_labels, all_real_labels
        

    def save_model(self, model, filename):
        # model_to_save = model
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), filename)
