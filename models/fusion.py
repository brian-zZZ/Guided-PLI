import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]

        self.scale = self.scale.to(query.device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, sent len_Q, sent len_K]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.drop(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]

        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]

        x = self.drop(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        # tgt = [batch_size, seq len, atom_dim]
        # tgt_mask = [batch size, seq len]

        tgt = self.ln(tgt + self.drop(self.sa(tgt, tgt, tgt, tgt_mask)))
        tgt = self.ln(tgt + self.drop(self.pf(tgt)))

        return tgt


class SeqStrucFusion(nn.Module):
    def __init__(self, n_layers, n_heads, pf_dim, seq_hid_dim, struc_hid_dim, seq_struc_hid_dim, dropout):
        super().__init__()
        self.multimodal_transform = nn.Linear(seq_hid_dim+struc_hid_dim, seq_struc_hid_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(seq_struc_hid_dim, n_heads, pf_dim, dropout)
             for _ in range(n_layers)])
        
        self.ln = nn.LayerNorm(seq_struc_hid_dim)

    def forward(self, seq_feat, struc_feat, protein_mask):
        multimodal_emb = torch.cat((seq_feat, struc_feat), dim=-1)
        multimodal_emb = self.multimodal_transform(multimodal_emb)
        multimodal_emb = F.relu(multimodal_emb)
        
        for layer in self.layers:
            multimodal_emb = layer(multimodal_emb, protein_mask)
        
        multimodal_emb = self.ln(multimodal_emb)
        return multimodal_emb


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_heads, hid_dim, pf_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        # tgt = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # tgt_mask = [batch size, 1, compound sent len, 1]
        # src_mask = [batch size, 1, 1, protein len]

        tgt = self.ln(tgt + self.drop(self.sa(tgt, tgt, tgt, tgt_mask)))
        tgt = self.ln(tgt + self.drop(self.ea(tgt, src, src, src_mask)))
        tgt = self.ln(tgt + self.drop(self.pf(tgt)))

        return tgt


class ProDrugCrossFusion(nn.Module):
    def __init__(self, n_layers, n_heads, pro_hid_dim, drug_hid_dim, dropout=.1, ffn_hid_dim=None):
        super().__init__()
        ffn_hid_dim = (pro_hid_dim * 4) if ffn_hid_dim is None else ffn_hid_dim
        self.pro_layers = nn.ModuleList(
            [TransformerDecoderLayer(n_heads, pro_hid_dim, ffn_hid_dim, dropout)
             for _ in range(n_layers)])
        self.drug2pro = nn.Linear(drug_hid_dim, pro_hid_dim)

    def forward(self, pro_feat, drug_feat, pro_mask=None, drug_mask=None):
        drug_feat_for_pro = self.drug2pro(drug_feat)
        for layer in self.pro_layers:
            decoded_pro_feat = layer(pro_feat, drug_feat_for_pro, pro_mask, drug_mask)

        return decoded_pro_feat