import math
import torch
import torch.nn as nn


# 1D CNN encoder: downsamples EEG time-series (batch,16,1250) to (seq_len, batch, d_model).
# Four convolutional blocks halve the time dimension three times (~156 frames) and project to d_model.

class CNNEncoder(nn.Module):

    def __init__(self, in_ch=16, d_model=512):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            # Block 1: in_ch -> hidden, stride 2
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2, stride=2),
            nn.GELU(), nn.BatchNorm1d(hidden),
            # Block 2: hidden -> hidden*2, stride 2
            nn.Conv1d(hidden, hidden*2, kernel_size=5, padding=2, stride=2),
            nn.GELU(), nn.BatchNorm1d(hidden*2),
            # Block 3: hidden*2 -> hidden*4, stride 2
            nn.Conv1d(hidden*2, hidden*4, kernel_size=5, padding=2, stride=2),
            nn.GELU(), nn.BatchNorm1d(hidden*4),
            # Final projection to d_model (no stride)
            nn.Conv1d(hidden*4, d_model, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):  # x: (batch, 16, 1250)
        z = self.net(x)  # (batch, d_model, seq_len)
        z = z.permute(2, 0, 1)  # (seq_len, batch, d_model)
        return z

# CNNEncoder → Transformer Decoder-only → Language Model head.

class EEG2Text(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 max_len: int = 128,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 pad_token_id: int = 0):
        super().__init__()
        # Encoder
        self.encoder = CNNEncoder(in_ch=16, d_model=d_model)
        # Learned positional embeddings for encoder memory
        mem_len = math.ceil(1250 / (2 ** 3))  # three halvings: 1250→625→312→156
        self.mem_pos_emb = nn.Embedding(mem_len, d_model)
        # Positional embeddings for decoder inputs
        self.pos_emb = nn.Embedding(max_len, d_model)
        # Token embeddings and LM head
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        # Store config
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask

 # Training pass:
        # - eeg: (batch,16,1250)
        # - target_ids: (batch, tgt_len)
        # Returns:
        # - logits: (batch, tgt_len, vocab_size)

    def forward(self, eeg, target_ids):
        
        # Encode EEG
        memory = self.encoder(eeg)  # (mem_len, batch, d_model)
        # Add positional embeddings to memory
        seq_len, batch_size, _ = memory.size()
        pos_ids = torch.arange(seq_len, device=memory.device)
        memory = memory + self.mem_pos_emb(pos_ids).unsqueeze(1)
        # Prepare decoder input embeddings
        b, tgt_len = target_ids.size()
        positions = torch.arange(tgt_len, device=target_ids.device).unsqueeze(0)
        tgt = self.token_emb(target_ids) * math.sqrt(self.d_model)
        tgt = tgt + self.pos_emb(positions)
        tgt = tgt.permute(1, 0, 2)  # (tgt_len, batch, d_model)
        # Causal mask
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(eeg.device)
        # Decode
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = self.lm_head(out)  # (tgt_len, batch, vocab_size)
        return logits.permute(1, 0, 2)  # (batch, tgt_len, vocab_size)

    # Greedy decoding:
        # - eeg: (1,16,1250)
        # Returns list of token IDs.
    
    @torch.no_grad()
    def generate(self, eeg, tokenizer, max_new_tokens: int = 128):
      
        self.eval()
        # Encode + positional
        memory = self.encoder(eeg)
        seq_len, batch_size, _ = memory.size()
        pos_ids = torch.arange(seq_len, device=memory.device)
        memory = memory + self.mem_pos_emb(pos_ids).unsqueeze(1)
        # Start token
        ys = torch.full(
            (1, 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
            device=memory.device
        )
        # Autoregressive decode
        for _ in range(max_new_tokens):
            tgt_len = ys.size(1)
            positions = torch.arange(tgt_len, device=memory.device).unsqueeze(0)
            tgt = self.token_emb(ys) * math.sqrt(self.d_model)
            tgt = tgt + self.pos_emb(positions)
            tgt = tgt.permute(1, 0, 2)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(memory.device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.lm_head(out[-1])  # (batch, vocab)
            next_id = logits.argmax(-1).unsqueeze(0)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        return ys.squeeze(0).tolist()
