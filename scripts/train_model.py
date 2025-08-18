import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer
from tqdm import tqdm
from eeg_cnn_decoder import EEG2Text

class EEGDataset(Dataset):
    def __init__(self, npz_dir, sentences_csv, max_len=128):
        self.files = sorted(glob.glob(f"{npz_dir}/*.npz"))
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        import pandas as pd
        df = pd.read_csv(sentences_csv, header=None, names=['sentence'])
        self.sentences = df['sentence'].tolist()
        self.max_len = max_len

    def __len__(self):
        return sum(np.load(f)['eeg'].shape[0] for f in self.files)

    def __getitem__(self, idx):
        cum = 0
        for f in self.files:
            data = np.load(f)
            n = data['eeg'].shape[0]
            if idx < cum + n:
                eeg = data['eeg'][idx - cum]
                lbl = int(data['labels'][idx - cum])
                text = self.sentences[lbl]
                enc = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len
                )
                return torch.from_numpy(eeg), enc.input_ids.squeeze(0)
            cum += n
        raise IndexError


def collate_fn(batch):
    eegs, ids = zip(*batch)
    return torch.stack(eegs), torch.stack(ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True)
    parser.add_argument('--sentences_csv', required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    ds = EEGDataset(args.npz_dir, args.sentences_csv)
    # Split into train/validation
    n_total = len(ds)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = EEG2Text(
        vocab_size=tokenizer.vocab_size,
        max_len=ds.max_len,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.epochs}", unit="batch")
        for eeg, ids in pbar:
            eeg = eeg.float().to(device)
            ids = ids.to(device)
            logits = model(eeg, ids[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} training avg loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg, ids in val_loader:
                eeg = eeg.float().to(device)
                ids = ids.to(device)
                logits = model(eeg, ids[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}")

        # Early stopping & checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            ckpt_path = f"models/best_eeg2text_epoch{epoch:02d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best model saved to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping early.")
                break