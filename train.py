import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent / "model"))
sys.path.append(str(Path(__file__).parent / "datasets"))

from model.tokenizer import Tokenizer
from model.transformer import Transformer, count_parameters


class MathDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        max_length: int = 32
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.equations = [line.strip() for line in f if line.strip()]
        
        print(f"sample: {len(self.equations)}")
    
    def __len__(self):
        return len(self.equations)
    
    def __getitem__(self, idx):
        equation = self.equations[idx]
        
        token_ids = self.tokenizer.encode(equation, add_special_tokens=True)
        
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            pad_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_id] * pad_length
        
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return input_ids, target_ids


def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    
    input_ids = torch.stack(input_ids)
    target_ids = torch.stack(target_ids)
    
    return input_ids, target_ids


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 損失関数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # オプティマイザー
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # スケジューラー
        total_steps = len(train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # 順伝播
            logits = self.model(input_ids)
            
            # 損失計算
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            # 逆伝播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # プログレスバーに表示
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc="Validation"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "history": self.history
        }
        
        save_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists() or val_loss < min(self.history["val_loss"] or [float('inf')]):
            torch.save(checkpoint, best_path)
            print(f"best model: epoch {epoch}, val_loss {val_loss:.4f}")
    
    def train(self):
        print(f"device: {self.device}")
        print(f"num_epochs: {self.num_epochs}")
        print(f"batch_size: {self.train_loader.batch_size}")
        print(f"learning_rate: {self.optimizer.param_groups[0]['lr']:.2e}\n")

        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # 学習
            train_loss = self.train_epoch()
            
            # バリデーション
            val_loss = self.validate()
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            if self.val_loader:
                print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            

            self.save_checkpoint(epoch, val_loss)

        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\ntraining complete")


def main():
    # ハイパーパラメータ
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 20
    MAX_LENGTH = 32
    
    # モデルパラメータ
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer = Tokenizer()

    data_path = Path(__file__).parent / "datasets" / "math_dataset.txt"
    
    
    dataset = MathDataset(data_path, tokenizer, max_length=MAX_LENGTH)
    
    # Train/Val分割 (90% / 10%)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # データローダー
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    vocab_size = tokenizer.get_vocab_size()
    
    # モデル作成
    model = Transformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_LENGTH,
        pad_idx=tokenizer.pad_id
    )
    
    num_params = count_parameters(model)
    print(f"params: {num_params:,} ({num_params/1e6:.2f}M)")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 学習実行
    trainer.train()


if __name__ == "__main__":
    main()