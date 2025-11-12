import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # トークン埋め込み層
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer Decoder層
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 出力層（語彙への射影）
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # パラメータ初期化
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.pad_idx)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = src.size()
        
        # 因果マスクを生成
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # パディングマスクを生成
        src_key_padding_mask = self.create_padding_mask(src)
        
        # 埋め込み + スケーリング
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # 位置エンコーディングを追加
        x = self.pos_encoding(x)
        
        # Transformer Decoder
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=src_mask,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # 出力層
        logits = self.fc_out(output)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        eos_token_id: int = 2
    ) -> torch.Tensor:
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 現在の系列から次のトークンを予測
                logits = self.forward(input_ids)
                
                # 最後のトークンのロジットを取得
                next_token_logits = logits[:, -1, :] / temperature
                
                # 最も確率の高いトークンを選択
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 系列に追加
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # EOSトークンが出たら終了
                if (next_token == eos_token_id).all():
                    break
        
        return input_ids
    
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    vocab_size = 20 
    model = Transformer(vocab_size=vocab_size)
    print(f"Parameters: {count_parameters(model)}")