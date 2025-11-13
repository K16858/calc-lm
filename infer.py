import torch
from pathlib import Path
from model.tokenizer import Tokenizer
from model.transformer import Transformer

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
MAX_LENGTH = 32

def load_model(checkpoint_path: str, device: str = "cpu"):
    tokenizer = Tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_LENGTH,
        pad_idx=tokenizer.pad_id
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def generate_answer(model, tokenizer, question: str, max_length: int = 32, device: str = "cpu"):
    model.eval()
    
    tokens = tokenizer.encode(question, add_special_tokens=True)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            output = model(input_ids)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            if next_token.item() == tokenizer.eos_id:
                break
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    result = tokenizer.decode(input_ids[0].tolist())
    return result

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
    model, tokenizer = load_model(str(checkpoint_path), device)
    
    print("\nInput :")
    while True:
        question = input("> ")
        if question.lower() == "quit":
            break
            
        answer = generate_answer(model, tokenizer, question, device=device)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()