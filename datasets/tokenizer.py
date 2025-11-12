from typing import Dict, List

class Tokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]

        self.operators_tokens = ["+", "-", "*", "/", "="]

        self.digits_tokens = [str(i) for i in range(10)]

        self.vocab = (
            self.special_tokens + self.operators_tokens + self.digits_tokens
        )

        self.token2id: Dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token: Dict[int, str] = {idx: token for token, idx in self.token2id.items()}
        
        self.pad_id = self.token2id[self.pad_token]
        self.sos_id = self.token2id[self.sos_token]
        self.eos_id = self.token2id[self.eos_token]
        self.unk_id = self.token2id[self.unk_token]

    def tokenize(self, text: str) -> List[str]:
        text = text.replace(" ", "")
        
        tokens = []
        for char in text:
            if char in self.token2id:
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        
        token_ids = [self.token2id.get(token, self.unk_id) for token in tokens]
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = [self.id2token.get(idx, self.unk_token) for idx in token_ids]
        
        if skip_special_tokens:
            tokens = [
                t for t in tokens 
                if t not in [self.pad_token, self.sos_token, self.eos_token]
            ]
        
        return "".join(tokens)
    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    sample_text = "12 + 34 = 46"
    tokens = tokenizer.tokenize(sample_text)
    print("Text:", sample_text)
    print("Tokens:", tokens)
    token_ids = tokenizer.encode(sample_text)
    print("Token IDs:", token_ids)
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded Text:", decoded_text)