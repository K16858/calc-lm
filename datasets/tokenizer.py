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