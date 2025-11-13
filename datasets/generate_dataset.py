import random
import os
import json

def generate_math(max_value=100):
    a = random.randint(1, max_value)
    b = random.randint(1, max_value)
    op = random.choice(["+", "-", "*", "/"])

    result = 9
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            result = "undefined"
        else:
            if a % b != 0:
                return {"problem": f"{a}{op}{b}=", "answer": f"{a}/{b}"}
            else:
                result = a // b

    problem = f"{a}{op}{b}="
    answer = str(result)
    
    return {"problem": problem, "answer": answer}

def generate_dataset(num_samples=1000, max_value=100):
    dataset = []
    for _ in range(num_samples):
        dataset.append(generate_math(max_value))
    return dataset

if __name__ == "__main__":
    samples = generate_dataset(1000000, 5)
    file_path = os.path.join(os.path.dirname(__file__), "math_dataset.jsonl")
    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")