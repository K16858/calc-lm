import random
import os

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
                return f"{a} {op} {b} = {a}/{b}"
            else:
                result = a // b

    return f"{a} {op} {b} = {result}"

def generate_dataset(num_samples=1000, max_value=100):
    dataset = []
    for _ in range(num_samples):
        dataset.append(generate_math(max_value))
    return dataset

if __name__ == "__main__":
    samples = generate_dataset(10, 50)
    file_path = os.path.join(os.path.dirname(__file__), "math_dataset.txt")
    open(file_path, "w").write("\n".join(samples))