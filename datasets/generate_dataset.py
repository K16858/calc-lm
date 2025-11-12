import random

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


if __name__ == "__main__":
    for _ in range(10):
        print(generate_math(50))