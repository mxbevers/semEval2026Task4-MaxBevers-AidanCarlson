import json
import re
from sympy import symbols, Eq, solve

x = symbols('x')

def sentence_to_equation(sentence):
    """Convert a word problem into a sympy equation."""
    s = sentence.lower().strip(".")

    # Base replacements → turn words into math
    replacements = [
        (r"the sum of a number and (\d+)", rf"x + \1"),
        (r"a number increased by (\d+)", rf"x + \1"),
        (r"(\d+) added to a number", rf"x + \1"),
        (r"a number decreased by (\d+)", rf"x - \1"),
        (r"a number minus (\d+)", rf"x - \1"),
        (r"(\d+) less than a number", rf"x - \1"),
        (r"(\d+) more than a number", rf"x + \1"),
        (r"a number plus (\d+)", rf"x + \1"),
        (r"a number divided by (\d+)", rf"x / \1"),
        (r"the quotient of a number and (\d+)", rf"x / \1"),
        (r"the difference between a number and (\d+)", rf"x - \1"),
        (r"the product of a number and (\d+)", rf"x * \1"),
        (r"(\d+) times a number", rf"\1 * x"),
        (r"twice a number", rf"2*x"),
        (r"triple a number", rf"3*x"),
        (r"half of a number", rf"x/2"),
        (r"a number squared", rf"x**2"),
        (r"the square of a number", rf"x**2"),
        (r"a number", rf"x"),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)

    # Handle equals phrases
    s = s.replace("equals", "=").replace("is", "=")

    # Split into LHS and RHS
    if "=" in s:
        lhs, rhs = s.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()
    else:
        lhs, rhs = s, "0"

    try:
        # Use sympy to safely parse
        equation = Eq(eval(lhs), eval(rhs))
    except Exception:
        equation = None

    return equation


def main():
    with open("src/data/word_problems.json", "r") as f:
        problems = json.load(f)

    for p in problems:
        text = p["Problem"]
        eq = sentence_to_equation(text)

        if eq is not None:
            solutions = solve(eq, x)
            print(f"Problem: {text}")
            print(f"Equation: {eq}")
            print(f"Solution: {solutions}\n")
        else:
            print(f"Problem: {text}")
            print("❌ Could not parse this problem.\n")


if __name__ == "__main__":
    main()
