import re
from pathlib import Path

from lark import Lark
from lark.exceptions import LarkError
from Levenshtein import distance as levenshtein


PATH = Path(__file__).parent.resolve()
ALLOWED_LOGIC_OPS = ["∧", "∨", "⊕", "→", "↔"]
UNARY_OPS = ["¬"]
COMP_OPS = ["<", ">", "<=", ">=", "="]

PRED_DEFAULT_THRESHOLD = 0.8
OP_DEFAULT_THRESHOLD = 0.9

parser = Lark.open(
    str(PATH / "fol_grammar.lark"),
    parser="lalr",
    start="formula",
    propagate_positions=True,
    maybe_placeholders=False
)


from lark import Transformer

class FOLTransformer(Transformer):
    @staticmethod
    def variable(items):
        return str(items[0])

    @staticmethod
    def constant(items):
        return str(items[0])

    @staticmethod
    def NUM_COMP(items):
        token = str(items[0])
        op = token[0]
        value = int(items[1:])
        return {"op": op, "value": value}

    @staticmethod
    def predicate_application(items):
        pred_name = str(items[0])
        args = items[1:] if len(items) > 1 else []
        return {"predicate": pred_name, "args": args}

    @staticmethod
    def quantified_formula(items):
        quantifier_tree = items[0]
        formula = items[1]

        quantifiers = []
        variables = []

        for child in quantifier_tree.children:
            if hasattr(child, 'data') and child.data == 'variable':
                variables.append(str(child.children[0]))
            else:
                val = str(child)
                if val in ("∀", "∃"):
                    quantifiers.append(val)
                else:
                    variables.append(val)

        return {"quantifiers": quantifiers, "variables": variables, "formula": formula}

    @staticmethod
    def binary_formula(items):
        return {"op": str(items[1]), "left": items[0], "right": items[2]}

    @staticmethod
    def unary_formula(items):
        return {"op": str(items[0]), "arg": items[1]}

    @staticmethod
    def atom(items):
        return items[0]


def is_valid_fol(formula: str) -> bool:
    """Check if a FOL formula is syntactically valid."""
    if not isinstance(formula, str):
        return False
    try:
        parser.parse(formula)
        return True
    except LarkError:
        return False


def parse_fol(formula: str):
    """Parse a FOL formula and return the AST."""
    tree = parser.parse(formula)
    transformer = FOLTransformer()
    return transformer.transform(tree)

def tokenize(formula: str) -> list[str]:
    f = formula.replace(" ", "")
    for s in ALLOWED_LOGIC_OPS + UNARY_OPS + COMP_OPS + ["(", ")", ","]:
        f = f.replace(s, f" {s} ")
    tokens = [t for t in f.split(" ") if t]
    return tokens

def parentheses_signature(tokens: list[str]) -> list[int]:
    sig = []
    depth = 0
    for t in tokens:
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
        sig.append(depth)
    return sig

def extract_predicates(tokens: list[str]) -> list[str]:
    predicates = []
    for i, t in enumerate(tokens):
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", t):
            if i + 1 < len(tokens) and tokens[i + 1] == "(":
                predicates.append(t)
    return predicates

def normalize_variables(tokens: list[str]) -> list[str]:
    mapping = {}
    next_name = ord("a")
    result = []
    for t in tokens:
        if re.fullmatch(r"[a-z]", t):
            if t not in mapping:
                mapping[t] = chr(next_name)
                next_name += 1
            result.append(mapping[t])
        else:
            result.append(t)
    return result

def predicate_similarity_score(a: list[str], b: list[str]) -> float:
    matches = 0
    used = set()
    for pa in a:
        best = 999
        best_b = None
        for pb in b:
            if pb in used:
                continue
            d = levenshtein(pa.lower(), pb.lower())
            if d < best:
                best = d
                best_b = pb
        max_len = max(len(pa), len(best_b)) if best_b else 0
        if best_b and max_len > 0 and (1 - best / max_len) >= PRED_DEFAULT_THRESHOLD:
            matches += 1
            used.add(best_b)
    if max(len(a), len(b)) == 0:
        return 1.0
    return matches / max(len(a), len(b))

def compare_fol(
        formula_a: str,
        formula_b: str,
        pred_threshold: float = PRED_DEFAULT_THRESHOLD,
        op_threshold: float = OP_DEFAULT_THRESHOLD
) -> bool:

    if not isinstance(formula_a, str) or not isinstance(formula_b, str):
        return False

    # Tokenize
    a = tokenize(formula_a)
    b = tokenize(formula_b)

    # Parentheses structure
    paren_similar = parentheses_signature(a) == parentheses_signature(b)

    # Normalize variables
    a_norm = normalize_variables(a)
    b_norm = normalize_variables(b)

    # Predicate similarity
    pred_score = predicate_similarity_score(
        extract_predicates(a_norm),
        extract_predicates(b_norm)
    )

    # Operator similarity
    ops_a = [t for t in a_norm if t in ALLOWED_LOGIC_OPS + UNARY_OPS + COMP_OPS]
    ops_b = [t for t in b_norm if t in ALLOWED_LOGIC_OPS + UNARY_OPS + COMP_OPS]
    op_score = len(set(ops_a).intersection(ops_b)) / max(1, len(set(ops_a)))

    return paren_similar and pred_score >= pred_threshold and op_score >= op_threshold


if __name__ == "__main__":
    f1 = "∀x (HealthyLifestyle(x) ↔ RegularExercise(x) ∧ BalancedDiet(x) ∧ AdequateSleep(x))"
    f2 = "∀z (HealthyLifestyle(z) ↔ BalancedDiet(z) ∧ AdequateSleep(z) ∧ RegularExercise(z))"
    print("Compare FOL:", compare_fol(f1, f2))

    # Parsing example
    f3 = "∀x (Dish(x) ∧ LiquidFood(x) ∧ UsuallyMadeByBoiling(x, vegetables, meat, fish, water, stock) → Soup(x))"
    print("Parse FOL:", parse_fol(f3))
