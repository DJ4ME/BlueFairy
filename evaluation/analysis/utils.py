import re
from pathlib import Path
from lark import Lark, Transformer, Tree, Token
from lark.exceptions import LarkError
from Levenshtein import distance as levenshtein

PATH = Path(__file__).parent.resolve()

ALLOWED_LOGIC_OPS = ["∧", "∨", "⊕", "→", "↔"]
UNARY_OPS = ["¬"]

PRED_DEFAULT_THRESHOLD = 0.8
OP_DEFAULT_THRESHOLD = 0.9

parser = Lark.open(
    str(PATH / "fol_grammar.lark"),
    parser="earley",
    propagate_positions=True,
    maybe_placeholders=False
)

class ASTNode:
    def __init__(self, type_, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _tree_lines(self, prefix="", is_last=True):
        name = self.type
        if self.type == "BINOP":
            name += f" ({self.op})"
        elif self.type == "NOT":
            name += ""
        elif self.type == "PRED":
            args_str_list = []
            for a in self.args:
                if isinstance(a, ASTNode) and a.type in ("VAR", "CONST"):
                    args_str_list.append(f"{a.name} ({a.type})")
                elif isinstance(a, ASTNode):
                    args_str_list.append(a.name if hasattr(a, "name") else str(a))
                else:
                    args_str_list.append(str(a))
            args_str = ", ".join(args_str_list)
            name += f" ({self.predicate}) [{args_str}]"
        elif self.type in ("VAR", "CONST"):
            name += f" {self.name}"
        elif self.type == "QUANTIFIED":
            vars_str_list = []
            for v in self.variables:
                if isinstance(v, ASTNode) and v.type == "VAR":
                    vars_str_list.append(f"{v.name} (VAR)")
                else:
                    vars_str_list.append(str(v))
            vars_str = ", ".join(vars_str_list)
            name += f" ({self.quantifier} {vars_str})"

        lines = [prefix + ("└── " if is_last else "├── ") + name]

        children = []
        if self.type == "QUANTIFIED":
            children.append(self.formula)
        elif self.type == "BINOP":
            children.extend([self.left, self.right])
        elif self.type == "NOT":
            children.append(self.arg)
        elif self.type == "PRED":
            children.extend(self.args)

        for i, c in enumerate(children):
            if c is not None:
                last = i == len(children) - 1
                if isinstance(c, ASTNode):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    lines.extend(c._tree_lines(new_prefix, last))
                elif isinstance(c, list):
                    for j, subc in enumerate(c):
                        sub_last = j == len(c) - 1
                        if isinstance(subc, ASTNode):
                            new_prefix = prefix + ("    " if is_last else "│   ")
                            lines.extend(subc._tree_lines(new_prefix, sub_last))
                        else:
                            new_prefix = prefix + ("    " if is_last else "│   ")
                            lines.append(new_prefix + ("└── " if sub_last else "├── ") + str(subc))
                else:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    lines.append(new_prefix + ("└── " if last else "├── ") + str(c))

        return lines

    def to_str(self):
        return "\n".join(self._tree_lines())

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return self.to_str()

class FolTransformer(Transformer):
    def formula(self, items):
        items = [i for i in items if not (isinstance(i, Token) and i.type in {"LPAR", "RPAR"})]

        if len(items) == 1:
            i = items[0]
            if isinstance(i, Tree):
                return self.transform(i)
            elif isinstance(i, ASTNode):
                return i
            elif isinstance(i, Token):
                return ASTNode("VAR", name=str(i))
            else:
                return i

        if len(items) == 3:
            left = self.transform(items[0]) if isinstance(items[0], Tree) else items[0]
            op = str(items[1])
            right = self.transform(items[2]) if isinstance(items[2], Tree) else items[2]
            return ASTNode("BINOP", op=op, left=left, right=right)

        return ASTNode("VAR", name=str(items[0]))

    def VAR(self, token):
        return ASTNode("VAR", name=str(token))

    def CONST(self, token):
        return ASTNode("CONST", name=str(token))

    def pred(self, items):
        predicate = str(items[0])
        args = []
        for item in items[1:]:
            if isinstance(item, Tree):
                transformed = self.transform(item)
                if isinstance(transformed, list):
                    args.extend(transformed)
                else:
                    args.append(transformed)
            elif isinstance(item, list):
                args.extend([self.transform(x) if isinstance(x, Tree) else x for x in item])
            else:
                args.append(item)
        args = [a for a in args if not (isinstance(a, Token) and a.type in {"LPAR", "RPAR", "COMMA"})]
        return ASTNode("PRED", predicate=predicate, args=args)

    def binop(self, items):
        left = self.transform(items[0]) if isinstance(items[0], Tree) else items[0]
        right = self.transform(items[2]) if isinstance(items[2], Tree) else items[2]
        op = str(items[1])
        return ASTNode("BINOP", op=op, left=left, right=right)

    def neg(self, items):
        arg = self.transform(items[0]) if isinstance(items[0], Tree) else items[0]
        return ASTNode("NOT", arg=arg)

    def multi_quant(self, items):
        quantifier = str(items[0])
        variables = []
        for v in items[1:]:
            if isinstance(v, Tree):
                variables.append(self.transform(v))
            else:
                variables.append(v)
        return quantifier, variables

    def quantified_sentence(self, items):
        quantifier, variables = items[0]
        formula = self.transform(items[1]) if isinstance(items[1], Tree) else items[1]
        return ASTNode("QUANTIFIED", quantifier=quantifier, variables=variables, formula=formula)

    def term(self, items):
        return self.transform(items[0]) if isinstance(items[0], Tree) else items[0]

    def terms(self, items):
        result = []
        for i in items:
            if isinstance(i, Tree):
                transformed = self.transform(i)
                if isinstance(transformed, list):
                    result.extend(transformed)
                else:
                    result.append(transformed)
            else:
                result.append(i)
        return result

class FreeVariableChecker:
    def __init__(self):
        self.bound_stack: list[set[str]] = []
        self.free_vars: set[str] = set()

    def visit(self, node):
        if isinstance(node, ASTNode):
            t = node.type
            if t == "QUANTIFIED":
                self.bound_stack.append(set(v.name if isinstance(v, ASTNode) else str(v) for v in node.variables))
                self.visit(node.formula)
                self.bound_stack.pop()
            elif t == "VAR":
                if not any(node.name in scope for scope in self.bound_stack):
                    self.free_vars.add(node.name)
            elif t in ("BINOP", "NOT", "PRED"):
                children = []
                if t == "BINOP":
                    children = [node.left, node.right]
                elif t == "NOT":
                    children = [node.arg]
                elif t == "PRED":
                    children = node.args
                for child in children:
                    if child is not None:
                        self.visit(child)
        elif isinstance(node, list):
            for item in node:
                self.visit(item)


def is_valid_fol(formula: str) -> bool:
    if not isinstance(formula, str):
        return False
    try:
        parser.parse(formula)
        return True
    except LarkError:
        return False


def is_closed_fol(formula: str) -> bool:
    try:
        tree = parser.parse(formula)
    except LarkError:
        return False
    transformed = FolTransformer().transform(tree)
    checker = FreeVariableChecker()
    checker.visit(transformed)
    return len(checker.free_vars) == 0


def is_valid_fol_closed(formula: str) -> bool:
    return is_valid_fol(formula) and is_closed_fol(formula)


def parse_fol(formula: str):
    tree = parser.parse(formula)
    transformer = FolTransformer()
    return transformer.transform(tree)


def tokenize(formula: str) -> list[str]:
    f = formula.replace(" ", "")
    for s in ALLOWED_LOGIC_OPS + UNARY_OPS + ["(", ")", ","]:
        f = f.replace(s, f" {s} ")
    return [t for t in f.split() if t]


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
    op_threshold: float = OP_DEFAULT_THRESHOLD,
) -> bool:
    if not isinstance(formula_a, str) or not isinstance(formula_b, str):
        return False
    a = tokenize(formula_a)
    b = tokenize(formula_b)
    paren_similar = parentheses_signature(a) == parentheses_signature(b)
    a_norm = normalize_variables(a)
    b_norm = normalize_variables(b)
    pred_score = predicate_similarity_score(
        extract_predicates(a_norm),
        extract_predicates(b_norm),
    )
    ops_a = [t for t in a_norm if t in ALLOWED_LOGIC_OPS + UNARY_OPS]
    ops_b = [t for t in b_norm if t in ALLOWED_LOGIC_OPS + UNARY_OPS]
    op_score = len(set(ops_a).intersection(ops_b)) / max(1, len(set(ops_a)))
    return paren_similar and pred_score >= pred_threshold and op_score >= op_threshold


if __name__ == "__main__":
    f1 = "∀x (HealthyLifestyle(x) ↔ RegularExercise(x) ∧ BalancedDiet(x) ∧ AdequateSleep(x))"
    f2 = "∀z (HealthyLifestyle(z) ↔ BalancedDiet(z) ∧ AdequateSleep(z) ∧ RegularExercise(z))"
    print("Compare FOL:", compare_fol(f1, f2))

    f3 = "∀x (Dish(x) ∧ LiquidFood(x) ∧ UsuallyMadeByBoiling(x, vegetables, meat, fish, water, stock) → Soup(x))"
    print("Parse FOL:\n", parse_fol(f3))

    f_ok = "∀x (Dish(x) → Food(x))"
    f_bad = "∀x (Dish(x) → Food(y))"

    print("syntactic ok:", is_valid_fol(f_ok))
    print("closed ok:", is_valid_fol_closed(f_ok))
    print("syntactic bad:", is_valid_fol(f_bad))
    print("closed bad:", is_valid_fol_closed(f_bad))
