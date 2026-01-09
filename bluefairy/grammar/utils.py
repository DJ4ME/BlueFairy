import re
from pathlib import Path
from lark import Lark, Transformer, Tree, Token
from lark.exceptions import LarkError
from Levenshtein import distance as levenshtein

PATH = Path(__file__).parent.resolve()

PRED_KEY = tuple[str, int]  # (predicate_name, arity)
ALLOWED_LOGIC_OPS = ["∧", "∨", "⊕", "→", "↔"]
UNARY_OPS = ["¬"]

PRED_DEFAULT_THRESHOLD = 0.8
OP_DEFAULT_THRESHOLD = 0.9
MAX_SIMILARITY_SCORE = 999

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

    def get_args_num(self):
        if self.args is not None and isinstance(self.args, list):
            return len([x for x in self.args if x.type in {"VAR", "CONST", "PRED"}])
        return 0

    def get_predicates(self) -> list[PRED_KEY]:
        predicates = []
        if self.type == "PRED":
            predicates.append((self.predicate, self.get_args_num()))
        elif self.type == "BINOP":
            predicates.extend(self.left.get_predicates())
            predicates.extend(self.right.get_predicates())
        elif self.type == "NOT":
            pass
        elif self.type == "QUANTIFIED" and isinstance(self.formula, ASTNode):
            predicates.extend(self.formula.get_predicates())
        return predicates

    def get_full_predicates(self) -> list[PRED_KEY, list[str]]:
        predicates = []
        if self.type == "PRED":
            arg_names = []
            for arg in self.args:
                if arg.type not in {"VAR", "CONST", "PRED"}:
                    continue
                if isinstance(arg, ASTNode):
                    arg_names.append(arg.name)
                else:
                    arg_names.append(str(arg))
            predicates.append(((self.predicate, self.get_args_num()), arg_names))
        elif self.type == "BINOP":
            predicates.extend(self.left.get_full_predicates())
            predicates.extend(self.right.get_full_predicates())
        elif self.type == "NOT":
            pass
        elif self.type == "QUANTIFIED" and isinstance(self.formula, ASTNode):
            predicates.extend(self.formula.get_full_predicates())
        return predicates

    def get_constants(self):
        constants = set()
        if self.type == "CONST":
            constants.add(self.name)
        elif self.type == "BINOP":
            constants.update(self.left.get_constants())
            constants.update(self.right.get_constants())
        elif self.type == "NOT":
            pass
        elif self.type == "PRED":
            for arg in self.args:
                if isinstance(arg, ASTNode):
                    constants.update(arg.get_constants())
                else:
                    pass
        elif self.type == "QUANTIFIED":
            constants.update(self.formula.get_constants())
        return constants

    def __str__(self):
        return "\n".join(self._tree_lines())

    def __repr__(self):
        return self.__str__()

    def _tree_lines(self, prefix="", is_last=True):
        name = self.type
        if self.type == "BINOP":
            name += f" ({self.op})"
        elif self.type == "PRED":
            name += f" ({self.predicate})"
        elif self.type in ("VAR", "CONST"):
            name += f" ({self.name})"
        elif self.type == "QUANTIFIED":
            vars_str = ", ".join(f"{v.name} (VAR)" for v in self.variables)
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

        children = [c for c in children if isinstance(c, ASTNode)]

        for i, c in enumerate(children):
            new_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(c._tree_lines(new_prefix, i == len(children) - 1))

        return lines


class FolTransformer(Transformer):
    def formula(self, items):
        items = [i for i in items if not (isinstance(i, Token) and i.type in {"LPAR", "RPAR"})]

        if len(items) == 1:
            i = items[0]
            if isinstance(i, Tree):
                return self.transform(i)
            return i

        if len(items) == 3:
            left = items[0] if isinstance(items[0], ASTNode) else self.transform(items[0])
            right = items[2] if isinstance(items[2], ASTNode) else self.transform(items[2])

            op_item = items[1]
            if isinstance(op_item, Tree):
                op = str(op_item.children[0])
            else:
                op = str(op_item)

            return ASTNode("BINOP", op=op, left=left, right=right)

        return items[0]

    def VAR(self, token):
        return ASTNode("VAR", name=str(token))

    def CONST(self, token):
        return ASTNode("CONST", name=str(token))

    def pred(self, items):
        predicate = str(items[0])
        args = []
        for item in items[1:]:
            if isinstance(item, Tree):
                item = self.transform(item)
            if isinstance(item, list):
                args.extend(item)
            elif isinstance(item, ASTNode):
                args.append(item)
        return ASTNode("PRED", predicate=predicate, args=args)

    def neg_pred(self, items):
        node = self.transform(items[0]) if isinstance(items[0], Tree) else items[0]
        return ASTNode("NOT", arg=node)

    def binop(self, items):
        left = items[0] if isinstance(items[0], ASTNode) else self.transform(items[0])
        right = items[2] if isinstance(items[2], ASTNode) else self.transform(items[2])

        op_item = items[1]
        if isinstance(op_item, Tree):
            op = str(op_item.children[0])
        else:
            op = str(op_item)

        return ASTNode("BINOP", op=op, left=left, right=right)

    def binop_sentence(self, items):
        return self.binop(items)

    def neg(self, items):
        node = items[0] if isinstance(items[0], ASTNode) else self.transform(items[0])
        return ASTNode("NOT", arg=node)

    def multi_quant(self, items):
        return str(items[0]), items[1]

    def quantified_sentence(self, items):
        formula = items[-1]
        if isinstance(formula, Tree):
            formula = self.transform(formula)

        for q, v in reversed(items[:-1]):
            formula = ASTNode(
                "QUANTIFIED",
                quantifier=q,
                variables=[v] if not isinstance(v, list) else v,
                formula=formula
            )
        return formula

    def term(self, items):
        return self.transform(items[0]) if isinstance(items[0], Tree) else items[0]

    def terms(self, items):
        result = []
        for i in items:
            if isinstance(i, Tree):
                i = self.transform(i)
            if isinstance(i, list):
                result.extend(i)
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
                vars_list = node.variables
                if not isinstance(vars_list, list):
                    vars_list = [vars_list]
                self.bound_stack.append(set(v.name if isinstance(v, ASTNode) else str(v) for v in vars_list))
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


def is_closed_fol(formula: str or Tree) -> bool:
    if isinstance(formula, str):
        try:
            tree = parser.parse(formula)
        except LarkError:
            return False
    else:
        tree = formula
    transformed = FolTransformer().transform(tree)
    checker = FreeVariableChecker()
    checker.visit(transformed)
    return len(checker.free_vars) == 0


def is_valid_fol(formula: str) -> bool:
    if not isinstance(formula, str):
        return False
    try:
        tree = parser.parse(formula)
        return True and is_closed_fol(tree)
    except LarkError:
        return False


def parse_fol(formula: str):
    tree = parser.parse(formula)
    transformer = FolTransformer()
    return transformer.transform(tree)


def parse_or_false(formula: str):
    try:
        return parse_fol(formula)
    except LarkError:
        return False


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
        best = MAX_SIMILARITY_SCORE
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

    f4 = "∀x ∃y (Person(x) → (Loves(x, y) ∧ Person(y)))"
    print("Parse FOL:\n", parse_fol(f4))

    f_ok = "∀x (Dish(x) → Food(x))"
    f_bad = "∀x (Dish(x) → Food(y))"

    print("syntactic ok:", is_valid_fol(f_ok))
    print("syntactic bad:", is_valid_fol(f_bad))
