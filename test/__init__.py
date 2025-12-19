import re

LOGICAL_OPERATORS = {"∧", "∨", "¬", "→", "↔", "⊕"}
QUANTIFIERS = {"∀", "∃"}
PREDICATE_PATTERN = re.compile(r"([A-Z][A-Za-z0-9_]*)\((.*?)\)")
VARIABLE_PATTERN = re.compile(r"[a-z][A-Za-z0-9_]*")
PREDICATE_CALL_PATTERN = re.compile(r"[A-Z][A-Za-z0-9_]*\s*\(([^()]*)\)")
QUANTIFIER_PATTERN = re.compile(r"[∀∃]\s*([a-z][A-Za-z0-9_]*)")


def check_parentheses_balance(formula: str) -> (bool, str):
    stack = []
    for c in formula:
        if c == "(":
            stack.append(c)
        elif c == ")":
            if not stack:
                return False, "Parenthesis closed without opening"
            stack.pop()
    if stack:
        return False, "Unbalanced parentheses"
    return True, ""


def extract_variables_declared(formula: str) -> (set, str):
    declared = set()
    i = 0
    while i < len(formula):
        if formula[i] in QUANTIFIERS:
            j = i + 1
            # consume variable characters
            var = ""
            while j < len(formula) and formula[j].isalnum():
                var += formula[j]
                j += 1

            if var == "":
                return None, f"quantifier {formula[i]} must declare a variable"

            declared.add(var)
            i = j
        i += 1
    return declared, ""


def extract_predicates_with_args(formula: str) -> list:
    """
    Returns:
        list of tuples: [(predicate_name, [args])]
    """
    results = []
    for match in PREDICATE_PATTERN.finditer(formula):
        name = match.group(1)
        args = match.group(2)

        # Split args by comma respecting no recursion
        arg_list = [a.strip() for a in args.split(",") if a.strip() != ""]

        results.append((name, arg_list))
    return results


def extract_variables(formula: str) -> set:
    variables = set()

    for v in QUANTIFIER_PATTERN.findall(formula):
        variables.add(v)

    for args in PREDICATE_CALL_PATTERN.findall(formula):
        parts = [p.strip() for p in args.split(",")]
        for p in parts:
            if p and p[0].islower():
                variables.add(p)

    return variables


def check_variables_usage(formula: str, declared_vars: set):
    vars_found = extract_variables(formula)

    undeclared = vars_found - declared_vars

    if undeclared:
        return False, f"undeclared variables used: {undeclared}"

    return True, ""


def check_operator_sequence(formula: str) -> (bool, str):
    """
    Completely superficial: rejects accidental operator repetition.
    """
    # remove predicates so we don't detect commas etc.
    stripped = PREDICATE_PATTERN.sub("PRED", formula)

    for op in LOGICAL_OPERATORS:
        bad = op + op
        if bad in stripped:
            return False, f"operator repeated: {bad}"

    return True, ""


def deduce_predicate_arities(predicates) -> (dict, str):
    """
    predicates: list of tuples from extract_predicates_with_args
    returns dict mapping predicate -> arity or error
    """
    arities = {}
    for name, args in predicates:
        arity = len(args)
        if name not in arities:
            arities[name] = arity
        else:
            if arities[name] != arity:
                return None, f"predicate {name} has inconsistent arity"
    return arities, ""


def check_formula(formula: str) -> (bool, dict or str):
    """
    Full procedure: returns (success, message or data dictionary)
    """

    ok, msg = check_parentheses_balance(formula)
    if not ok:
        return False, msg

    declared, msg = extract_variables_declared(formula)
    if declared is None:
        return False, msg

    preds = extract_predicates_with_args(formula)

    arities, msg = deduce_predicate_arities(preds)
    if arities is None:
        return False, msg

    ok, msg = check_variables_usage(formula, declared)
    if not ok:
        return False, msg

    ok, msg = check_operator_sequence(formula)
    if not ok:
        return False, msg

    return True, {
        "declared_variables": declared,
        "predicates_detected": preds,
        "predicate_arities": arities,
    }
