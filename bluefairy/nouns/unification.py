from bluefairy.grammar.utils import parse_or_false, PRED_KEY


def collect_predicates(list_of_fol_formulas: list[str]) -> set[PRED_KEY]:
    """
    Collects unique predicates name and their arity from a list of first-order logic formulas.
    :param list_of_fol_formulas: the list of first-order logic formulas
    :return: a set of unique predicates
    """
    predicates: set[PRED_KEY] = set()
    for fol in list_of_fol_formulas:
        parsed = parse_or_false(fol)
        if not parsed:
            continue
        for pred_name, arity in parsed.get_predicates():
            predicates.add((pred_name, arity))
    return predicates


if __name__ == "__main__":
    test_formulas = [
        '∀x (Person(x) → ∃y (Loves(x, y) ∧ Person(y)))',
        '∃z (Dog(z) ∧ Barks(z))',
        '∀a ∀b (Friend(a, b) → Friend(b, a))',
        '¬∃c (Cat(c) ∧ Chases(c, Mouse))'
    ]
    preds = collect_predicates(test_formulas)
    print("Collected predicates and their arities:")
    for pred in preds:
        print(pred)

