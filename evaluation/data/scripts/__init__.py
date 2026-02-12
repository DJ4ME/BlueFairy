from bluefairy.nouns.unification import collect_predicates, collect_constants
from evaluation.data import load_test_set

TEST_SET = load_test_set()


if __name__ == "__main__":
    predicates = collect_predicates(TEST_SET['FOL'].tolist())
    print(f"Total unique predicates: {len(predicates)}")
    constants = collect_constants(TEST_SET['FOL'].tolist())
    print(f"Total unique constants: {len(constants)}")
