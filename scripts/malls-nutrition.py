import pandas as pd
from test.data import PATH as DATA_PATH

SPLITS = {'train': 'MALLS-v0.1-train.json', 'test': 'MALLS-v0.1-test.json'}
TRAIN_FILE = DATA_PATH / SPLITS['train']
TRAIN_NUTRITION_FILE = DATA_PATH / 'MALLS-nutrition-train.json'
TEST_FILE = DATA_PATH / SPLITS['test']
TEST_NUTRITION_FILE = DATA_PATH / 'MALLS-nutrition-test.json'


def download_malls_nutrition_data() -> None:
    """
    Downloads and filters the MALLS nutrition-related dataset if not already present.
    :return: None
    """
    if not TRAIN_FILE.exists():
        train = pd.read_json("hf://datasets/yuan-yang/MALLS-v0/" + SPLITS["train"])
        train.to_json(TRAIN_FILE, orient='records', lines=False)
    if not TEST_FILE.exists():
        test = pd.read_json("hf://datasets/yuan-yang/MALLS-v0/" + SPLITS["test"])
        test.to_json(TEST_FILE, orient='records', lines=False)


def filter_nutrition_related_data(df: pd.DataFrame) -> pd.DataFrame:
    nutrition_keywords = [
        'nutrition', 'calorie', 'vitamin', 'mineral', 'protein', 'carbohydrate','fat',
        'fiber', 'sugar', 'sodium', 'dietary', 'nutrient', 'health', 'healthy', 'wellness'
    ]

    def is_nutrition_related(text):
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in nutrition_keywords)

    filtered_df = df[df['NL'].apply(is_nutrition_related) | df['FOL'].apply(is_nutrition_related)]
    print(f"Filtered {len(filtered_df)} nutrition-related entries out of {len(df)} total entries.")
    return filtered_df


if __name__ == "__main__":
    download_malls_nutrition_data()

    # Process training data
    train_df = pd.read_json(TRAIN_FILE)
    nutrition_train_df = filter_nutrition_related_data(train_df)
    nutrition_train_df.to_json(TRAIN_NUTRITION_FILE, orient='records', lines=False)

    # Process testing data
    test_df = pd.read_json(TEST_FILE)
    nutrition_test_df = filter_nutrition_related_data(test_df)
    nutrition_test_df.to_json(TEST_NUTRITION_FILE, orient='records', lines=False)