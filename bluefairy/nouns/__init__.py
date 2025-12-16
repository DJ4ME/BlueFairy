import csv
import re
from typing import Optional
from bluefairy.nouns.utils import Stakeholder, TextualNorms, NounsCollection, Role
from ollamaUtils import OllamaService, OLLAMA_URL, OLLAMA_PORT
from bluefairy.prompts import PromptTask, load_system_prompt, PATH

OLLAMA_SERVICE = OllamaService(OLLAMA_URL, OLLAMA_PORT)
OLLAMA_MODEL = "phi3.5:latest"
TEMPERATURE = 0.0
NOUNS_FILE = "nouns_collection.csv"
HEADER = "Noun,Role\n"


try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

try:
    import inflect
    _INFLECT = inflect.engine()
except Exception:
    _INFLECT = None


STOP_TERMS = {
    "i", "you", "he", "she", "they",
    "we", "it", "all", "none", "note",
    "sentence", "output"
}

INVALID_POS = {"DET", "PRON", "ADP", "AUX"}


def run_context_identification(norms: TextualNorms) -> str:
    """
    Step 1.1
    The function asks an LLM to generate a brief description of the domain of the norms.
    :param norms: the textual norms to analyze
    :return: a brief description of the domain of the norms
    """

    system_prompt = load_system_prompt(PromptTask.context_identification)
    ollama_model = OLLAMA_SERVICE.use(OLLAMA_MODEL, system_prompt)
    # unfold the list into a single string with new lines between each norm
    textual_norm = '\n'.join(norms)
    print("Identifying the context of the textual norms...")
    response = ollama_model.ask(textual_norm, max_output=2048, temperature=TEMPERATURE)
    print("Context identification completed.")
    return response.strip()


def run_nouns_generation(norms: TextualNorms, context: str, file = PATH / NOUNS_FILE) -> None:
    """
    Step 1.2
    The function asks an LLM to generate a list of relevant nouns that are present in each norm.
    Append the nouns to the nouns file (empty role).
    :param norms: the textual norms to analyze
    :param context: the context of the norms
    :param file: the nouns file path
    :return: none
    """

    system_prompt = load_system_prompt(PromptTask.nouns_generation)
    # substitute the context placeholder in the system prompt
    system_prompt = system_prompt.replace("{domain_context}", context)
    ollama_model = OLLAMA_SERVICE.use(OLLAMA_MODEL, system_prompt)
    # delete the file if it exists and create a new one with header
    with open(file, "w", encoding="utf-8") as _file:
        _file.write(HEADER)
    # interactively query the model
    size = len(norms)
    for norm in norms:
        print(f"Processing textual norm {norms.index(norm)+1}/{size}")
        response = ollama_model.ask(norm, max_output=1024, temperature=TEMPERATURE)
        # collect each noun, assume they are separated by new lines or commas
        nouns = re.split(r"[\n,]+", response.strip())
        # append the nouns to the nouns file
        with open(file, "a", encoding="utf-8") as _file:
            for noun in nouns:
                noun = noun.strip().replace('"', '')
                _file.write(f"{noun},{Role.undefined.name}\n")


def run_nouns_cleaning(file = PATH / NOUNS_FILE) -> None:
    """
    Step 1.3
    The function cleans the nouns file from duplicates and inconsistent or irrelevant nouns.
    :param file: the nouns file path
    :return: none
    """

    def singularize(term: str) -> str:
        """Singularize a term if possible."""
        if not _INFLECT:
            return term
        words = term.split()
        if len(words) == 1:
            singular = _INFLECT.singular_noun(words[0])
            return str(singular) if singular else term
        return term

    def is_valid_pos(term: str) -> bool:
        """Check POS constraints using spaCy if available."""
        if not _NLP:
            return True
        doc = _NLP(term)
        return all(tok.pos_ not in INVALID_POS for tok in doc)

    def clean_term(term: str) -> Optional[str]:
        """Apply all cleaning rules to a single term."""
        if not term:
            return None

        # lowercase + strip
        term = term.lower().strip()

        # remove special characters except hyphen
        term = re.sub(r"[^\w\s-]", "", term)

        # remove numbers
        if re.search(r"\d", term):
            return None

        # normalize underscores
        term = term.replace("_", " ").strip()

        # collapse whitespace
        term = re.sub(r"\s+", " ", term)

        # stoplist
        if term in STOP_TERMS:
            return None

        # length constraint
        if len(term.split()) > 3:
            return None

        # POS-based filter
        if not is_valid_pos(term):
            return None

        # singularize (last step)
        term = singularize(term)

        return term or None

    cleaned = {}

    with open(file, "r", encoding="utf-8") as _file:
        reader = csv.DictReader(_file)
        for row in reader:
            noun, role = row['Noun'], row['Role']
            cleaned_noun = clean_term(noun)
            if not cleaned_noun:
                continue

            # keep first non-undefined role if conflicts arise
            if cleaned_noun not in cleaned:
                cleaned[cleaned_noun] = role
            else:
                if cleaned[cleaned_noun] == "undefined" and role != "undefined":
                    cleaned[cleaned_noun] = role

    with open(file, "w", encoding="utf-8") as _file:
        _file.write(HEADER)
        for noun, role in cleaned.items():
            _file.write(f"{noun},{role}\n")


def run_nouns_classification(context: str, file = PATH / NOUNS_FILE) -> None:
    """
    Step 1.4
    The function asks an LLM to classify each noun in the nouns file with a role.
    Update the nouns file with the classified roles.
    :param context: the context of the norms
    :param file: the nouns file path
    :return: none
    """

    system_prompt = load_system_prompt(PromptTask.nouns_classification)
    # substitute the context placeholder in the system prompt
    system_prompt = system_prompt.replace("{domain_context}", context)
    ollama_model = OLLAMA_SERVICE.use(OLLAMA_MODEL, system_prompt)
    # load the nouns collection
    nouns_collection = NounsCollection()
    with open(file, "r", encoding="utf-8") as _file:
        reader = csv.DictReader(_file)
        for row in reader:
            noun, role = row['Noun'], row['Role']
            nouns_collection.add_noun(noun, role)
    # classify each noun with undefined role
    size = len(nouns_collection.nouns)
    idx = 0
    for noun, role in nouns_collection.nouns.items():
        if role != Role.undefined.name:
            continue
        print(f"Classifying noun '{noun}' ({idx+1}/{size})")
        idx += 1
        response = ollama_model.ask(noun, max_output=256, temperature=TEMPERATURE)
        # extract only the first word as the classified role and parse it
        classified_role = response.strip().split()[0].lower().replace('"', '').replace("'", "")
        if classified_role in [role.name for role in Role]:
            nouns_collection.add_noun(noun, classified_role)
        else:
            print(f"Warning: Invalid role '{classified_role}' for noun '{noun}'. Keeping 'undefined'.")
    # save the updated nouns collection
    nouns_collection.save_nouns_to_csv(file)


def run_nouns_collection(stakeholders: list[Stakeholder], file = PATH / NOUNS_FILE) -> None:
    """
    Main function to run the nouns collection process.
    :param stakeholders: the list of stakeholders with their norms
    :param file: the output file path for the nouns collection
    :return: none
    """

    # aggregate all norms from stakeholders
    all_norms: TextualNorms = []
    for stakeholder in stakeholders:
        all_norms.extend(stakeholder.norms)

    # Step 1.1: Context Identification
    context = run_context_identification(all_norms)
    print(f"Identified context: {context}")

    # Step 1.2: Nouns Generation
    run_nouns_generation(all_norms, context, file)

    # Step 1.3: Nouns Cleaning
    run_nouns_cleaning(file)

    # Step 1.4: Nouns Classification
    run_nouns_classification(context, file)