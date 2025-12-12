from bluefairy.nouns.utils import Stakeholder, TextualNorms, NounsCollection, Role
from ollamaUtils import OllamaService, OLLAMA_URL, OLLAMA_PORT
from bluefairy.prompts import PromptTask, load_system_prompt

OLLAMA_SERVICE = OllamaService(OLLAMA_URL, OLLAMA_PORT)
OLLAMA_MODEL = "phi3.5"
TEMPERATURE = 0.0



def collect_nouns_from_stakeholders(stakeholders: list[Stakeholder]) -> NounsCollection:
    textual_norms: TextualNorms = []
    labeled_names: NounsCollection = NounsCollection()
    for stakeholder in stakeholders:
        textual_norms.extend(stakeholder.norms)

    # Query Ollama to extract names and their roles from the textual norms
    system_prompt = load_system_prompt(PromptTask.nouns_collection)
    for textual_norm in textual_norms:
        ollama_model = OLLAMA_SERVICE.use(OLLAMA_MODEL, system_prompt)
        response = ollama_model.ask(textual_norm, temperature=TEMPERATURE)
        # Parse the response into labeled names
        for line in response.splitlines():
            if ':' in line:
                name, role = line.split(':', 1)
                name = name.strip().strip('"')
                role = role.strip().strip('"')
                if role not in [enum.name for enum in Role]:
                    role = Role.undefined.name
                labeled_names.add_noun(name, role)
    return labeled_names