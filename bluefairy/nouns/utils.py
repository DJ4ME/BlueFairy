import csv
from enum import Enum
from PyPDF2 import PdfReader

Noun = str
Role = Enum("Role", "predicate constant undefined")
LabeledNoun = tuple[Noun, Role]
LabeledNouns = dict[Noun: Role]
TextualNorm = str
TextualNorms = list[TextualNorm]


class Stakeholder:
    def __init__(self, noun: str):
        self.noun: str = noun
        self._norms: TextualNorms = []

    @property
    def norms(self) -> TextualNorms:
        return self._norms

    @norms.setter
    def norms(self, norms: TextualNorms) -> None:
        self._norms = norms

    def __repr__(self):
        return f"Stakeholder(noun={self.noun}, has_norms={len(self._norms) > 0})"


def load_norms_from_csv(file_path: str) -> TextualNorms:
    import csv
    norms: TextualNorms = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            norms.append(row[0])  # Assuming norms are in the first column
    return norms


def load_norms_from_txt(file_path: str) -> TextualNorms:
    norms: TextualNorms = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        for line in file:
            norms.append(line.strip())
    return norms


def load_norms_from_pdf(file_path: str) -> TextualNorms:
    norms: TextualNorms = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            norms.extend([line.strip() for line in text.splitlines() if line.strip()])
    return norms


class NounsCollection:
    def __init__(self):
        self._nouns: LabeledNouns = LabeledNouns()

    def add_noun(self, noun: Noun, role: Role) -> None:
        if noun not in self._nouns:
            self._nouns.update({noun: role})
        else:
            if role == Role.undefined.name:
                pass
            elif self._nouns[noun] != role:
                if self._nouns[noun] == Role.undefined.name:
                    self._nouns.update({noun: role})
                else:
                    print(f"Name '{noun}' already exists with a different role.")
                    print(f"Existing role: {self._nouns[noun]}, New role: {role}")

    @property
    def nouns(self) -> LabeledNouns:
        return self._nouns

    def save_nouns_to_csv(self, file_path: str) -> None:
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Noun', 'Role'])
            for noun, role in self._nouns.items():
                writer.writerow([noun, role])



