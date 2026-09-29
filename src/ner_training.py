import pandas as pd
import torch

from src.text_generation import generate_text

class NerTraining:
    """Convert a string into the BIO NER representation for training and evaluation."""
    def __init__(self, tokenizer: any, device: torch.device) -> None:
        """Init the NER Training class."""

        self.device = device

        # Define the BIO tags for each entity type
        # B- = Beginning of an entity, I- = Inside of an entity, O = Outside of any entity
        # Adverse Reaction (REACT), Active Principle (ACTIVE), Frequency (FREQ), System (SYS)
        self.start_adverse_reaction = "B-REACT"
        self.inside_adverse_reaction = "I-REACT"
        self.start_active_principle = "B-ACTIVE"
        self.inside_active_principle = "I-ACTIVE"
        self.start_frequency = "B-FREQ"
        self.inside_frequency = "I-FREQ"
        self.start_system = "B-SYS"
        self.inside_system = "I-SYS"
        self.label_to_id = {
            "O": 0,
            self.start_active_principle: 1,
            self.inside_active_principle: 2,
            self.start_adverse_reaction: 3,
            self.inside_adverse_reaction: 4,
            self.start_frequency: 5,
            self.inside_frequency: 6,
            self.start_system: 7,
            self.inside_system: 8,
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.tokenizer = tokenizer

    def get_label_names(self):
        """Get the label names."""
        labels = set()
        for label in self.label_to_id.keys():
            # Remove B- or I- prefix if present
            clean_label = label[2:] if label.startswith(("B-", "I-")) else label
            labels.add(clean_label.upper())
        return list(labels)

    def find_token_sequence(self, tokens, sequence):
        """
        Find the starting index of a token sequence in a list of tokens
        """
        if not sequence:
            return -1

        for i in range(len(tokens) - len(sequence) + 1):
            if tokens[i : i + len(sequence)] == sequence:
                return i
        return -1

    def convert_id_to_labels(self, ids: list[int]) -> list[str]:
        """Convert the ids to the value of the labes."""
        return [self.id_to_label[id_] for id_ in ids]

    def tokenize_sentence(
        self, value: str, active: str, medication: str, adverse: str, frequency: str, system: str
    ) -> tuple[any, list[int]]:
        """Tokenize a sentence and convert into the NER representation."""
        tokens = self.tokenizer(
            value, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        tokens = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        token_labels = ["O"] * len(tokens)

        entities = [
            (active, self.start_active_principle, self.inside_active_principle),
            (adverse, self.start_adverse_reaction, self.inside_adverse_reaction),
            (frequency, self.start_frequency, self.inside_frequency),
            (system, self.start_system, self.inside_system),
        ]

        for text, start_tag, inside_tag in entities:
            target_tokens = self.tokenizer.tokenize(text)
            if not target_tokens:
                continue

            for i in range(len(tokens) - len(target_tokens) + 1):
                if tokens[i : i + len(target_tokens)] == target_tokens:
                    token_labels[i] = start_tag
                    for j in range(1, len(target_tokens)):
                        if i + j < len(token_labels):
                            token_labels[i + j] = inside_tag

        label_ids = [self.label_to_id[label] for label in token_labels]

        return (tokens, label_ids)

    def prepare_ner_data(self, df: pd.DataFrame):
        """
        Prepare NER data with BIO tagging scheme using more varied templates.
        B-ACTIVE = Beginning of Active Principle
        I-ACTIVE = Inside Active Principle
        B-REACT = Beginning of Adverse Reaction
        I-REACT = Inside Adverse Reaction
        B-FREQ = Beginning of Frequency
        I-FREQ = Inside Frequency
        B-SYS = Beginning of System
        I-SYS = Inside System
        O = Outside (other tokens)
        """
        texts = []
        labels = []

        # Support both uppercase (PACTIVO) and lowercase (principio_activo) column names
        col_map = {
            "active": "PACTIVO" if "PACTIVO" in df.columns else "principio_activo",
            "adverse": "REACADV" if "REACADV" in df.columns else "reaccion_adversa",
            "frequency": "FRECUENCIA" if "FRECUENCIA" in df.columns else "frecuencia",
            "system": "SISTEMA" if "SISTEMA" in df.columns else "sistema",
            "medication": "MEDICAMENTO" if "MEDICAMENTO" in df.columns else "medicamento",
        }

        for _, row in df.iterrows():
            active = str(row[col_map["active"]]).strip() if pd.notna(row[col_map["active"]]) else ""
            adverse = str(row[col_map["adverse"]]).strip() if pd.notna(row[col_map["adverse"]]) else ""
            frequency = str(row[col_map["frequency"]]).strip() if pd.notna(row[col_map["frequency"]]) else ""
            system = str(row[col_map["system"]]).strip() if pd.notna(row[col_map["system"]]) else ""
            medication = str(row[col_map["medication"]]).strip() if pd.notna(row[col_map["medication"]]) else ""

            # More varied sentence templates
            sentence_templates = generate_text(
                active, medication, adverse, frequency, system
            )

            for template in sentence_templates:
                tokens, label_ids = self.tokenize_sentence(
                    template, active, medication, adverse, frequency, system
                )
                texts.append(tokens)
                labels.append(label_ids)

        return texts, labels, self.label_to_id
