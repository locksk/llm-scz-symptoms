"""Scientific constants recovered from the historical experiment."""

ID_COLUMN = "Study_ID"
TEXT_COLUMN = "Case_note_summary"
STRATUM_COLUMN = "sumstrat"
FIXED_GROUP = "S1"
RANDOM_SEED = 42

LABEL_COLUMNS = (
    "SAPS_Global_Hals",
    "SAPS_Global_Delusions",
    "SAPS_Global_Positive_Thought_Disorder",
    "SAPS_Global_Bizarre_Behaviour",
    "SANS_Global_Affective_Flattening",
    "SANS_Global_Alogia",
    "SANS_Global_Avolition_Apathy",
    "SANS_Global_Anhedonia_Asociality",
)

SUPPORTED_ENCODERS = (
    "mental/mental-bert-base-uncased",
    "AIMH/mental-bert-large-uncased",
    "UFNLP/gatortron-base",
    "AIMH/mental-longformer-base-4096",
    "google/bigbird-roberta-large",
)

SCHEME_1 = "Scheme_1_S1_Train"
SCHEME_2 = "Scheme_2_S1_ValTest"
SCHEMES = (SCHEME_1, SCHEME_2)


def safe_model_name(model_name: str) -> str:
    """Return the historical filesystem-safe encoder name."""
    return model_name.replace("/", "_")

