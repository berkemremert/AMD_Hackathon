import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LocalGenerationConfig:
    model_id: str
    device: str
    dtype: str
    max_input_tokens: int
    default_max_new_tokens: int
    temperature: float
    do_sample: bool
    num_beams: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    maximum_repair_attempts: int

def get_config() -> LocalGenerationConfig:
    return LocalGenerationConfig(
        model_id=os.environ.get("LOCAL_SUMMARY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        device=os.environ.get("LOCAL_SUMMARY_DEVICE", "cpu"),
        dtype=os.environ.get("LOCAL_SUMMARY_DTYPE", "auto"),
        max_input_tokens=int(os.environ.get("LOCAL_SUMMARY_MAX_INPUT_TOKENS", "4096")),
        default_max_new_tokens=int(os.environ.get("LOCAL_SUMMARY_DEFAULT_NEW_TOKENS", "256")),
        temperature=float(os.environ.get("LOCAL_SUMMARY_TEMPERATURE", "0.0")),
        do_sample=os.environ.get("LOCAL_SUMMARY_DO_SAMPLE", "false").lower() == "true",
        num_beams=int(os.environ.get("LOCAL_SUMMARY_NUM_BEAMS", "1")),
        repetition_penalty=float(os.environ.get("LOCAL_SUMMARY_REPETITION_PENALTY", "1.0")),
        no_repeat_ngram_size=int(os.environ.get("LOCAL_SUMMARY_NO_REPEAT_NGRAM", "0")),
        maximum_repair_attempts=int(os.environ.get("LOCAL_SUMMARY_MAX_REPAIR_ATTEMPTS", "1"))
    )

def get_mode() -> str:
    return os.environ.get("LOCAL_SUMMARY_MODE", "compress_only")

def get_failure_policy() -> str:
    return os.environ.get("LOCAL_SUMMARY_FAILURE_POLICY", "return_best")

def get_min_compression_words() -> int:
    return int(os.environ.get("LOCAL_SUMMARY_MIN_COMPRESSION_WORDS", "250"))
