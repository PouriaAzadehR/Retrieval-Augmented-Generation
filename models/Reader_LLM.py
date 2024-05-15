from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


class ReaderLLM:

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=self.bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
        self.task = "text-generation"
        self.do_sample = True
        self.temperature = 0.2
        self.repetition_penalty = 1.1
        self.return_full_text = False
        self.max_new_tokens = 500
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.llm_pipeline = self.reader_LLM_pipeline()

    def reader_LLM_pipeline(self):
        return pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task,
            do_sample=self.do_sample,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            return_full_text=self.return_full_text,
            max_new_tokens=self.max_new_tokens,
        )
