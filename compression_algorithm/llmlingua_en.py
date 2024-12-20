import os
import warnings
from llmlingua import PromptCompressor

warnings.filterwarnings("ignore", category=FutureWarning)

class LLMLinguaTextCompressor:
    def __init__(self, 
                 model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                 compression_rate: float = 0.5,
                 force_tokens: list = None,
                 chunk_end_tokens: list = None,
                 device_map: str = "cpu"):
        """
        A wrapper for llmlingua PromptCompressor to compress Chinese text.
        Parameters:
        - model_name: model name
        - compression_rate: compression rate (0-1), lower means more compression
        - force_tokens: tokens to be preserved
        - chunk_end_tokens: tokens marking the end of a chunk
        - device_map: device mapping ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.compression_rate = compression_rate
        self.force_tokens = force_tokens if force_tokens is not None else ['\n']
        self.chunk_end_tokens = chunk_end_tokens if chunk_end_tokens is not None else ['.', '\n']

        self.compressor = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=True,
            device_map=device_map
        )
    
    def compress_text(self, text: str) -> dict:
        """
        Compress the given text and return a dictionary:
        {
            "original_text": original text,
            "compressed_text": cleaned compressed text (no extra spaces),
            "cleaned_ratio": the compression ratio after cleaning
        }
        """
        results = self.compressor.compress_prompt_llmlingua2(
            text,
            rate=self.compression_rate,
            force_tokens=self.force_tokens,
            chunk_end_tokens=self.chunk_end_tokens,
            return_word_label=False,
            drop_consecutive=True
        )

        compressed_prompt = results["compressed_prompt"]
        origin_tokens = results["origin_tokens"]

        # Remove all spaces
        cleaned_compressed_prompt = compressed_prompt.replace(' ', '')

        # Recalculate token count and ratio after cleaning
        encoded = self.compressor.tokenizer.encode(cleaned_compressed_prompt, add_special_tokens=False)
        cleaned_token_count = len(encoded)
        cleaned_ratio = cleaned_token_count / origin_tokens if origin_tokens > 0 else 1.0

        return {
            "original_text": text,
            "compressed_text": cleaned_compressed_prompt,
            "cleaned_ratio": cleaned_ratio
        }


if __name__ == "__main__":
    original_prompt = """请详细描述你的计划，包括目标、步骤和时间安排。
                       这个计划需要清晰的行动方案和可执行的步骤，以确保目标的实现。
                       同时，尽量减少冗余描述，突出重点信息。
                       别再‘纸上谈兵’，要实实在在地行动。"""
    
    lingua_compressor = LLMLinguaTextCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        compression_rate=0.5,
        force_tokens=['\n'],
        chunk_end_tokens=['.', '\n']
    )

    result = lingua_compressor.compress_text(original_prompt)

    print("Original text:", result["original_text"])
    print("Cleaned compressed text:", result["compressed_text"])
    print("Cleaned compression ratio:", result["cleaned_ratio"])
