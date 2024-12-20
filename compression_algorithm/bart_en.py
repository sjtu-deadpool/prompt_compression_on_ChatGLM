from transformers import BartForConditionalGeneration, BertTokenizer

class BartTextCompressor:
    def __init__(self, 
                 model_name: str = "fnlp/bart-base-chinese",
                 max_length=512,
                 summary_max_length=20,
                 summary_min_length=10,
                 length_penalty=5.0,
                 num_beams=4):
        """
        Use a BART model to compress (summarize) Chinese text.
        Parameters:
        - model_name: BART model name
        - max_length: maximum length for encoding the original text
        - summary_max_length: maximum length for the generated summary
        - summary_min_length: minimum length for the generated summary
        - length_penalty: length penalty for the generation
        - num_beams: number of beams for beam search
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length
        self.summary_max_length = summary_max_length
        self.summary_min_length = summary_min_length
        self.length_penalty = length_penalty
        self.num_beams = num_beams

    def compress_text(self, text: str) -> dict:
        """
        Compress the given text (summarization) and return a dict consistent with previous interfaces:
        {
            "original_text": original text,
            "compressed_text": cleaned compressed text (with no spaces),
            "cleaned_ratio": compression ratio after cleaning (token_count_compressed / token_count_original)
        }
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
        original_token_count = len(inputs["input_ids"][0])

        # Generate summary using BART
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.summary_max_length,
            min_length=self.summary_min_length,
            length_penalty=self.length_penalty,
            num_beams=self.num_beams
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        
        # Remove all spaces
        cleaned_summary = "".join(summary.split())

        # Re-tokenize the cleaned summary to get token count
        summary_inputs = self.tokenizer(cleaned_summary, return_tensors="pt", max_length=self.max_length, truncation=True)
        compressed_token_count = len(summary_inputs["input_ids"][0])

        cleaned_ratio = compressed_token_count / original_token_count if original_token_count > 0 else 1.0

        return {
            "original_text": text,
            "compressed_text": cleaned_summary,
            "cleaned_ratio": cleaned_ratio
        }


if __name__ == "__main__":
    original_prompt = "请详细描述你的计划，包括目标、步骤和时间安排。这个计划需要清晰的行动方案和可执行的步骤，以确保目标的实现。同时，尽量减少冗余描述，突出重点信息。"
    bart_compressor = BartTextCompressor(
        model_name="fnlp/bart-base-chinese",
        max_length=512,
        summary_max_length=20,
        summary_min_length=10,
        length_penalty=5.0,
        num_beams=4
    )

    result = bart_compressor.compress_text(original_prompt)
    print("Original text:", result["original_text"])
    print("Cleaned compressed text:", result["compressed_text"])
    print("Cleaned compression ratio:", result["cleaned_ratio"])
