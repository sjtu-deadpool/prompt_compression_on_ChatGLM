import json
import os
from mycompression_en import ChineseTextCompressor
from llmlingua_en import LLMLinguaTextCompressor
from bart_en import BartTextCompressor

# Define file paths
input_files = {
    "complicated": "long_instructions_complecated.json"
}
output_files = {
    "plain": {
        "mycompression_A_plain.json": "mycompression_en_A",
        "mycompression_B_plain.json": "mycompression_en_B",
        "llmlingua_plain.json": "llmlingua_en",
        "bart_plain.json": "bart_en"
    },
    "complicated": {
        "mycompression_A_complecated.json": "mycompression_en_A",
        "mycompression_B_complecated.json": "mycompression_en_B",
        "llmlingua_complecated.json": "llmlingua_en",
        "bart_complecated.json": "bart_en"
    }
}

# Initialize compressors
chinese_compressor = ChineseTextCompressor()
llmlingua_compressor = LLMLinguaTextCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    compression_rate=0.5
)
bart_compressor = BartTextCompressor(
    model_name="fnlp/bart-base-chinese",
    max_length=512,
    summary_max_length=50,
    summary_min_length=25
)

def mycompress_clean_before_compress(instruction, compressor):
    """Clean before compress using mycompression_en."""
    clarified_text = compressor.compress_text(instruction)
    return clarified_text

def mycompress_compress_before_clean(instruction, compressor):
    """Compress before clean using mycompression_en."""
    compressed_text = compressor.compress_text(instruction)
    return compressed_text

# Function to process instructions and save to file
def process_instructions(input_file, output_files):
    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize result dictionaries
    results = {name: [] for name in output_files}

    for entry in data:
        instruction = entry["instruction"]
        compressed_results = {}

        # Apply mycompression_en with two modes
        compressed_results["mycompression_en_A"] = mycompress_clean_before_compress(instruction, chinese_compressor)
        compressed_results["mycompression_en_B"] = mycompress_compress_before_clean(instruction, chinese_compressor)

        # Apply llmlingua and bart
        compressed_results["llmlingua_en"] = llmlingua_compressor.compress_text(instruction)["compressed_text"]
        compressed_results["bart_en"] = bart_compressor.compress_text(instruction)["compressed_text"]

        # Append results to respective output lists
        for file_name, method_name in output_files.items():
            new_entry = {
                "instruction": compressed_results[method_name],
                "input": entry["input"],
                "output": entry["output"]
            }
            results[file_name].append(new_entry)

    # Save results to corresponding files
    for file_name, content in results.items():
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {file_name}")

if __name__ == "__main__":
    for version, input_file in input_files.items():
        print(f"Processing {version} instructions...")
        process_instructions(input_file, output_files[version])
