import os
import re
import torch
import numpy as np
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="API_KEY"  
)


class ChineseTextCompressor:
    def __init__(self, model_name: str = "uer/gpt2-chinese-cluecorpussmall"):
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add pad_token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Keywords
        self.keywords = {"描述", "计划", "目标", "步骤", "时间安排", "实现", "行动方案"}

        # Compression parameters
        self.importance_threshold = 0.05
        # Lower compression_level means more content is retained
        self.compression_level = 0.3

    def calculate_perplexity(self, text: str) -> float:
        if not text.strip():
            return 0.0
        if len(text) < 2:
            text = text + text

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            if loss is not None:
                return torch.exp(loss).item()
        return 0.0

    def segment_text(self, text: str) -> List[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def sentence_split(self, text: str) -> List[str]:
        sentences = re.split('[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]

    def token_level_filter(self, sentence: str) -> str:
        words = self.segment_text(sentence)
        if not words:
            return sentence

        # If sentence contains a keyword, keep it unchanged
        if any(w in self.keywords for w in words):
            return sentence
        
        perplexities = [self.calculate_perplexity(w) for w in words]
        if not perplexities:
            return sentence

        median_pp = np.median(perplexities)
        threshold = median_pp * 0.2

        filtered_words = [w for w, pp in zip(words, perplexities) if pp >= threshold]
        
        # If filtered result is too short, revert to original
        if len(filtered_words) < max(1, len(words)*0.4):
            return sentence
        
        return ''.join(filtered_words)

    def phrase_level_compression(self, sentences: List[str]) -> List[str]:
        all_sent_words = [' '.join(self.segment_text(s)) for s in sentences if s.strip()]
        if not all_sent_words:
            return sentences
        
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(all_sent_words)
        vocab = vectorizer.get_feature_names_out()

        compressed_sentences = []
        for idx, sentence in enumerate(sentences):
            words = self.segment_text(sentence)
            if not words:
                compressed_sentences.append(sentence)
                continue

            # If sentence contains a keyword, keep it unchanged
            if any(w in self.keywords for w in words):
                compressed_sentences.append(sentence)
                continue

            tfidf_scores = tfidf_matrix[idx].toarray().flatten()
            
            word2tfidf = {}
            for w in words:
                if w in vocab:
                    word_index = np.where(vocab == w)[0]
                    score = tfidf_scores[word_index[0]] if len(word_index) > 0 else 0.0
                else:
                    score = 0.0
                word2tfidf[w] = score

            avg_tfidf = np.mean(list(word2tfidf.values())) if word2tfidf else 0.0
            retained_words = [w for w in words if word2tfidf[w] >= avg_tfidf * 0.4]

            if len(retained_words) < max(1, len(words)*0.3):
                retained_words = words
            
            compressed_sentences.append(''.join(retained_words))
        
        return compressed_sentences

    def sentence_level_selection(self, sentences: List[str], max_ratio=0.5) -> List[str]:
        scored_sents = []
        for s in sentences:
            pp = self.calculate_perplexity(s)
            words = self.segment_text(s)
            keyword_count = sum([1 for w in words if w in self.keywords])
            # Scoring: perplexity + keyword importance
            score = pp + (keyword_count * 5.0)
            scored_sents.append((s, score))
        
        scored_sents.sort(key=lambda x: x[1], reverse=True)
        retain_count = max(1, int(len(scored_sents)*max_ratio))
        selected = scored_sents[:retain_count]

        # Maintain original order
        original_order = {s:i for i,s in enumerate(sentences)}
        selected.sort(key=lambda x: original_order[x[0]])
        
        return [x[0] for x in selected]

    def compress_text(self, text: str) -> str:
        sentences = self.sentence_split(text)
        if not sentences:
            return ""

        # Token-level compression
        token_filtered_sents = [self.token_level_filter(s) for s in sentences if s.strip()]

        # Phrase-level compression
        phrase_filtered_sents = self.phrase_level_compression(token_filtered_sents)

        # Sentence-level selection
        final_sents = self.sentence_level_selection(phrase_filtered_sents, max_ratio=(1-self.compression_level))

        return '。'.join(final_sents) + '。' if final_sents else ""


def clarify_text(client: OpenAI, text: str, model: str, prompt_prefix: str) -> str:
    """
    Call GPT to clarify rhetorical, metaphorical or idiomatic expressions.
    """
    prompt = f"{prompt_prefix}\n{text}"
    response = client.chat.completions.create(
        messages=[
          {"role": "system", "content": "You are a helpful assistant who excels at understanding and rewriting Chinese text."},
          {"role": "user", "content": prompt}
        ],
        model=model,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def process_clean_before_compress(original_text: str, client: OpenAI, compressor: ChineseTextCompressor, model: str = "gpt-4o"):
    """
    Process A (Clean Before Compress):
    1. Use GPT once to clarify the original text.
    2. Compress the clarified text.
    """
    prompt_prefix = """Please rewrite the following text by explaining any rhetorical, metaphorical, or idiomatic expressions 
in a direct and easily understood manner. Be faithful to the original meaning and make it easily understandable. 
You may rewrite phrases, but do not shorten the overall content."""
    
    clarified_text = clarify_text(client, original_text, model, prompt_prefix)
    compressed_text = compressor.compress_text(clarified_text)
    return clarified_text, compressed_text


def process_compress_before_clean(original_text: str, client: OpenAI, compressor: ChineseTextCompressor, model: str = "gpt-4o"):
    """
    Process B (Compress Before Clean):
    1. Compress the original text first.
    2. Then use GPT once to clarify any rhetorical, metaphorical, or idiomatic expressions in the compressed text.
       Conditions for B process:
       - If such expressions exist, rewrite them without removing or shortening any original sentence. 
         Any rewritten segment must not be shorter than the original segment.
       - If no such expressions exist, return the compressed text unchanged.
    """
    compressed_text = compressor.compress_text(original_text)
    prompt_prefix = """Please review the following compressed text. If it contains any rhetorical, metaphorical, or idiomatic expressions,
rewrite those parts to be direct and easily understood without removing any original sentences or shortening them.
If you need to clarify a phrase, your rewritten version must not be shorter than the original phrase. 
If no such expressions exist, return the text unchanged."""
    
    clarified_compressed_text = clarify_text(client, compressed_text, model, prompt_prefix)
    return compressed_text, clarified_compressed_text


if __name__ == "__main__":
    original_text = """请详细描述你的计划，包括目标、步骤和时间安排。
                       这个计划需要清晰的行动方案和可执行的步骤，以确保目标的实现。
                       同时，尽量减少冗余描述，突出重点信息。
                       别再‘纸上谈兵’，要实实在在地行动。"""

    compressor = ChineseTextCompressor()

    # Process A: Clean before compress
    clarified_text_A, compressed_text_A = process_clean_before_compress(original_text, client, compressor, model="gpt-4o")

    # Process B: Compress before clean
    compressed_text_B, clarified_compressed_text_B = process_compress_before_clean(original_text, client, compressor, model="gpt-4o")

    print("========== Process A (Clean Before Compress) ==========")
    print("Clarified Text (before compression):", clarified_text_A)
    print("Compressed Text:", compressed_text_A)

    print("\n========== Process B (Compress Before Clean) ==========")
    print("Compressed Text (before cleaning):", compressed_text_B)
    print("Clarified Compressed Text:", clarified_compressed_text_B)
