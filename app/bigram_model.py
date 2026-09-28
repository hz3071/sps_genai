from collections import Counter, defaultdict
import random
import re


class BigramModel:
    def __init__(self, corpus):
        text = " ".join(corpus)
        self.bigram_probs = self.analyze_bigrams(text)

    def simple_tokenizer(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def analyze_bigrams(self, text):
        words = self.simple_tokenizer(text)

        bigram_counts = Counter(zip(words, words[1:]))
        word_counts = Counter(words)

        bigram_probs = defaultdict(dict)

        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / word_counts[word1]

        return bigram_probs

    def generate_text(self, start_word, length):
        words = [start_word]
        current_word = start_word.lower()

        for i in range(length - 1):
            if current_word not in self.bigram_probs:
                break

            next_words = list(self.bigram_probs[current_word].keys())
            probabilities = list(self.bigram_probs[current_word].values())

            next_word = random.choices(
                next_words,
                weights=probabilities
            )[0]

            words.append(next_word)
            current_word = next_word

        return " ".join(words)