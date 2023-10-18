from nltk.translate.bleu_score import sentence_bleu
from collections import Counter


class Evaluators:
    def __init__(self, decoded_text):
        self.text = decoded_text.split()

    def bleu_score(self, reference):
        return sentence_bleu([reference.split()], self.text)

    def word_perplexity(self):
        word_freq = Counter(self.text)
        perplexity = sum(
            [
                -word_freq[word] / len(self.text) * (word_freq[word] / len(self.text))
                for word in set(self.text)
            ]
        )
        return perplexity

    def prevalence_of_word(self, word):
        if not self.text:  # If text is empty
            return 0
        return self.text.count(word) / len(self.text)

    def bias_flags(self):
        biases = ["Cat", "Dog"]
        return sum([self.text.count(bias) for bias in biases])

    def sentence_length(self):
        return len(self.text)

    def co_occurrence(self, word1, word2, window_size=5):
        count = 0
        for i, word in enumerate(self.text):
            if (
                word == word1
                and i + window_size < len(self.text)
                and word2 in self.text[i + 1 : i + window_size + 1]
            ):
                count += 1
            elif (
                word == word2
                and i + window_size < len(self.text)
                and word1 in self.text[i + 1 : i + window_size + 1]
            ):
                count += 1
        return count

    def all_metrics(self, reference):
        return {
            "bleu_score": self.bleu_score(reference),
            "prevalence_the": self.prevalence_of_word("the"),
            "prevalence_and": self.prevalence_of_word("and"),
            "sentence_length": self.sentence_length(),
            "word_perplexity": self.word_perplexity(),
            "bias_flags": self.bias_flags(),
            "co_occurrence_example": self.co_occurrence("Lilly", "Dog"),
        }


def evaluate_textual_metrics(decoded_text, reference):
    evaluator = Evaluators(decoded_text)
    metrics = evaluator.all_metrics(reference)
    return metrics
