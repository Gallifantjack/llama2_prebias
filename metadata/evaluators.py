from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textstat import flesch_kincaid_grade, gunning_fog
from textblob import TextBlob
import re

PROFANITIES = set(["prison", "hate"])


class Evaluators:
    # Moved this outside of the __init__ method to ensure it's compiled once
    PROFANITY_REGEX = re.compile(
        r"\b(?:" + "|".join(map(re.escape, PROFANITIES)) + r")\b", re.IGNORECASE
    )

    def __init__(self, decoded_text):
        if not decoded_text:
            # Default values for empty text
            self.text = []  # Tokenized form
            self.text_str = ""  # Detokenized form
            self.blob = None
        else:
            self.text = decoded_text.split()
            self.text_str = decoded_text
            self.blob = TextBlob(self.text_str)
        self.errors = []  # To collect errors

    def bleu_score(self, reference):
        try:
            smoothing = SmoothingFunction().method1
            return sentence_bleu(
                [reference.split()], self.text, smoothing_function=smoothing
            )
        except Exception as e:
            self.errors.append(f"Error computing BLEU score: {str(e)}")
            return 0.0

    def vocabulary_diversity(self):
        try:
            total_tokens = len(self.text)
            unique_tokens = len(set(self.text))
            return unique_tokens / total_tokens if total_tokens else 0
        except Exception as e:
            self.errors.append(f"Error computing vocabulary diversity: {str(e)}")
            return 0.0

    def subjectivity_score(self):
        try:
            return self.blob.sentiment.subjectivity
        except Exception as e:
            self.errors.append(f"Error computing subjectivity score: {str(e)}")
            return 0.0

    def sentiment_score(self):
        try:
            return self.blob.sentiment.polarity
        except Exception as e:
            self.errors.append(f"Error computing sentiment score: {str(e)}")
            return 0.0

    def flesch_kincaid_grade(self):
        try:
            return flesch_kincaid_grade(self.text_str)
        except Exception as e:
            self.errors.append(f"Error computing Flesch-Kincaid grade: {str(e)}")
            return 0.0

    def gunning_fog(self):
        try:
            return gunning_fog(self.text_str)
        except Exception as e:
            self.errors.append(f"Error computing Gunning Fog index: {str(e)}")
            return 0.0

    def profanity_check(self):
        try:
            profanity_count = len(self.PROFANITY_REGEX.findall(self.text_str))
            return profanity_count / len(self.text) if self.text else 0
        except Exception as e:
            self.errors.append(f"Error computing profanity check: {str(e)}")
            return 0.0

    def all_metrics(self, reference):
        metrics = {
            "bleu_score": self.bleu_score(reference),
            "flesch_kincaid_grade": self.flesch_kincaid_grade(),
            "gunning_fog": self.gunning_fog(),
            "vocabulary_diversity": self.vocabulary_diversity(),
            "subjectivity_score": self.subjectivity_score(),
            "sentiment_score": self.sentiment_score(),
            "profanity_check": self.profanity_check(),
        }
        if self.errors:
            print("\n".join(self.errors))
        return metrics


def evaluate_textual_metrics(decoded_text, reference):
    evaluator = Evaluators(decoded_text)
    metrics = evaluator.all_metrics(reference)
    return metrics
