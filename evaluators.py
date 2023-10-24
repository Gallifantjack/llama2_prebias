from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textstat import flesch_kincaid_grade, gunning_fog
from textblob import TextBlob

PROFANITIES = set(["prison", "hate"])


class Evaluators:
    def __init__(self, decoded_text):
        if not decoded_text:
            # Default values for empty text
            print("Empty text")
            self.text = []
        else:
            self.text = decoded_text.split()

    def bleu_score(self, reference):
        try:
            smoothing = SmoothingFunction().method1
            return sentence_bleu(
                [reference.split()], self.text, smoothing_function=smoothing
            )
        except Exception as e:
            print(f"Error computing BLEU score: {str(e)}")  # Log the error
            return 0.0

    def flesch_kincaid_grade(self):
        try:
            return flesch_kincaid_grade(" ".join(self.text))
        except Exception as e:
            print(f"Error computing Flesch-Kincaid grade: {str(e)}")  # Log the error
            return 0.0

    def gunning_fog(self):
        try:
            return gunning_fog(" ".join(self.text))
        except Exception as e:
            print(f"Error computing Gunning Fog index: {str(e)}")  # Log the error
            return 0.0

    def vocabulary_diversity(self):
        try:
            total_tokens = len(self.text)
            unique_tokens = len(set(self.text))
            return unique_tokens / total_tokens if total_tokens else 0
        except Exception as e:
            print(f"Error computing vocabulary diversity: {str(e)}")  # Log the error
            return 0.0

    def subjectivity_score(self):
        try:
            blob = TextBlob(" ".join(self.text))
            return blob.sentiment.subjectivity
        except Exception as e:
            print(f"Error computing subjectivity score: {str(e)}")  # Log the error
            return 0.0

    def sentiment_score(self):
        try:
            blob = TextBlob(" ".join(self.text))
            return blob.sentiment.polarity
        except Exception as e:
            print(f"Error computing sentiment score: {str(e)}")  # Log the error
            return 0.0

    def profanity_check(self):
        try:
            return (
                sum(1 for word in self.text if word.lower() in PROFANITIES)
                / len(self.text)
                if self.text
                else 0
            )
        except Exception as e:
            print(f"Error computing profanity check: {str(e)}")  # Log the error
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
        return metrics


def evaluate_textual_metrics(decoded_text, reference):
    evaluator = Evaluators(decoded_text)
    metrics = evaluator.all_metrics(reference)
    return metrics
