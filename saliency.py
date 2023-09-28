from allennlp.predictors import Predictor
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
import torch
from allennlp.interpret.saliency_interpreters import SimpleGradient, IntegratedGradient

SMALL_MODEL = 'gpt2'
MEDIUM_MODEL = 'https://storage.googleapis.com/allennlp/models/gpt2-345M-dump'

class Gpt2Predictor(Predictor):
    """
    The HuggingFace implementation of GPT-2 is not an AllenNLP model;
    however, our demo only expects an AllenNLP ``Predictor``. Accordingly,
    we implement a ``Predictor`` that wraps the HuggingFace GPT-2 implementation.
    """
    def __init__(self,
                 model_name: str = SMALL_MODEL,
                 cache_size: int = 0) -> None:
        """
        Each cache element is about 8MB, so size accordingly.
        """
        # Cache stores tuples, so default value is a tuple
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._model = GPT2LMHeadModel.from_pretrained(model_name)

        # The end of text marker.
        self.END_OF_TEXT = self.tokenizer.encoder["<|endoftext|>"]


    def predict_json(self, inputs: dict) -> dict:
        previous_str = inputs["previous"]
        next_str = inputs.get("next")
        topk = inputs.get("topk", 10)

        logits = self._predict(previous_str, next_str)
        probabilities = torch.nn.functional.softmax(logits, dim=0)

        best_logits, best_indices = logits.topk(topk)
        best_words = [self.tokenizer.decode([idx.item()])
                      for idx in best_indices]
        best_probabilities = probabilities[best_indices].tolist()
        
        next_str = best_words[0]

        return {
            "logits": best_logits.tolist(),
            "probabilities": best_probabilities,
            "words": best_words,
            "output": previous_str + (next_str or "")
        }

    def _predict(self, previous: str, next: str = None) -> torch.Tensor:

        past_logits, past = (None, None)

        # CASE 1: Previously seen input, no next
        if next is None and past is not None:
            return past_logits

        # CASE 2: Previously seen input, yes next
        elif past is not None:
            token_ids = self.tokenizer.encode(next)
        # CASE 3: Brand new input, no next
        elif next is None:
            token_ids = self.tokenizer.encode(previous)
        # CASE 4: Brand new input, yes next
        else:
            token_ids = self.tokenizer.encode(previous) + self.tokenizer.encode(next)

        inputs = torch.LongTensor([token_ids])

        result = self._model(inputs)
        logits = result.logits.squeeze()
        new_token_logit = logits[-1, :]
        key = previous if next is None else previous + next

        return new_token_logit

    def __getitem__(self, index: int) -> str:
        return self.tokenizer.decode([index])
    

def main():
    predictor = Gpt2Predictor()
    result = predictor.predict_json({"previous": "Toronto Raptors, who are currently tied for the league leader in wins"})
    interpreter = SimpleGradient(predictor)
    sample_text = "Toronto Raptors, who are currently tied for the league leader in wins"

    interpretation = interpreter.saliency_interpret_from_json({"previous": sample_text})    

if __name__ == "__main__":
    main()