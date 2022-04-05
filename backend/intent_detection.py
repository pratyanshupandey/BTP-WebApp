import torch
from model import Model


class IntentDetector:
    def __init__(self) -> None:
        self.word2int = {}
        with open("vocab/token_vocab") as f:
            words = f.read().split()
            for (i, word) in enumerate(words):
                self.word2int[word] = i
            self.word2int["<UNK>"] = len(words)

        with open("vocab/intent_vocab") as f:
            self.intent2int = f.read().split()
            self.intent2int = [word.split("_", 1)[1].title().replace("_", " ") for word in self.intent2int]

        self.model_inference = Model(word_vocab_size=len(self.word2int), intent_vocab_size=len(self.intent2int))
        self.model_inference.load_state_dict(torch.load("model.pkl"))
        self.model_inference.eval()
        print("Model Loaded")


    def map_token_sequence_to_ints(self, sequence):
        mapped = []
        for word in sequence:
            if word not in self.word2int:
                word = "<UNK>"
            mapped.append(self.word2int[word])

        return mapped
    

    def detect_intent(self, query_text: str) -> str:
        """
        Returns the intent behind the query_text.
        :param query_text: The query_text for which the model will detect the intent.
        :return: The intent
        """
        query_text = query_text.lower().strip()
        query_text = query_text.split()
        query_text = self.map_token_sequence_to_ints(query_text)
        query_text = torch.tensor(query_text, dtype=torch.long)


        pred = self.model_inference(query_text)
        pred_class = pred.argmax(dim=-1).numpy()[0]
        return self.intent2int[pred_class]


if __name__=="__main__":
    intent_detector = IntentDetector()
    print(intent_detector.detect_intent("Show me the flights from New York to Boston"))
    print(intent_detector.detect_intent("Show me the flights from New York to Boston"))
    