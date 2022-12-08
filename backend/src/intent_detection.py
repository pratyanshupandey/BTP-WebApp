from model import IntentModel
from torch import load
from transformers import BertTokenizer
import json
import torch
from asr import ASR


class IntentDetector:
    def __init__(self, device) -> None:
        self.device = device
        model_paths = {
            "Assistant_amazon": "models/massive/",
            "Assistant_snips": "models/snips/",
            "Flight_atis": "models/atis/"
        }
        self.nn_models = {}
        for k, v in model_paths.items():
            self.nn_models[k] = IntentInference(v, device)

        self.asr = ASR(device, model_name="small.en")


    def get_intent_from_audio(self, audio_path : str ,domain : str):
        transcript = self.asr.asr(audio_path)
        intent = self.get_intent_from_text(transcript, domain)
        return transcript, intent


    def get_intent_from_text(self, query, domain):
        return self.nn_models[domain].detect_intent(query)


class IntentInference:
    def __init__(self, dir_path, device) -> None:
        self.dir_path = dir_path
        file = open(dir_path + "intent_classes.json")
        self.intents = [" ".join(intent.split("_")) for intent in json.load(file)["classes"]]
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    

    def detect_intent(self, query_text: str) -> str:
        """
        Returns the intent behind the query_text.
        :param query_text: The query_text for which the model will detect the intent.
        :return: The intent
        """

        # loading model only during query to save ram
        model = IntentModel(len(self.intents))
        model.load_state_dict(load(self.dir_path + "model.pt", map_location=self.device))
        model = model.to(self.device)
        model.eval()

        query_text = query_text.lower().strip()
        ids = self.tokenizer.batch_encode_plus([query_text], 
                                                padding="longest", 
                                                max_length=32, 
                                                truncation=True, 
                                                return_tensors="pt")
        ids = ids.to(self.device)

        pred_class = model(ids).tolist()[0]
        return self.intents[pred_class]



if __name__=="__main__":
    intent_detector = IntentDetector(torch.device('cpu'))
    print(intent_detector.get_intent_from_text("Show me the flights from New York to Boston", "Flight_atis"))
    