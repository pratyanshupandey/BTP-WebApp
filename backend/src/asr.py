import whisper


class ASR:
    def __init__(self, device, model_name="small.en") -> None:
        self.model_name = model_name
        self.location = "whisper/"
        self.device = device

    def asr(self, audio: str) -> str:
        """
        Loading model during query only to save RAM.
        audio: The path to the audio file to be used.
        """
        asr_model = whisper.load_model(self.model_name, download_root=self.location).to(self.device)
        result = asr_model.transcribe(audio)
        return result["text"]


if __name__ == "__main__":
    import torch
    asr = ASR(torch.device("cpu"))
    
    