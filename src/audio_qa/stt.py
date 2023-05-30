# Refer: https://pytorch.org/hub/snakers4_silero-models_stt/
# https://github.com/snakers4/silero-models/blob/21ed251aa28d023db96a8fdaaf5b22877bc8c0af/src/silero/utils.py#L40

import torch
import zipfile
import torchaudio
from glob import glob
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

class STT:
	def __init__(self, device='cpu'):
		self.device = torch.device(device)

	def inference(self, temp_path=None, wav_data=None):
		# Should return self.text
		pass

class Silero(STT):
	def __init__(self, device='cpu'):
		super().__init__()
		self.device = torch.device(device)  # gpu also works, but our models are fast enough for CPU
		self.model, self.decoder, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-models', 
											   model='silero_stt', language='en', # also available 'de', 'es'
											   device=self.device)
		(self.read_batch, self.split_into_batches, self.read_audio, self.prepare_model_input) = self.utils  # see function signature for details
			
	def inference(self, temp_path=None, wav_data=None):
#         temp_path = r"C:\Users\manas\Downloads\manas_smartphone_speech.wav"
		if wav_data == None:
			if temp_path == None:
				print("Cannot sense audio")
				return
			else:
				wav_data = self.read_audio(temp_path)
		
		self.input_tensor = wav_data.unsqueeze(0)
		del wav_data
		
		self.output = self.model(self.input_tensor)
		self.text = self.decoder(self.output.cpu()[0])
		
		return self.text

class FacebookS2T(STT):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("facebook/s2t-small-librispeech-asr")
       
    def inference(self, temp_path=None, wav_data=None):
#         temp_path = r"C:\Users\manas\Downloads\manas_smartphone_speech.wav"
        if wav_data == None:
            if temp_path == None:
                print("Cannot sense audio")
                return
            else:
                wav_data, fs = librosa.load(r"../../tests/temp2.wav", sr=16000) #C:\Users\manas\Downloads\manas_smartphone_speech.wav")
        
        inputs = processor(wav_data, sampling_rate=fs, return_tensors="pt")
        
        generated_ids = self.model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        self.text = transcription[0][:512]
        return self.text

        
