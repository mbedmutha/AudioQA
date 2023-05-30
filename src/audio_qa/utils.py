import torchaudio

def read_audio(path: str,
			   target_sr: int = 16000):

	wav, sr = torchaudio.load(path)

	if wav.size(0) > 1:
		wav = wav.mean(dim=0, keepdim=True)

	if sr != target_sr:
		transform = torchaudio.transforms.Resample(orig_freq=sr,
												   new_freq=target_sr)
		wav = transform(wav)
		sr = target_sr

	assert sr == target_sr
	return wav.squeeze(0)

def read_text(text_path: str):
	assert text_path # Check if not none?
	with open(text_path, encoding='utf-8') as f:
		lines = f.readlines()
		text = "".join(lines).replace("\n","")
	return text