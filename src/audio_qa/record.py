import numpy as np
import queue
import sys

import sounddevice as sd
import soundfile as sf

class Record:
	def __init__(self):
		self.q = queue.Queue()
		self.sr = 16000
		self.file_name = "../../tests/temp2.wav"
		self.voice_detected = False
		self.threshold = 2e-9
		self.energy = []
		print('Init Record')

	def callback(self, indata, frames, time, status):
		"""This is called (from a separate thread) for each audio block."""
		if status:
			print(status, file=sys.stderr)
		self.q.put(indata.copy())

	def record_audio(self):
		self.__init__()
		print(self.energy, self.voice_detected)
		
		with sf.SoundFile(self.file_name, mode='w', samplerate=self.sr, channels=1) as file:
			with sd.InputStream(samplerate=self.sr, callback=self.callback, channels=1):
				print('#' * 80)
				print('press Ctrl+C to stop the recording')
				print('#' * 80)
				while True:
					data = self.q.get()
					frame_energy = np.mean(data * data)
					if frame_energy >= 5e-9:
						self.voice_detected = True
						
					if len(self.energy) < 4:
						self.energy.append(frame_energy)
					else:
						self.energy = self.energy[1:] + [np.mean(frame_energy)]
						
					if self.voice_detected and sum(self.energy) <= self.threshold:
						break
						
#                     print(self.energy)
					file.write(data)
		print('Finished', self.energy, self.voice_detected)
		return self.file_name