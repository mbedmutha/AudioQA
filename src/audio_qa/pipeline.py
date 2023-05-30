import llm_qa
import record
import stt
from utils import read_text

class AudioQA:
	def __init__(self, 
			record_class="rec",
			stt_class = "stt_silero",
			qa_class = "qa_mobilebert"
		):
		if record_class == "rec":
			self.rec = record.Record()

		if stt_class == "stt_silero":
			self.stt = stt.Silero()
		if stt_class == "stt_facebooks2t":
			self.stt = stt.FacebookS2T()

		if qa_class == "qa_mobilebert":
			self.qa = llm_qa.MobileBERT()
		
	def inference(self, text_path = None, text = None):
		self.rec.__init__()
		if text == None:
			if text_path == None:
				return "Cannot find any text"
			else:
				text = read_text(text_path)

		self.file_name = self.rec.record_audio()
		# self.file_name = '../../tests/temp2.wav'
		self.question = self.stt.inference(temp_path = self.file_name) #"where was the auction held"
		print(f'Detected Speech: {self.question}')
		self.answer = self.qa.question_answer(self.question, text)
		return self.answer