import torch

from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForQuestionAnswering
from utils import read_text, read_audio

class QA:
	def __init__(self):
		pass

	def question_answer(self):
		pass

class MobileBERT(QA):
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/mobilebert-uncased-finetuned-squadv2") #"google/mobilebert-uncased")
		self.model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/mobilebert-uncased-finetuned-squadv2") #"google/mobilebert-uncased")
		
	def question_answer(self, question, text=None, text_path=None):
		if text == None:
			if text_path == None:
				return "Cannot find text"
			else:
				text = read_text(text_path)
			
		#tokenize question and text as a pair
		input_ids = self.tokenizer.encode(question, text)

		#string version of tokenized ids
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

		#segment IDs
		#first occurence of [SEP] token
		sep_idx = input_ids.index(self.tokenizer.sep_token_id)    #number of tokens in segment A (question)
		num_seg_a = sep_idx+1    #number of tokens in segment B (text)
		num_seg_b = len(input_ids) - num_seg_a

		#list of 0s and 1s for segment embeddings
		segment_ids = [0]*num_seg_a + [1]*num_seg_b
		assert len(segment_ids) == len(input_ids)

		#model output using input_ids and segment_ids
		self.output = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

		#reconstructing the answer
		self.answer = ""
		answer_start = torch.argmax(self.output.start_logits)
		answer_end = torch.argmax(self.output.end_logits)
		if answer_end >= answer_start:
			self.answer = tokens[answer_start]
			for i in range(answer_start+1, answer_end+1):
				if tokens[i][0:2] == "##":
					self.answer += tokens[i][2:]
				else:
					self.answer += " " + tokens[i]

		if self.answer.startswith("[CLS]") or self.answer=="":
			self.answer = "Unable to find the answer to your question."

#         print("\nPredicted answer:\n{}".format(self.answer.capitalize()))
		
		return self.answer.capitalize()