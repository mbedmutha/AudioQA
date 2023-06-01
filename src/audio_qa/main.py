import pipeline
import argparse
import pyttsx3



parser = argparse.ArgumentParser(add_help=True)
# action='store_true',
parser.add_argument('--stt', default='stt_silero',
    help="Choose STT model. Currently supported 'stt_silero', 'stt_facebooks2t'")
parser.add_argument('--qa', action='store_true', default='qa_mobilebert',
    help="Choose QA model. Currently supported 'qa_mobilebert'")
parser.add_argument('--no-speaker', action='store_true', default=False,
	help="Boolean variable for speech output")

args = parser.parse_args()

if __name__ == "__main__":
	custom_pipeline = pipeline.AudioQA(stt_class = args.stt, qa_class = args.qa)
	engine = pyttsx3.init()

	cont = "Yes"
	while cont == "Yes":
		ans = custom_pipeline.inference(text_path=r"../../tests/unbiased_news.txt")
		print(ans)
		engine.say(ans)
		engine.runAndWait()
		cont = input("Do you want to ask another question? 'Yes' to continue ")
