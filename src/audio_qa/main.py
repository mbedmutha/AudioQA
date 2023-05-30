import pipeline
import argparse

parser = argparse.ArgumentParser(add_help=True)
# action='store_true',
parser.add_argument('--stt', default='stt_silero',
    help="Choose STT model. Currently supported 'stt_silero', 'stt_facebooks2t'")
parser.add_argument('--qa', action='store_true', default='qa_mobilebert',
    help="Choose QA model. Currently supported 'qa_mobilebert'")

args = parser.parse_args()

if __name__ == "__main__":
	custom_pipeline = pipeline.AudioQA(stt_class = args.stt, qa_class = args.qa)
	ans = custom_pipeline.inference(text_path=r"../../tests/unbiased_news.txt")
	print(ans)
