# Audio Question Answering

AudioQA is a pipeline to answer questions based on a known context (user-fed) by speech or text. Users can speak their question and the system will return the answer as text. Future work will include TTS support as well.

## Arguments
|Argument|Info|Notes|
|--help, -h| Help for passing params ||
|--stt| Chooses stt model. Default silero| Supports silero, facebooks2t|
|--qa| Chooses model for question answering | Supports MobileBERT |


Models: Torch Hub and Transformers/HF
Package Development: https://packaging.python.org/en/latest/tutorials/packaging-projects/
