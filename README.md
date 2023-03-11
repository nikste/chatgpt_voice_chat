# voice chat with a large language model

crude version for conversational bot using openai-api (gpt-3) and tacotron2 (needs some stronger nvidia gpu)


## installation ##
* get openai api_key
* `touch openai_api_key`
* paste api key to file
* `conda env create -f environment.yml`
* `conda activate openai_voice_chat`
* `pip install -r requirements.txt` # note this might not be necessary but im lazy and its not a lot of work for you

## Usage
* `python voice_chat.py`
* in UI press button to record
* press button again to stop recording
* requests will be sent to openai api's to transcribe and generate response

