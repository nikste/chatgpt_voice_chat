# chatgpt_voice_chat

crude version for conversational bot using openai-api and tacotron2 (needs some stronger gpu ~10gb memory)


## installation ##
* get openai api_key
* `touch openai_api_key`
* paste api key to file
* `conda env create -f environment.yml`
* `conda activate openai_voice_chat`

## Usage
* `python voice_chat.py`
* in UI press upper button to record prompt
* press lower button to stop recording
* requests will be sent to openai api's to transcribe and generate response

# todo:
change Gui to pyqt (or something nicer)
