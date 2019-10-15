# RapLyrics-Back

Generate genuine AI-powered rap lyrics.

Fast lyricist - raplyrics as a cloud function.
This branch holds the code ran by the Google cloud function `fast-lyricist`.

### Prerequisites:
Having trained the model and having the model parameters (config, vocab, and weights) available under a bucket `cloud_function_asset/tmp`. 

### Deploying

Use a `Python 3.7` Runtime. 

Push the files and directory :

- `main.py`
- `fasttextgenrnn/`
- `requirements.txt`

to the cloud function.  
