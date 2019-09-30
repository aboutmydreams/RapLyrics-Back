from keras.models import Model
from keras.preprocessing import sequence
from keras import backend as K
import numpy as np
import json

def textgenrnn_sample(preds, temperature):
    '''
    Samples predicted probabilities of the next character to allow
    for the network to show "creativity."
    '''

    preds = np.asarray(preds).astype('float64')

    if temperature is None or temperature == 0.0:
        return np.argmax(preds)

    preds = np.log(preds + K.epsilon()) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    index = np.argmax(probas)

    # prevent function from being able to choose 0 (placeholder)
    # choose 2nd best index from preds
    if index == 0:
        index = np.argsort(preds)[-2]

    return index

def textgenrnn_generate(model, vocab,
                        indices_char, prefix=None, temperature=0.5,
                        maxlen=40, meta_token='<s>',
                        word_level=True,
                        single_text=False,
                        max_gen_length=300):
    '''
    Generates and returns a single text.
    '''
    # If generating word level, must add spaces around each punctuation.
    # https://stackoverflow.com/a/3645946/9314418
    #FIXME: punctuation processing may be removed depending on choice we make @LyricsVindicator
    #FIXME: in fact except for "'", all other punctuations may be removed when coming from api call
    #if word_level and prefix:
    #    punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
    #    prefix = re.sub('([{}])'.format(punct), r' \1 ', prefix)

    prefix = prefix.lower() if prefix else None

    if single_text:
        if word_level:
            text = prefix.split() if prefix else ['']
        else:
            text = list(prefix) if prefix else ['']
        max_gen_length += maxlen
    else:
        if word_level:
            text = [meta_token] + prefix.split() if prefix else [meta_token]
        else:
            text = [meta_token] + list(prefix) if prefix else [meta_token]
    next_char = ''

    if not isinstance(temperature, list):
        temperature = [temperature]

    if model_input_count(model) > 1:
        model = Model(inputs=model.input[0], outputs=model.output[1])

    while next_char != meta_token and len(text) < max_gen_length:
        encoded_text = textgenrnn_encode_sequence(text[-maxlen:],
                                                  vocab, maxlen)
        next_temperature = temperature[(len(text) - 1) % len(temperature)]
        next_index = textgenrnn_sample(
            model.predict(encoded_text, batch_size=1)[0],
            next_temperature)
        next_char = indices_char[next_index]
        text += [next_char]

    collapse_char = ' ' if word_level else ''

    # if single text, ignore sequences generated w/ padding
    # if not single text, strip the <s> meta_tokens
    if single_text:
        text = text[maxlen:]
    else:
        text = text[1:-1]

    text_joined = collapse_char.join(text)

    return text_joined


def textgenrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def textgenrnn_encode_cat(chars, vocab):
    '''
    One-hot encodes values at given chars efficiently by preallocating
    a zeros matrix.
    '''

    a = np.float32(np.zeros((len(chars), len(vocab) + 1)))
    rows, cols = zip(*[(i, vocab.get(char, 0))
                       for i, char in enumerate(chars)])
    a[rows, cols] = 1
    return a


def model_input_count(model):
    if isinstance(model.input, list):
        return len(model.input)
    else:   # is a Tensor
        return model.input.shape[0]

