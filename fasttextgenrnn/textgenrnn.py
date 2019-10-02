from keras.preprocessing.text import Tokenizer
from .model import textgenrnn_model
from .utils import *


class textgenrnn:
    META_TOKEN = '<s>'
    config = {
        'rnn_layers': 2,
        'rnn_size': 128,
        'rnn_bidirectional': False,
        'max_length': 40,
        'max_words': 10000,
        'dim_embeddings': 100,
        'word_level': False,
        'single_text': False
    }
    default_config = config.copy()

    def __init__(self, weights_path,
                 vocab_path,
                 config_path,
                 name="textgenrnn"):

        with open(config_path, 'r', encoding='utf8', errors='ignore') as json_file:
            self.config = json.load(json_file)

        self.config.update({'name': name})
        self.default_config.update({'name': name})

        with open(vocab_path, 'r', encoding='utf8', errors='ignore') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', char_level=True)
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.vocab) + 1
        self.model = textgenrnn_model(self.num_classes,
                                      cfg=self.config,
                                      weights_path=weights_path)
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    def generate(self, n=1, return_as_list=False, prefix=None,
                 temperature=0.5, max_gen_length=300):
        gen_texts = []
        for _ in range(n):
            gen_text = textgenrnn_generate(self.model,
                                           self.vocab,
                                           self.indices_char,
                                           prefix,
                                           temperature,
                                           self.config['max_length'],
                                           self.META_TOKEN,
                                           self.config['word_level'],
                                           self.config.get(
                                               'single_text', False),
                                           max_gen_length)
            if not return_as_list:
                print("{}\n".format(gen_text))
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts

    def save(self, weights_path="textgenrnn_weights_saved.hdf5"):
        self.model.save_weights(weights_path)

    def load(self, weights_path):
        self.model = textgenrnn_model(self.num_classes,
                                      cfg=self.config,
                                      weights_path=weights_path)

    def reset(self):
        self.config = self.default_config.copy()
        self.__init__(name=self.config['name'])
