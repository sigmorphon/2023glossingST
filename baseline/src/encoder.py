"""Defines a tokenizer that uses multiple distinct vocabularies"""
from typing import List
import torch
import pickle

special_chars = ["[UNK]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]


def create_vocab(sentences: List[List[str]], threshold=2, should_not_lower=False):
    """Creates a set of the unique words in a list of sentences, only including words that exceed the threshold"""
    all_words = dict()
    for sentence in sentences:
        if sentence is None:
            continue
        for word in sentence:
            # Grams should stay uppercase, stems should be lowered
            if not word.isupper() and not should_not_lower:
                word = word.lower()
            all_words[word] = all_words.get(word, 0) + 1

    all_words_list = []
    for word, count in all_words.items():
        if count >= threshold:
            all_words_list.append(word)

    return sorted(all_words_list)


class MultiVocabularyEncoder:
    """Encodes and decodes words to an integer representation"""

    def __init__(self, vocabularies: List[List[str]], segmented=False):
        """
        :param vocabularies: A list of vocabularies for the tokenizer
        """
        self.vocabularies = vocabularies
        self.all_vocab = special_chars + sum(self.vocabularies, [])
        self.segmented = segmented

        self.PAD_ID = special_chars.index("[PAD]")
        self.SEP_ID = special_chars.index("[SEP]")
        self.BOS_ID = special_chars.index("[BOS]")
        self.EOS_ID = special_chars.index("[EOS]")

    def encode_word(self, word: str, vocabulary_index: int, separate_vocab=False) -> int:
        """Converts a word to the integer encoding
        :param word: The word to encode
        :param vocabulary_index: The index of the vocabulary to use
        :param separate_vocab: If True, get the index of the word in just the specified vocabulary.
        :return: An integer encoding
        """
        if not word.isupper() and vocabulary_index < 2:
            # A bit of a hack, but we don't want to lowercase combined glosses, which should be in the third vocab
            word = word.lower()

        if word in special_chars:
            return special_chars.index(word)
        elif vocabulary_index < len(self.vocabularies):
            if word in self.vocabularies[vocabulary_index]:
                if separate_vocab:
                    return self.vocabularies[vocabulary_index].index(word) + len(special_chars)
                # Otherwise we need the combined index
                prior_vocab_padding = len(sum(self.vocabularies[:vocabulary_index], []))  # Sums the length of all preceding vocabularies
                return self.vocabularies[vocabulary_index].index(word) + prior_vocab_padding + len(special_chars)
            else:
                return 0
        else:
            # We got a bad vocabulary
            raise ValueError('Invalid vocabulary index.')

    def encode(self, sentence: List[str], vocabulary_index, separate_vocab=False) -> List[int]:
        """Encodes a sentence (a list of strings)"""
        return [self.encode_word(word, vocabulary_index=vocabulary_index, separate_vocab=separate_vocab) for word in sentence]

    def batch_decode(self, batch, from_vocabulary_index=None):
        """Decodes a batch of indices to the actual words
        :param batch: The batch of ids
        :param from_vocabulary_index: If provided, returns only words from the specified vocabulary. For instance, id=1 and vocab_index=2 will return the first word in the second vocabulary.
        """
        def decode(seq):
            if isinstance(seq, torch.Tensor):
                indices = seq.detach().cpu().tolist()
            else:
                indices = seq.tolist()
            if from_vocabulary_index is not None:
                decode_vocab = self.vocabularies[from_vocabulary_index]
                return ['[UNK]' if index == 0 else decode_vocab[index-len(special_chars)] for index in indices if (index >= len(special_chars) or index == 0)]
            return ['[UNK]' if index == 0 else self.all_vocab[index] for index in indices if (index >= len(special_chars) or index == 0)]

        return [decode(seq) for seq in batch]

    def vocab_size(self):
        return len(self.all_vocab)

    def save(self):
        """Saves the encoder to a file"""
        with open('encoder_data.pkl', 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)


def load_encoder(path) -> MultiVocabularyEncoder:
    with open(path, 'rb') as inp:
        return pickle.load(inp)