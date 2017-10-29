
#! /usr/bin/python
# -*- coding: utf-8 -*-


"""Rank sentences based on cosine similarity and a query."""


from argparse import ArgumentParser
import numpy as np
from numpy import dot
from numpy.linalg import norm

def get_sentences(file_path):
    """Return a list of sentences from a file."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().splitlines()


def get_top_k_words(sentences, k):
    """Return the k most frequent words as a list."""
    dictionary = {}
    for sentence in sentences:
        words = sentence.strip(" |.|!|?").replace(","," ").replace(":"," ").replace(";"," ").split(" ")
        for word in words:
            if dictionary.get(word) == None:
                dictionary[word] = 1
            else :
                dictionary[word] = dictionary.get(word) + 1
    topk = sorted(dictionary.items(), key=lambda x: x[1], reverse = True)[1:k+1]
    elements = []
    for e in topk:
        elements.append(e[0])
    return elements

def encode(sentence, vocabulary):
     """Return a vector encoding the sentence."""
     words = sentence.strip(" |.|!|?").replace(","," ").replace(":"," ").replace(";"," ").split(" ")
     counts = np.zeros(len(vocabulary)).tolist()
     for word in words:
         if word in vocabulary:
             counts[vocabulary.index(word)] = counts[vocabulary.index(word)] + 1
     return np.asarray(counts)


def get_top_l_sentences(sentences, query, vocabulary, l):
    """
    For every sentence in "sentences", calculate the similarity to the query.
    Sort the sentences by their similarities to the query.

    Return the top-l most similar sentences as a list of tuples of the form
    (similarity, sentence).
    """
    sim_sentences = []
    vec_query = encode(query,vocabulary)
    for sentence in sentences:
        vec_sentence = encode(sentence,vocabulary)
        cos_sim = cosine_sim(vec_query,vec_sentence)
        sim_sentences.append((cos_sim,sentence))
    topl = sorted(sim_sentences, key=lambda x : x[0],reverse = True)[:l]
    
    return topl


def cosine_sim(u, v):
    """Return the cosine similarity of u and v."""
    if norm(u)*norm(v) == 0:
        sim = 0
    else:
        sim = dot(u, v)/(norm(u)*norm(v))
    return sim


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='An input file containing sentences, one per line')
    arg_parser.add_argument('QUERY', help='The query sentence')
    arg_parser.add_argument('-k', type=int, default=1000,
                            help='How many of the most frequent words to consider')
    arg_parser.add_argument('-l', type=int, default=10, help='How many sentences to return')
    args = arg_parser.parse_args()
    
    
    sentences = get_sentences(args.INPUT_FILE)
    top_k_words = get_top_k_words(sentences, args.k)
    query = args.QUERY.lower()

    print('using vocabulary: {}\n'.format(top_k_words))
    print('using query: {}\n'.format(query))

    # suppress numpy's "divide by 0" warning.
    # this is fine since we consider a zero-vector to be dissimilar to other vectors
    with np.errstate(invalid='ignore'):
        result = get_top_l_sentences(sentences, query, top_k_words, args.l)

    print('result:')
    for sim, sentence in result:
        print('{:.5f}\t{}'.format(sim, sentence))


if __name__ == '__main__':
    main()
