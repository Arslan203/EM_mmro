from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    def str_pairs2tuple(text):
        if text is None:
            return []
        return list(map(lambda x: tuple(np.fromstring(x, sep='-', dtype=np.int32)), text.split()))
    
    # fixing & chars
    with open(filename, 'r') as f:
        content = f.read()
    content = content.replace('&', '&amp;')

    # parser = ET.XMLParser(encoding='utf-8')
    # tree = ET.parse(filename, parser=parser)
    # print(content)
    root = ET.fromstring(content)
    sentence_pairs = []
    alignments = []
    for child in root:
        tmp_dir = {ch.tag: ch.text for ch in child}
        sentence_pairs.append(SentencePair(source=tmp_dir['english'].split(), target=tmp_dir['czech'].split()))
        alignments.append(LabeledAlignment(sure=str_pairs2tuple(tmp_dir['sure']), possible=str_pairs2tuple(tmp_dir['possible'])))

    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    count_src, count_trg = defaultdict(int), defaultdict(int)
    for sents in sentence_pairs:
        for src in sents.source:
            count_src[src] += 1
        for trg in sents.target:
            count_trg[trg] += 1
    # handling freq_cutoff
    if freq_cutoff is not None:
        src_tokens = sorted(count_src.items(), key=lambda x: x[1], reverse=True)[:freq_cutoff]
        src_tokens = [i[0] for i in src_tokens]

        trg_tokens = sorted(count_trg.items(), key=lambda x: x[1], reverse=True)[:freq_cutoff]
        trg_tokens = [i[0] for i in trg_tokens]
    else:
        src_tokens = count_src.keys()
        trg_tokens = count_trg.keys()
    
    source_dict = {src: i for i, src in enumerate(src_tokens)}
    target_dict = {trg: i for i, trg in enumerate(trg_tokens)}
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    for sents in sentence_pairs:
        src_tokens = np.array(list(map(lambda x: source_dict.get(x, -1), sents.source)), dtype=np.int32)
        trg_tokens = np.array(list(map(lambda x: target_dict.get(x, -1), sents.target)), dtype=np.int32)
        if np.sum(src_tokens == -1) + np.sum(trg_tokens == -1):
            # pair contain not popular token
            continue
        tokenized_sentence_pairs.append(TokenizedSentencePair(source_tokens=src_tokens, target_tokens=trg_tokens))
    
    return tokenized_sentence_pairs
