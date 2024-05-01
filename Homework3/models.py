from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple
# from tqdm import tqdm

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        # def _get_mask(par):
        #     mask = np.zeros((par.source_tokens.shape[0], par.target_tokens.shape[0]), dtype=bool)
        #     mask[par.source_tokens, :][:, par.target_tokens] = True
        #     return mask
        
        # self.masks = [_get_mask(par) for par in parallel_corpus]

        # print(self.translation_probs[parallel_corpus[0].source_tokens][:, parallel_corpus[0].target_tokens].shape)
        A = [self.translation_probs[par.source_tokens][:, par.target_tokens] / self.translation_probs[par.source_tokens][:, par.target_tokens].sum(axis=0, keepdims=True) for i, par in enumerate(parallel_corpus)]
        return A

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
            
        Tips: 
            1) Compute mathematical expectation with a constant
            2) It is preferred to write this computation with 1 cycle only
            
        """
        S = set(np.concatenate([par.source_tokens for par in parallel_corpus], axis=0))
        T = set(np.concatenate([par.target_tokens for par in parallel_corpus], axis=0))

        S, T = list(S), list(T)

        # S_d = {tok: idx for idx, tok in enumerate(sorted(S))}
        # T_d = {tok: idx for idx, tok in enumerate(sorted(T))}
        
        # C = np.zeros((len(S), len(T)))
        s = 0

        for post, par in zip(posteriors, parallel_corpus):
            src_tokens = np.unique(par.source_tokens, return_counts=True, return_index=True)
            trg_tokens = np.unique(par.target_tokens, return_counts=True, return_index=True)
            s += np.sum(np.log(self.translation_probs[src_tokens[0]][:, trg_tokens[0]]) * post[src_tokens[1]][:, trg_tokens[1]] * src_tokens[2][:, np.newaxis] * trg_tokens[2][np.newaxis, :], where=self.translation_probs[src_tokens[0]][:, trg_tokens[0]] != 0)
        #     for i in range(post.shape[0]):
        #         for j in range(post.shape[1]):
        #             C[S_d[src_tokens[i]], T_d[trg_tokens[j]]] += post[i, j]
        # return np.sum((np.log(np.clip(self.translation_probs[S][:, T], 1e-70, 1)) * C)[self.translation_probs[S][:, T] != 0])
        return s

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array], verbose=0):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        # S = set(np.concatenate([par.source_tokens for par in parallel_corpus], axis=0))
        # T = set(np.concatenate([par.target_tokens for par in parallel_corpus], axis=0))
        # if verbose:
        #     print(f'S, T built. S_shape:{len(S)}, T_shape:{len(T)}')
        
        # # A = np.zeros((len(S), len(T)))
        # T = list(T)
        # self.translation_probs[:, T] = 0
        self.translation_probs.fill(0)

        for post, par in zip(posteriors, parallel_corpus):
            src_tokens = np.unique(par.source_tokens, return_counts=True, return_index=True)
            trg_tokens = np.unique(par.target_tokens, return_counts=True, return_index=True)
            self.translation_probs[src_tokens[0]][:, trg_tokens[0]] += post[src_tokens[1]][:, trg_tokens[1]] * src_tokens[2][:, np.newaxis] * trg_tokens[2][np.newaxis, :]
            # for i in range(post.shape[0]):
            #     for j in range(post.shape[1]):
            #         self.translation_probs[src_tokens[i], trg_tokens[j]] += post[i, j]
        self.translation_probs /= self.translation_probs.sum(axis=1, keepdims=True)

        if verbose:
            print('Computing ELBO...')

        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus, verbose=0):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors, verbose=verbose)
            # print(np.unique(self.translation_probs, return_counts=True))
            # print(elbo)
            history.append(elbo)
        return history

    def align(self, sentences):
        # A = [self.translation_probs[par.source_tokens][:, par.target_tokens] / self.translation_probs[par.source_tokens][:, par.target_tokens].sum(axis=0, keepdims=True) for i, par in enumerate(sentences)]
        # aligns = [np.argmax(poster, axis=0) + 1 for poster in A]
        res = []
        for par in sentences:
            A = self.translation_probs[par.source_tokens][:, par.target_tokens] / self.translation_probs[par.source_tokens][:, par.target_tokens].sum(axis=0, keepdims=True)
            aligns_rel = enumerate(np.argmax(A, axis=0) + 1, 1)
            res.append([(j, i) for i, j in aligns_rel])
        return res


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        pass

    def _e_step(self, parallel_corpus):
        pass

    def _compute_elbo(self, parallel_corpus, posteriors):
        pass

    def _m_step(self, parallel_corpus, posteriors):
        pass
