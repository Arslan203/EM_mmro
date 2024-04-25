from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    num, den = [], []
    for sents, preds in zip(reference, predicted):
        poss = sents.sure + [i for i in sents.possible if i not in sents.sure]
        num.append(0)
        den.append(len(preds))
        # print(f'poss:{poss}')
        # print(f'preds:{preds}')
        for i in preds:
            for j in poss:
                if i[0] == j[0] and i[1] == j[1]:
                    num[-1] += 1
        # print(f'intersect:{num[-1]}')
        # print(f'predicted:{den[-1]}')
    return sum(num), sum(den)


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    num, den = [], []
    for sents, preds in zip(reference, predicted):
        sure = sents.sure
        num.append(0)
        den.append(len(sure))
        for i in preds:
            for j in sure:
                if i[0] == j[0] and i[1] == j[1]:
                    num[-1] += 1
    return sum(num), sum(den)


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    num0, den0 = compute_precision(reference, predicted)
    num1, den1 = compute_recall(reference, predicted)
    return 1 - (num0 + num1) / (den0 + den1)
