import re
from vocab import load_vocab, vocab_fraction

VOCAB = load_vocab()


def reward_vocab(
    completions: list[str],
    answer_aliases: list[list[str]],
    **kwargs,
) -> list[float]:
    # Reward fluent vocabulary ONLY when the answer is actually correct,
    # so the model can't farm this reward with wrong / refusal / padding text.
    return [
        vocab_fraction(c, VOCAB) * 0.4 if _correctness_score(c, a) > 0 else 0.0
        for c, a in zip(completions, answer_aliases)
    ]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _correctness_score(completion: str, aliases: list[str]) -> float:
    norm_c = _normalise(completion)
    score = 0.0
    for alias in aliases:
        norm_a = _normalise(alias)
        if not norm_a:
            continue
        if norm_a == norm_c:
            return 1.0
        if norm_a in norm_c:
            score = max(score, 0.5)
    return score


def reward_correctness(
    completions: list[str],
    answer_aliases: list[list[str]],
    **kwargs,
) -> list[float]:
    return [_correctness_score(c, a) * 1.6 for c, a in zip(completions, answer_aliases)]

def reward_no_repetition(completions: list[str], **kwargs) -> list[float]:
    scores = []
    for c in completions:
        words = _normalise(c).split()
        if len(words) < 4:
            scores.append(0.0)
            continue
        trigrams = list(zip(words, words[1:], words[2:]))
        distinct = len(set(trigrams)) / len(trigrams)
        scores.append((distinct - 1.0) * 0.5) # 0 if no repetition, to -0.5 if many repeated
    return scores