from feature_utils import clean, get_tokenized_lemmas
from tqdm import tqdm

refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract',
        'notwithstanding',
        'although'
        'incorrect',
        'wrong',
        'lie',
        'fabricate',
        'unsupported',
        'untrue',
        'refute',
        'rebut',
        'disprove',
        'contradict',
        'falsify',
        'deceive',
        'invalid'
]

refuting_phrases = [
        'not true',
        'wrong on both counts'
        'actually incorrect',
        'materially false',
        'false representation',
        'failing to',
        'no longer',
        'simply not',
        'in spite',
        'have doubts',
        'some doubts',
        'certain doubts',
        'question marks'
]


def refuting_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline_tokens = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline_tokens else 0 for word in refuting_words]
        features.extend([1 if phrase in clean_headline else 0 for phrase in refuting_phrases])
        X.append(features)
    return X
