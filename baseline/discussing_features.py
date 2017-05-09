from feature_utils import clean, get_tokenized_lemmas
from tqdm import tqdm


discussing_words = [
	'ambiguous', 
        'equivocal', 
        'indeterminate', 
        'uncertain', 
        'neutral', 
        'impartial', 
        'debatable', 
        'unclear'
]


def discussing_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in discussing_words]
        X.append(features)
    return X


