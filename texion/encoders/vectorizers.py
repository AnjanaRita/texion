from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer


OPTIONS = {"CountVectorizer": CountVectorizer,
           "HashingVectorizer": HashingVectorizer,
           "TfidfVectorizer": TfidfVectorizer}


class BaseVectorizer:
    """Base Text Vectorizer class"""
    def __new__(cls, name, params=None):

        if name not in OPTIONS.keys():
            raise NameError(
                f"please select one of these as the name: {[x for x in OPTIONS.keys()]}")

        if params:
            clf = OPTIONS.get(name)(**params)
            print(
                f""" Using {clf.__class__.__name__} with parameters:\n{params}""")

        else:
            clf = OPTIONS.get(name)()
            print(
                f"""Using {clf.__class__.__name__} .\nnote: running with default configuration""")
        return clf
