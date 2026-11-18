"""Known gateway alias equivalence for compatibility checks, never API routing."""


def same_embedding_model(actual, expected):
    aliases = {'text-embedding-3-small', 'openai/text-embedding-3-small'}
    return bool(actual and expected) and (
        actual == expected or actual in aliases and expected in aliases)
