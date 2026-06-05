from features import extract_basic_features

sample_text = """
This is a simple test. It checks how the system behaves.
Sometimes sentences are short. Sometimes they are much longer and more complex.
"""

features = extract_basic_features(sample_text)

print(features)