import re
from spellchecker import SpellChecker

def extract_answer_text(text):
    lines = text.splitlines()

    filtered_lines = []

    ignore_keywords = [
        "question",
        "instructions",
        "answer all",
        "discuss",
        "explain",
        "describe",
        "define",
        "state",
        "outline",
        "topic",
        "title",
        "objective",
        "section",
        "part a",
        "part b",
        "q1",
        "q2",
        "q3",
        "q4"
    ]

    for line in lines:
        clean_line = line.strip()

        if clean_line == "":
            continue

        lower_line = clean_line.lower()

        # Ignore question-like lines
        if any(keyword in lower_line for keyword in ignore_keywords):
            continue

        # Ignore lines ending with question mark
        if clean_line.endswith("?"):
            continue

        # Ignore very short lines
        if len(clean_line.split()) < 4:
            continue

        filtered_lines.append(clean_line)

    answer_text = "\n".join(filtered_lines)

    return answer_text

spell = SpellChecker()


def count_ai_phrases(text):
    ai_phrases = [
        "in conclusion",
        "it is important to note",
        "furthermore",
        "moreover",
        "additionally",
        "therefore",
        "overall",
        "in summary",
        "as a result",
        "this highlights",
        "this demonstrates"
    ]

    text_lower = text.lower()
    count = 0

    for phrase in ai_phrases:
        count += text_lower.count(phrase)

    return count


def count_structural_punctuation(text):
    quote_count = (
        text.count('"') +
        text.count("'") +
        text.count("“") +
        text.count("”") +
        text.count("‘") +
        text.count("’") +
        text.count("<<") +
        text.count(">>")
    )

    dash_count = text.count("--") + text.count("—")
    parenthesis_count = text.count("(") + text.count(")")
    colon_count = text.count(":")
    comma_count = text.count(",")
    fullstop_count = text.count(".")

    return {
        "quote_count": quote_count,
        "dash_count": dash_count,
        "parenthesis_count": parenthesis_count,
        "colon_count": colon_count,
        "comma_count": comma_count,
        "fullstop_count": fullstop_count
    }


def extract_basic_features(text):
    text = text.strip()

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    sentence_count = len(sentences)
    word_count = len(words)

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    sentence_lengths = [
        len(re.findall(r'\b\w+\b', sentence))
        for sentence in sentences
    ]

    if len(sentence_lengths) > 1:
        mean = sum(sentence_lengths) / len(sentence_lengths)
        sentence_variance = sum(
            (length - mean) ** 2 for length in sentence_lengths
        ) / len(sentence_lengths)
    else:
        sentence_variance = 0

    unique_words = len(set(words))
    vocab_diversity = unique_words / word_count if word_count > 0 else 0

    # Fast spelling check using pyspellchecker
    sample_size = 100

    if len(words) > sample_size:
        start = (len(words) - sample_size) // 2
        words_to_check = words[start:start + sample_size]
    else:
        words_to_check = words

    misspelled_words = spell.unknown(words_to_check)
    spelling_errors = len(misspelled_words)

    checked_word_count = len(words_to_check)
    error_rate = spelling_errors / checked_word_count if checked_word_count > 0 else 0

    ai_phrase_count = count_ai_phrases(text)
    punctuation = count_structural_punctuation(text)

    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
        "sentence_variance": sentence_variance,
        "vocab_diversity": vocab_diversity,
        "spelling_errors": spelling_errors,
        "error_rate": error_rate,
        "ai_phrase_count": ai_phrase_count,
        "quote_count": punctuation["quote_count"],
        "dash_count": punctuation["dash_count"],
        "parenthesis_count": punctuation["parenthesis_count"],
        "colon_count": punctuation["colon_count"],
        "comma_count": punctuation["comma_count"],
        "fullstop_count": punctuation["fullstop_count"]
    }


def generate_feature_explanation(features):
    reasons = []

    if features["avg_sentence_length"] > 25:
        reasons.append("The average sentence length is high, suggesting structured AI-like writing.")
    elif features["avg_sentence_length"] < 8:
        reasons.append("The average sentence length is low, suggesting human-like writing.")
    else:
        reasons.append("The average sentence length is moderate.")

    if features["sentence_variance"] < 10:
        reasons.append("Sentence variation is low, meaning the writing has a rhythmic and uniform structure.")
    else:
        reasons.append("Sentence variation is high, indicating more natural human variation.")

    if features["vocab_diversity"] > 0.7:
        reasons.append("Vocabulary diversity is high, suggesting expressive human writing.")
    elif features["vocab_diversity"] < 0.4:
        reasons.append("Vocabulary diversity is low, suggesting repetitive writing patterns.")
    else:
        reasons.append("Vocabulary diversity is moderate.")

    if features["error_rate"] > 0.1:
        reasons.append("High spelling error rate detected, suggesting human-written text.")
    elif features["error_rate"] < 0.02:
        reasons.append("Very low spelling error rate, suggesting clean AI-like writing.")
    else:
        reasons.append("Moderate spelling accuracy observed.")

    if features["ai_phrase_count"] > 2:
        reasons.append("Repeated AI-like transition phrases were detected.")
    elif features["ai_phrase_count"] > 0:
        reasons.append("Some formal transition phrases were detected, but usage is limited.")

    if features["quote_count"] > 2:
        reasons.append("Quotation marks appear repeatedly, which may indicate structured or generated formatting.")

    if features["dash_count"] > 3:
        reasons.append("Frequent dash or em-dash usage detected, which can suggest AI-style sentence linking.")

    if features["parenthesis_count"] > 2:
        reasons.append("Frequent parenthesis usage detected, suggesting heavily structured explanation.")

    if features["colon_count"] > 3:
        reasons.append("Frequent colon usage detected, suggesting structured AI-like formatting.")

    return reasons


def adjust_ai_score(ml_ai_score, features):
    adjusted_score = ml_ai_score

    breakdown = {
        "increased": [],
        "reduced": []
    }

    if features["error_rate"] < 0.02:
        adjusted_score += 8
        breakdown["increased"].append("Very low spelling error rate increased AI probability by 8%.")
    elif features["error_rate"] > 0.10:
        adjusted_score -= 10
        breakdown["reduced"].append("High spelling error rate reduced AI probability by 10%.")

    if features["sentence_variance"] < 10:
        adjusted_score += 7
        breakdown["increased"].append("Low sentence variation increased AI probability by 7%.")
    elif features["sentence_variance"] > 40:
        adjusted_score -= 5
        breakdown["reduced"].append("High sentence variation reduced AI probability by 5%.")

    if features["vocab_diversity"] < 0.4:
        adjusted_score += 5
        breakdown["increased"].append("Low vocabulary diversity increased AI probability by 5%.")
    elif features["vocab_diversity"] > 0.75:
        adjusted_score -= 5
        breakdown["reduced"].append("High vocabulary diversity reduced AI probability by 5%.")

    if features["ai_phrase_count"] > 2:
        increase = min(12, features["ai_phrase_count"] * 2)
        adjusted_score += increase
        breakdown["increased"].append(
            f"Repeated AI-like phrases increased AI probability by {increase:.2f}%."
        )

    if features["quote_count"] > 2:
        increase = min(8, (features["quote_count"] - 2) * 1.5)
        adjusted_score += increase
        breakdown["increased"].append(
            f"Repeated quotation marks increased AI probability by {increase:.2f}%."
        )

    if features["dash_count"] > 3:
        increase = min(10, (features["dash_count"] - 3) * 2)
        adjusted_score += increase
        breakdown["increased"].append(
            f"Frequent dash/em-dash usage increased AI probability by {increase:.2f}%."
        )

    if features["parenthesis_count"] > 2:
        increase = min(8, (features["parenthesis_count"] - 2) * 1.5)
        adjusted_score += increase
        breakdown["increased"].append(
            f"Frequent parenthesis usage increased AI probability by {increase:.2f}%."
        )

    if features["colon_count"] > 3:
        increase = min(10, (features["colon_count"] - 3) * 2)
        adjusted_score += increase
        breakdown["increased"].append(
            f"Frequent colon usage increased AI probability by {increase:.2f}%."
        )

    adjusted_score = max(0, min(100, adjusted_score))

    return adjusted_score, breakdown


def highlight_suspicious_text(text, features):
    highlighted = text

    suspicious_phrases = [
        "In conclusion",
        "It is important to note",
        "Furthermore",
        "Moreover",
        "Additionally",
        "Therefore",
        "Overall",
        "In summary",
        "As a result",
        "This highlights",
        "This demonstrates"
    ]

    for phrase in suspicious_phrases:
        highlighted = re.sub(
             phrase,
             f'<span class="highlight-ai" title="AI transition phrase">{phrase}</span><span class="ai-label">AI Phrase</span>',
             highlighted,
             flags=re.IGNORECASE
        )

    if features["dash_count"] > 3:
        highlighted = highlighted.replace("--", '<span class="highlight-ai">--</span>')
        highlighted = highlighted.replace("—", '<span class="highlight-ai">—</span>')

    if features["colon_count"] > 3:
        highlighted = highlighted.replace(":", '<span class="highlight-ai">:</span>')

    if features["parenthesis_count"] > 2:
        highlighted = highlighted.replace("(", '<span class="highlight-ai">(</span>')
        highlighted = highlighted.replace(")", '<span class="highlight-ai">)</span>')

    return highlighted
    