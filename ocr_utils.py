import cv2
import easyocr
import pytesseract
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)


def preprocess_for_tesseract(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(
        image,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gray = cv2.equalizeHist(gray)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        15
    )

    kernel = np.ones((1, 1), np.uint8)

    cleaned = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned


def extract_with_easyocr_original(image_path):
    results = reader.readtext(
        image_path,
        detail=1,
        paragraph=True
    )

    text_parts = []

    for result in results:
        text_parts.append(result[1])

    return "\n".join(text_parts)


def extract_with_tesseract_processed(image_path):
    processed_image = preprocess_for_tesseract(image_path)

    if processed_image is None:
        return ""

    custom_config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(
        processed_image,
        config=custom_config
    )

    return text


def text_quality_score(text):
    words = text.split()

    if len(words) == 0:
        return 0

    valid_words = 0

    for word in words:
        clean_word = ''.join(char for char in word if char.isalpha())

        if len(clean_word) >= 2:
            valid_words += 1

    return valid_words / len(words)


def extract_handwritten_text(image_path):
    easy_text = extract_with_easyocr_original(image_path)
    easy_quality = text_quality_score(easy_text)

    tesseract_text = extract_with_tesseract_processed(image_path)
    tess_quality = text_quality_score(tesseract_text)

    if easy_quality >= tess_quality:
        return easy_text

    return tesseract_text