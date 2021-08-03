import nltk
import re

from nltk.tokenize.texttiling import TextTilingTokenizer

nltk.download('stopwords')


def clean_chapter(text):
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("\(.*?\)", "", text)
    return text


def get_chapters_from_nltk(text_file):
    tt = TextTilingTokenizer()
    paragraphs = tt.tokenize(text_file)
    paragraphs = [p.replace('\r\n', '\n') for p in paragraphs]
    paragraphs = [clean_chapter(p) for p in paragraphs]
    return paragraphs


def get_chapters_from_text(text):
    text = text.replace('\r\n', '\n')
    chapters = get_chapters_from_nltk(text)

    return chapters
