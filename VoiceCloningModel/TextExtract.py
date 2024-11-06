import re
import inflect
from unidecode import unidecode
import eng_to_ipa as ipa
import torch
import numpy as np
import re
import soundfile
import os
import librosa


_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]





_lazy_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('r', 'ɹ'),
    ('æ', 'e'),
    ('ɑ', 'a'),
    ('ɔ', 'o'),
    ('ð', 'z'),
    ('θ', 's'),
    ('ɛ', 'e'),
    ('ɪ', 'i'),
    ('ʊ', 'u'),
    ('ʒ', 'ʥ'),
    ('ʤ', 'ʥ'),
    ('ˈ', '↓'),
]]

# List of (ipa, lazy ipa2) pairs:
_lazy_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('r', 'ɹ'),
    ('ð', 'z'),
    ('θ', 's'),
    ('ʒ', 'ʑ'),
    ('ʤ', 'dʑ'),
    ('ˈ', '↓'),
]]

# List of (ipa, ipa2) pairs
_ipa_to_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('r', 'ɹ'),
    ('ʤ', 'dʒ'),
    ('ʧ', 'tʃ')
]]
symbols = [
    '_', ',', '.', '!', '?', '-', '~', '…', 'N', 'Q', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
    'o', 'p', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ɑ', 'æ', 'ʃ', 'ʑ', 'ç', 'ɯ', 'ɪ', 'ɔ', 'ɛ', 'ɹ', 'ð', 'ə', 'ɫ', 
    'ɥ', 'ɸ', 'ʊ', 'ɾ', 'ʒ', 'θ', 'β', 'ŋ', 'ɦ', '⁼', 'ʰ', '`', '^', '#', '*', '=', 'ˈ', 'ˌ', '→', '↓', '↑', ' '
]
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}


def split_sentences(text,minimum_length=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[“”]', '"', text)
    text = re.sub('[‘’]', "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: 
        del sentences[-1]
    new_sentences = []
    current_sentence = []
    current_len = 0
    for index, sentence in enumerate(sentences):
        current_sentence.append(sentence)
        current_len += len(sentence.split(" "))
        if current_len > minimum_length or index == len(sentences) - 1:
            current_len = 0
            new_sentences.append(' '.join(current_sentence))
            current_sentence = []
    formatted_sentences=[]
    for sentence in new_sentences:
        if len(formatted_sentences)>0 and len(formatted_sentences[-1].split(" "))<=2:
            formatted_sentences[-1]=formatted_sentences[-1]+" "+sentence
        else:
            formatted_sentences.append(sentence)
    try: 
        if len(formatted_sentences[-1].split(" "))<=2:
            formatted_sentences[-2]=formatted_sentences[-2]+" "+formatted_sentences[-1]
            formatted_sentences.pop(-1)
    except:
        pass
    return formatted_sentences


def text_to_sequence(text):
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sequence = []
    clean_text = _clean(text)
    print(clean_text)
    print(f" length:{len(clean_text)}")
    for symbol in clean_text:
        if symbol not in symbol_to_id.keys():
            continue
        symbol_id = symbol_to_id[symbol]
        sequence += [symbol_id]
    print(f" length:{len(sequence)}")
    return sequence
def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text
def collapse_whitespace(text):
    return re.sub(r'\s+', ' ', text)
def _remove_commas(m):
    return m.group(1).replace(',', '')
def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')
def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'
def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))
def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
def mark_dark_l(text):
    return re.sub(r'l([^aeiouæɑɔəɛɪʊ ]*(?: |$))', lambda x: 'ɫ'+x.group(1), text)
def english_to_ipa(text):
    text = unidecode(text).lower()
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    phonemes = ipa.convert(text)
    phonemes = collapse_whitespace(phonemes)
    return phonemes
def english_to_ipa2(text):
    text = english_to_ipa(text)
    text = mark_dark_l(text)
    for regex, replacement in _ipa_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text.replace('...', '…')
def _clean(text):
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text

def cleaned_text_english(text):
    mark="EN"
    list_of_sequences=[]
    sentences=split_sentences(text)
    for sentence in sentences:
        sentence=re.sub(r'([a-z])([A-Z])', r'\1 \2', sentence)
        sentence=f"[{mark}]{sentence}[{mark}]"
        sequence=text_to_sequence(sentence)
        sequence=intersperse(sequence,0)    
        sequence=torch.LongTensor(sequence)
        list_of_sequences.append(sequence)
    return list_of_sequences
