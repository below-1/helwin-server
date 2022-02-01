"""
Build document based database from words in dictionary.
The resulting database have two collections, 'words' and 'summary'.
"""
import pickle
import os
import re
import json
import pymongo
from collections import namedtuple
from dataclasses import dataclass
from fono.parser import Parser


cons = [
    'kh',
    'ng',
    'ny',
    'sy',
    'b',
    'c',
    'd',
    'f',
    'g',
    'h',
    'j',
    'k',
    'l',
    'm',
    'n',
    'p',
    'q',
    'r',
    's',
    't',
    'v',
    'w',
    'x',
    'y',
    'z'
]
vocabs = [
    'ai',
    'au',
    'ei',
    'oi',
    'a',
    'i',
    'u',
    'e',
    'o'
]


def is_vocal(c: str) -> bool:
    return c in vocabs


@dataclass
class FDPoint:
    letter: str
    nucleus: int = 0
    coda: int = 0
    onset: int = 0

    def hit_coda(self):
        self.coda += 1

    def hit_onset(self):
        self.onset += 1
    
    def hit_nucleus(self):
        self.nucleus += 1

    def count_in_map(self, map: dict):
        if self.letter not in map:
            return
        val = map[self.letter]
        if is_vocal(self.letter):
            if type(val) is not int:
                raise Exception(f"Unknown value: {val}")
            self.nucleus = val
        else:
            if type(val) is not dict:
                raise Exception(f"Unknown value: {val}")
            if 'coda' not in val:
                raise Exception(f"coda is not in: {val}")
            if 'onset' not in val:
                raise Exception(f"onset is not in: {val}")
            self.coda = val['coda']
            self.onset = val['onset']

    def asdict(self):
        if is_vocal(self.letter):
            return { 'letter': self.letter, 'total': self.nucleus }
        return { 
            'letter': self.letter, 
            'coda': self.coda,
            'onset': self.onset,
            'total': self.coda + self.onset }

@dataclass
class FDType:
    ftype: str
    points: dict

    def add_letter(self, char: str):
        if char not in self.points:
            self.points[char] = FDPoint(letter=char)

    def count_in_map(self, map: dict):
        for fd_point in self.points.values():
            fd_point.count_in_map(map)
    
    def asdict(self):
        return {
            'ftype': self.ftype,
            'points': [ v.asdict() for (_, v) in self.points.items() ]
        }

@dataclass
class FD:
    name: str
    plus: FDType
    minus: FDType

    def count_in_map(self, counter_map):
        self.plus.count_in_map(counter_map)
        self.minus.count_in_map(counter_map)

    def asdict(self):
        return {
            'name': self.name,
            'plus': self.plus.asdict(),
            'minus': self.minus.asdict()
        }


parser = Parser()
parser.load()
re_vocab = re.compile('(ai|au|ei|oi|[aiueo])')

BASE_PATH = os.path.join(
    os.getcwd(),
    'fono',
    'data'
)
KATA_DASAR_PICKLE_PATH = os.path.join(BASE_PATH, 'dasar.pickle')
KONSONAN_FITUR_SPEC_PATH = os.path.join(BASE_PATH, 'fitur_konsonan.txt')
VOCAL_FITUR_SPEC_PATH = os.path.join(BASE_PATH, 'fitur_vocab.txt')
KONSONAN_FITURS = [
  'consonantal',
  'sonorant',
  'coronal',
  'anterior',
  'labial',
  'round',
  'distributed',
  'high',
  'low',
  'back',
  'atr',
  'spread',
  'constricted',
  'voice',
  'continuant',
  'lateral',
  'nasal',
  'strident',
  'del rel',
  'bilabial',
  'labiodental',
  'dental',
  'alveolar',
  'postalveolar',
  'palatal',
  'velar',
  'glottal'
]
VOCAL_FITURS = [
    'high', 'low', 'back', 'tense', 'round', 'atr', 'front'
]


def parse_token(word : str):
    tokens = parser.parse(word)
    result = {
        'word': word,
        'tokens': []
    }
    for token in tokens:
        m = re_vocab.search(token)
        if m is None:
            # print(f'word = {word}')
            # print(f"tokens = {tokens}")
            # input()
            continue
        
        nucleus_start, nucleus_end = m.span()
        onset = token[:nucleus_start]
        nucleus = token[nucleus_start:nucleus_end]
        coda = token[nucleus_end:]
        result['tokens'].append({
          'token': token,
          'nucleus': nucleus,
          'onset': onset,
          'coda': coda
        })
    return result


def split_char(s: str):
    gab_kons = re.compile('^(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])')
    v_pattern = re.compile('^[aiueo]')
    patterns = [
        (gab_kons, 'konsonan'), 
        (v_pattern, 'vocal')
    ]

    word = s
    result= []
    while word != '':
        for (pattern, tag)  in patterns:
            is_match = pattern.match(word)
            if is_match:
                _, end = is_match.span()
                match_w = is_match.group()
                word = word[end:]
                result.append((match_w, tag))
    return result


def tokenize(path=KATA_DASAR_PICKLE_PATH):
    database = None
    with open(path, mode='rb') as f:
        database = pickle.load(f)
    results = []
    for word in database:
        result = parse_token(word)
        results.append(result)
    return results
    

def count_syllables(tokens):
    SyllCounter = namedtuple(
        'SyllCounter', 
        ['vocab', 'kons']
    )

    kons_counters = {
        c: { 'onset': 0, 'coda': 0 } for c in cons
    }
    vocab_counters = {
        c: 0 for c in vocabs
    }
    # breakpoint()

    for data in tokens:
        w = data['word']
        for token in data['tokens']:
            try:
                for (char, _) in split_char(token['onset']):
                    kons_counters[char]['onset'] += 1
                for (char, _) in split_char(token['coda']):
                    kons_counters[char]['coda'] += 1
                vocab_counters[token['nucleus']] += 1
            except Exception as ex:
                continue

    return SyllCounter(vocab=vocab_counters, kons=kons_counters)


def load_fitures_spec(fiturs, path):
    fit_descs = [
        FD(
            name=name,
            plus=FDType("plus", points={}),
            minus=FDType("minus", points={})
        ) for name in fiturs 
    ]

    with open(path, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        char, *fitur_flags = line.split(' ')
        for fitur_flag, fd in zip(fitur_flags, fit_descs):
            if fitur_flag == '+':
                fd.plus.add_letter(char)
            elif fitur_flag == '-':
                fd.minus.add_letter(char)
    return fit_descs


def count_fiture(fitur_spec, fget_total):
    fitur_data = []
    for fitur in fitur_spec.keys():
        letters = fitur_spec[fitur]['letters'].keys()
        temp = []
        total = 0
        for letter in letters:
            # _map = counter.kons[letter]
            total_in_letter = fget_total(letter)
            # total_in_letter = _map['coda'] + _map['onset']
            total += total_in_letter
            temp.append({ 
                'letter': letter,
                'total': total_in_letter
            })
        fitur_data.append({
            'fitur': fitur,
            'letters': temp,
            'total': total
        })
    return fitur_data

def count_from_file(fiturs, path, fget_total):
    fitur_spec = load_fitures_spec(fiturs, path)
    return count_fiture(fitur_spec, fget_total)


def add_tag(fiturs_data, tag):
    return [
        { **fd, 'tag': tag } for fd in fiturs_data
    ]


if __name__ == '__main__':
    tokens = tokenize()
    counter = count_syllables(tokens)
    ckons = counter.kons
    cvoc = counter.vocab

    kons_fiturs = load_fitures_spec(KONSONAN_FITURS, KONSONAN_FITUR_SPEC_PATH)
    for fd in kons_fiturs:
        fd.count_in_map(ckons)
    kons_fiturs = [
        { **fd.asdict(), 'tag': 'konsonan' } for fd in kons_fiturs
    ]

    voc_fiturs = load_fitures_spec(VOCAL_FITURS, VOCAL_FITUR_SPEC_PATH)
    for fd in voc_fiturs:
        fd.count_in_map(cvoc)
    voc_fiturs = [
        { **fd.asdict(), 'tag': 'vocal' } for fd in voc_fiturs
    ]

    all_fiturs = [
        *kons_fiturs, *voc_fiturs
    ]


    # kons_fiturs = count_from_file(
    #     KONSONAN_FITURS, 
    #     KONSONAN_FITUR_SPEC_PATH,
    #     lambda char: ckons[char]['coda'] + ckons[char]['onset']
    # )
    # # Add tag based on type (konsonan or vocal)
    # kons_fiturs = add_tag(kons_fiturs, 'konsonan')

    # vocal_fiturs = count_from_file(
    #     VOCAL_FITURS, 
    #     VOCAL_FITUR_SPEC_PATH,
    #     lambda char: cvoc[char]
    # )
    # # Add tag based on type (konsonan or vocal)
    # vocal_fiturs = add_tag(vocal_fiturs, 'vocal')

    # # Merge fiturs
    # fiturs = [
    #     *vocal_fiturs,
    #     *kons_fiturs
    # ]

    # # Add order
    # fiturs = [
    #     { **fit, 'idx': idx } for (idx, fit) in enumerate(fiturs)
    # ]

    # # Transform counter to list of document
    # counter_docs = []
    # for k, v in counter.vocab.items():
    #     counter_docs.append({
    #         'tag': 'vocal',
    #         'letter': k,
    #         'total': v
    #     })
    # for k, doc in counter.kons.items():
    #     counter_docs.append({
    #         'tag': 'konsonan',
    #         'letter': k,
    #         'total': doc['onset'] + doc['coda'],
    #         'syll_detail': doc
    #     })

    #     # for (char, tag) in split_char(w):
    #     #     key = (char, tag)
    #     #     if key not in counters:
    #     #         counters[key] = 0
    #     #     counters[key] += 1
    # # vocal_counters = [ {'char': c, 'count': count, 'tag': tag } for ((c, tag), count) in counters.items() if tag == 'vocal']
    # # kons_counters = [ {'char': c, 'count': count, 'tag': tag} for ((c, tag), count) in counters.items() if tag == 'konsonan']
    # # data_counters = vocal_counters + kons_counters
    # # with open(os.path.join(os.getcwd(), 'fono', 'data', 'results.json'), mode='w') as f:
    # #     json.dump(results, f)
    client = pymongo.MongoClient()
    db = client.fono
    db.fiturs.delete_many({})
    db.fiturs.insert_many(all_fiturs)
    # db.counter.insert_many(counter_docs)
