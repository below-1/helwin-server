import re
import pickle
import os


huruf = '[a-z]'
vocab = '[aiueo]'
konsonan = '[bcdfghjklmnpqrstvwxyz]'
diftong = '(ai|au|ei|oi)'
gab_konsonan = '(kh|ng|ny|sy)'


# V
type_1 = re.compile(
	# Vocal
	'(^'
		'('
			'([aiueo]|ai|au|ei|oi)'
			'(?='
				# Followed by any number of pair of CV or VC
				# Just like cvvccv...
				'('
					'('
						'('
							'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
							'([aiueo]|ai|au|ei|oi)'
						')'
						'|'
						'('
							'([aiueo]|ai|au|ei|oi)'
							'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
						')'
					')+'
				')'
			')'
		')'
		'|'
		'^[aiueo]$'
	')'
	,
	re.X)

# C+V
type_2 = re.compile(
	'^'
	# C+
	'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])+'
	# V
	'(?P<vocab>ai|au|ei|oi|[aiueo])'
	'(?='
		'('
			'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
			'(ai|au|ei|oi|[aiueo])'
		')'
		'|'
		'('
			'(ai|au|ei|oi|[aiueo])'
			'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
		')'
		'|'
		'('
			'(ai|au|ei|oi|[aiueo])'
			'$'
		')'
		'|'
		'(?P=vocab)'
	')',
	re.X)

# VC+
type_3 = re.compile(
	'^'
	'(ai|au|ei|oi|[aiueo])'
	'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
	'(?='
		'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz]){2,6}'
		'|'
		'('
			'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
			'([aiueo]|ai|au|ei|oi)'
		')'
		'|'
		'('
			'([aiueo]|ai|au|ei|oi)'
			'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
		')'
		'|'
		'$'
	')',
	re.X)

# (C+VC+)V
type_4 = re.compile(
	'('
		'^'
		'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])+'
		'([aiueo]|ai|au|ei|oi)'
		'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])+'
		'$'
		'|'
		'^'
		'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])+'
		'([aiueo]|ai|au|ei|oi)'
		'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])'
		'(?='
			'('
				'('
					'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])+'
					'([aiueo]|ai|au|ei|oi)'
				')'
				'|'
				'('
					'(kh|ng|ny|sy|[bcdfghjklmnpqrstvwxyz])*'
					'$'
				')'
			')'
		')'
	')'
	,
	re.X)

special_type_1 = re.compile(
	'^'
	'trans',
	re.X
)

class Parser:

	def load(self):
		with open(os.path.join(os.getcwd(), 'fono', 'data', 'dasar.pickle'), mode='rb') as f:
			self.database = pickle.load(f)

	def parse(self, s: str):
		special_match = special_type_1.match(s)
		if special_match is not None:
			start, end = special_match.span()
			sub_word = s[end:]
			base_word_tokens = self.parse(sub_word)
			return ['trans'] + base_word_tokens
		
		word = s
		tokens = []
		patterns = [ type_1, type_3, type_2, type_4]
		while len(word) != 0:
			found = False
			for idx, pattern in enumerate(patterns):
				m = pattern.match(word)
				# breakpoint()
				if m is not None:
					start, end = m.span()
					tokens.append(word[start:end])
					word = word[end:]
					found = True
					break
			if not found:
				tokens.append(word)
				word = ''
		return tokens


def test_regex():
	test_data = [
		{
			'words': ['air', 'itu', 'ini', 'api', 'abaskus', 'ia', 'ingin', 'ikan', 'idaman'],
			'rule': type_1
		}
	]

	for data in test_data:
		rule = data['rule']
		words = data['words']
		for w in words:
			print(f'word={w}  match={rule.match(w)}')

if __name__ == '__main__':
	s = 'homoterm'
	parser = Parser()
	parser.load()
	result = parser.parse(s)
	print(result)
