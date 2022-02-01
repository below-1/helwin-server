import unittest
import os
import csv
from fono.parser import (
	type_1,
	type_2,
	type_3,
	type_4
)


class TestTypePattern(unittest.TestCase):

	data = []

	@classmethod
	def setUpClass(cls):
		for i in range(1, 5):
			with open(os.path.join('fono', 'data', f'test_type_{i}.csv')) as f:
				reader = csv.reader(f)
				# Skip header
				next(reader, None)
				_data = []
				for row in reader:
					kata, match = row
					_data.append((kata.strip(), match.strip()))
				TestTypePattern.data.append(_data)

	
	def test_01_positive_pattern(self):
		patterns = [type_1, type_2, type_3, type_4]
		for i, pattern in enumerate(patterns):
			print(f'TESTING POSITIVE PATTERN {i+1}')
			pattern_data = TestTypePattern.data[i]
			for kata, match in pattern_data:
				print('===', f'kata={kata}', f'expected={match}')
				m = pattern.match(kata)
				self.assertIsNotNone(m)
				self.assertEqual(m.group(), match)

	def test_02_negative_pattern(self):
		patterns = [type_1, type_2, type_3, type_4]
		for i, pattern in enumerate(patterns):
			print(f'TESTING NEGATIVE PATTERN {i+1}')

			# Join all words except data for current pattern
			_data = TestTypePattern.data[:]
			del _data[i]
			pattern_data = [ ]
			for _d in _data:
				pattern_data.extend(_d)

			for kata, _ in pattern_data:
				print('===', f'kata={kata}')
				m = pattern.match(kata)
				self.assertIsNone(m)

if __name__ == '__main__':
    unittest.main()