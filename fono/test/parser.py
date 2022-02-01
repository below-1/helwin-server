import unittest
import os
import csv
from fono.parser import (
	Parser
)


class TestParser(unittest.TestCase):
	
	def test_01_positive_pattern(self):
		data = (
			('imbuhan', ('im', 'bu', 'han')),
			('musyawarah', ('mu', 'sya', 'wa', 'rah')),
			('bangkrut', ('bang', 'krut')),
			('struktur', ('struk', 'tur')),
			('struktur', ('struk', 'tur')),
			('struktural', ('struk', 'tu', 'ral')),
			('transaksi', ('trans', 'ak', 'si')),
			('transfusi', ('trans', 'fu', 'si')),
            ('transgender', ('trans', 'gen', 'der')),
            ('masyarakat', ('ma', 'sya', 'ra' ,'kat')),
            ('ideal',('i', 'de' , 'al')),
            ('main',('ma', 'in')),
            ('kembangbiak', ('kem' , 'bang', 'bi','ak')),
            ('mereka', ('me' , 're', 'ka')),
            ('interaksi', ('in', 'te', 'rak','si')),
            ('tautologi', ('tau', 'to', 'lo', 'gi')),
            ('tsunami', ('tsu', 'na', 'mi')),
            ('zoologi', ('zo', 'o', 'lo', 'gi')),
            ('zoetrop', ('zo', 'et', 'rop')),
            ('abad',('a','bad')),
            ('aku', ('a', 'ku')),
            ('api', ('a', 'pi')),
            ('itu', ('i', 'tu')),
            ('anda', ('an', 'da')),
			('bueng', ('bu', 'eng')),
			('homoterm', ('ho', 'mo', 'term'))
		)
		
		parser = Parser()
		for i, (kata, tokens) in enumerate(data):
			result = tuple(parser.parse(kata))
			if result != tokens:
				print('FAIL')
				print('expected: ', tokens)
				print('result: ', result)

if __name__ == '__main__':
    unittest.main()