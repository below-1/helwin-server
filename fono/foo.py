import os
import pickle


def _main():
	database = set()
	with open(os.path.join(os.getcwd(), 'fono', 'data', 'dasar.txt')) as f:
		lines = f.read().splitlines()
		for line in lines:
			tokens = line.split(',')
			kata = tokens[1]
			kata = kata.strip()
			kata = kata[1:-1]
			if kata.isalpha():
				database.add(kata)
	database = frozenset(database)
	with open(os.path.join(os.getcwd(), 'fono', 'data', 'dasar.pickle'), mode='wb') as f:
		pickle.dump(database, f)


if __name__ == '__main__':
	_main()
