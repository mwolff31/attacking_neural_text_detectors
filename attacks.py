import random
from utils import load_txt, write_txt, load_misspelling_dict
import numpy as np


def create_word(word):

	upper = False
	eos = False
	eos_char = None
	poss = False
	new_word = word[1:]
	if word[0].isupper():
		first_letter = word[0].lower()
		upper = True
	else:
		first_letter = word[0]
	word = first_letter + new_word

	if word[-1] in ['.', ',', '"']:
		eos = True
		eos_char = word[-1]
		word = word[:-1]

	if word[-2:] == "'s":
		poss = True
		word = word[:-2]

	return word, upper, eos, eos_char, poss


def attack(
	text, 
	homoglyphs,
	attack_type='unlimited',
	max_percent_change=None,
	misspelling_dict=None,
	throwout = False):
	

	if attack_type == 'unlimited':
		adv_text = text
		for pair in homoglyphs:
			adv_text = adv_text.replace(pair[0], pair[1])
		num_changes=0
		for char in text:
			for pair in homoglyphs:
				if char == pair[0]:
					num_changes+=1
		return adv_text, num_changes

	if attack_type == 'random_limited':
		adv_text = ''
		og_max_num_changes = int(max_percent_change*len(text))
		homoglyph_indices = []
		for pair in homoglyphs:
			homoglyph_indices_i = [i for i in range(len(text)) if text[i] == pair[0]]
			homoglyph_indices.append(homoglyph_indices_i)
		homoglyph_indices_list = [i for sublist in homoglyph_indices for i in sublist]
		max_num_changes = min(len(homoglyph_indices_list), og_max_num_changes)
		homoglyph_indices = np.asarray(homoglyph_indices_list)
		np.random.shuffle(homoglyph_indices)
		homoglyph_indices = homoglyph_indices[:max_num_changes]
		indices_to_change = np.random.choice(homoglyph_indices, max_num_changes, replace=False)
		indices_to_change = indices_to_change.tolist()
		if (len(indices_to_change) < og_max_num_changes) and throwout:
			return text, len(indices_to_change)
		else:
			for i in range(len(text)):
				if i in indices_to_change:
					for pair in homoglyphs:
						if pair[0]==text[i]:
							adv_text += pair[1]
				else:
					adv_text += text[i]
				return adv_text, len(indices_to_change)


	if attack_type == 'misspelling':
		adv_text = ''
		misspelling_dict = load_misspelling_dict(misspelling_dict)
		adv_text = ''
		og_max_changes = int(max_percent_change*len(text.split(' ')))
		special_chars = [' ', '.', '"']
		words = text.split(' ')
		valid_word_indices = []
		for i in range(len(words)):
			word = words[i]

			new_word, _, _, _, _ = create_word(word)

			if new_word in misspelling_dict.keys():
				valid_word_indices.append(i)

		if len(valid_word_indices) == 0:
			return text, 0

		valid_word_indices = np.asarray(valid_word_indices)
		max_changes = min(len(valid_word_indices), og_max_changes)
		indices_to_change = np.random.choice(valid_word_indices, max_changes, replace=False).tolist()
		if (len(indices_to_change) < og_max_changes) and throwout:
			return text, len(indices_to_change)
		else:
			for i in range(len(words)):
				word = words[i]
				new_word = word
				if i in indices_to_change:
					new_word, upper, eos, eos_char, poss = create_word(word)
					misspell = np.asarray(misspelling_dict[new_word])
					misspell = np.random.choice(misspell)
					new_word = misspell
					if upper:
						new_word = ''
						for j in range(len(misspell)):
							if j == 0:
								new_word += misspell[0].upper()
							else:
								new_word += misspell[j]
					if eos:
						new_word += eos_char
					if poss:
						new_word += "'s"	

				adv_text += new_word + ' '
			return adv_text, len(indices_to_change)

	else:
		print('Attack type not defined!')
