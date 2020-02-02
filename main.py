import warnings

with warnings.catch_warnings():
	warnings.simplefilter('ignore')
	import numpy as np
	from detector import Detector
	from attacks import attack
	from utils import load_txt, write_txt, load_json_file, get_results
	import os
	from tqdm import tqdm
	import time

def run_experiment(
	homoglyphs, 
	attack_type, 
	detector,
	experiment_name,
	data_file,
	percent_change = None,
	mispelling_dict = None,
	throwout=False):

	start_time = time.time()

	out_path = './experimental_results/' + experiment_name + '/'
	adv_text_path = out_path + 'adv_texts/'
	numerical_results_path = out_path + 'results.txt'
	num_changes_path = out_path +'num_changes.txt'

	print('Running Experiment: {} ...'.format(experiment_name))

	if not os.path.isdir(out_path):
		os.mkdir(out_path)
		os.mkdir(adv_text_path)

	text_list = load_json_file(data_file)

	_range = tqdm(range(len(text_list)))
	i = 0

	for _ in _range:

		text_to_use = det.tokenizer.decode(
			det.tokenizer.encode(text_list[i], max_length=det.tokenizer.max_len))[3:-4]

		adv_text, num_changes = attack(
			text_to_use, homoglyphs, attack_type, percent_change, mispelling_dict, throwout)

		if throwout and (adv_text==text_to_use):
			pass

		else:

			write_txt(adv_text_path+str(i)+'.txt', adv_text)

			probs = detector.predict(adv_text)

			human_prob = probs[1]

			_range.set_description('{} | {}'.format(i, human_prob))

			with open(numerical_results_path, 'a') as f:
				f.write(str(human_prob) + ' ')
			f.close()

			with open(num_changes_path, 'a') as f:
				f.write(str(num_changes)+' ')
			f.close()

		i+=1

	end_time = time.time()

	print('Time to complete experiment (minutes):', (end_time-start_time)/60.)



if __name__ == '__main__':
	
	data_file = './data/xl-1542M-k40.test.jsonl'

	det = Detector()

	homoglyphs = [['e', 'е'], ['a', 'а']]

	exp_name = 'unlimited_e_a'
	run_experiment(
		homoglyphs,
		'unlimited',
		det, 
		exp_name,
		data_file,
		None,
		None,
		False)

	get_results(exp_name)
