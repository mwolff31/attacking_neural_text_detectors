import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

def load_txt(filename):
	lines = ''
	with open(filename) as fp:
		line = fp.readline()
		while line:
			l = line.strip()
			lines += l
			lines += '\n'
			line = fp.readline()
	fp.close()
	return lines

def write_txt(filename, string):
	with open(filename, 'w') as fp:
		fp.write(string)
	fp.close()


def load_misspelling_dict(fname):
	with open(fname, 'r') as fp:
		d = json.load(fp)
	fp.close()
	return d


def load_json_file(filename):

	with open(filename, 'r') as json_file:
		json_list = list(json_file)
	json_file.close()

	texts = [json.loads(x)['text'] for x in json_list]

	return texts

def get_results(experiment_name):
	print('Results for ' + experiment_name)
	results_pth = './experimental_results/'+experiment_name
	asr_pth = results_pth+'/results.txt'
	num_ch_pth = results_pth+'/num_changes.txt'

	probs = load_txt(asr_pth).split(' ')
	num_ch = load_txt(num_ch_pth).split(' ')
	probs = np.float32(np.asarray(probs))

	num_human = 0
	num_changed = 0
	pths = glob.glob('./experimental_results/'+experiment_name+'/adv_texts/*.txt')
	avg = 0
	for pth in pths:
		txt_i = load_txt(pth)
		avg += len(txt_i)
	print(avg / len(pths))


	for i in range(probs.size):
		if np.float32(probs[i]) >= 0.5:
			num_human+=1
		else:
			pass
		num_changed+=np.float32(num_ch[i])

	print('Average confidence:', np.mean(probs))
	print('Detector accuracy:', 1. - (num_human / probs.size))
	print('Average number of changes:', num_changed/len(num_ch))
	print('Number of Attacks Run:', probs.size)



def get_graph_data(exp_name):
	exp_name_list = exp_name.split('_')
	pths = glob.glob('./experimental_results/{}'.format(exp_name))
	ext = './xperimental_results'
	pths = [pth for pth in pths if '_1.0_' not in pth]
	x = np.arange(0.00, 0.0525, 0.0025)
	raw = []
	for i in range(x.size):
		probs = np.float32(np.asarray(load_txt(ext+'/{}_{}_{}'.format(exp_name_list[0], str(x[i]), exp_name_list[-1])+'/results.txt').split(' ')))
		raw.append(probs)
	raw = np.asarray(raw)
	success = np.uint8(raw+0.5)
	num_human = np.sum(success, axis=-1)
	asrs = 1. - (num_human / probs.size)
	fig = plt.figure()
	plt.scatter(x, asrs, s=10)
	plt.plot(x, asrs)
	plt.xlabel('Max. Pct. of Text Sample Characters Replaced')
	plt.ylabel('Detector Accuracy')
	plt.title(r'Detector Accuracy vs. Max. Pct. of Text Sample Characters Replaced')
	plt.xlim(-0.002, 0.052)
	plt.savefig('all_graph.png')
	plt.show()
	

	return x, np.asarray(asrs)



