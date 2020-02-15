import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class Detector(object):

	def __init__(self):

		print('Initializing Detector...')

		data = torch.load('detector-large.pt')
		self.model = RobertaForSequenceClassification.from_pretrained('roberta-large')
		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
		self.model.load_state_dict(data['model_state_dict'])
		self.model.eval().cuda()


	def predict(self, txt):

		tokens = self.tokenizer.encode(txt, max_length=self.tokenizer.max_len)
		#tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
		tokens = torch.Tensor(tokens)

		tokens = tokens.unsqueeze(0).cuda().long()

		mask = torch.ones_like(tokens).cuda().long()

		logits = self.model(tokens, attention_mask=mask)

		probs = logits[0].softmax(dim=-1)

		probs = probs.detach().cpu().flatten().numpy()

		return probs

	def get_result(self, txt):
		p = self.predict(txt)
		prob = np.max(p)
		result = np.argmax(p)
		if result == 1:
			result = 'Human'
		else:
			result = 'Machine'
		print(result, ' | ', prob)


if __name__ == '__main__':

	text = 'adversarial machine learning'

	det = Detector()
	p = det.predict(text)
	prob = np.max(p)
	result = np.argmax(p)
	if result == 1:
		result = 'Human'
	else:
		result = 'Machine'
	print(result, ' | ', prob)



