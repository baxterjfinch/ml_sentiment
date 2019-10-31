from textblob import TextBlob
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob.sentiments import NaiveBayesAnalyzer

class TextBlobUtility():
	def __init__(self, title, content):
		self.blob = TextBlob(content, analyzer=NaiveBayesAnalyzer())
		self.title_blob = TextBlob(title, analyzer=NaiveBayesAnalyzer())
		# self.pblob = TextBlob(text)

	def get_title_blob(self):
		return self.title_blob

	def get_content_blob(self):
		return self.blob

	def get_sentences(self):
		return self.blob.sentences

	def get_words(self):
		return self.blob.words

	def get_noun_phrases(self):
		return self.blob.noun_phrases

	def get_text_sentiment(self):
		return self.blob.sentiment

	def get_blob_sentiment(self, blob=None):
		if blob.sentiment.classification == 'pos':
			return 1
		else:
			return 0

	def get_parsed_sentences_sentiment(self, blob=None):
		sent = {
			'pos_count': 0,
			'neg_count': 0,
			'p_pos': 0,
			'p_neg': 0,
			'count': 0
		}
		
		for sentence in blob.sentences:
			sent['count'] += 1
			sent['p_pos'] += sentence.sentiment.p_pos
			sent['p_neg'] -= sentence.sentiment.p_neg

			if sentence.sentiment.classification == 'pos':
				sent['pos_count'] +=1

			else:
				sent['neg_count'] -=1

		return sent




