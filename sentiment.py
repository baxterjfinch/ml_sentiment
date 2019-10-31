import json
import nltk
import pickle
import random
import requests
import string, re
import time
from datetime import date
from nltk.text import Text
from statistics import mode
from bs4 import BeautifulSoup
from collections import Counter
from text_blob_helper import TextBlobUtility
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

from model import DecisionModel
import mysql.connector as mysql

db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "",
    database = "sentiment"
)

cursor = db.cursor()

class SentimentAnalysis():
	def __init__(self, content=None, kind=None, search_keyword=None, category=None, count=None):
		self.api_key = ""
		self.type = kind
		self.content = content
		self.search_keyword = search_keyword
		self.category = category
		self.count = count
		self.save_queue = {}

		self.authors = []
		self.process_queue = []

		self.positive_articles = []
		self.negative_articles = []
		self.urls = []

		self.current_article = None
		self.all_words = []
		self.content_features = []
		self.title_features = []
		self.word_features = None

		self.hits = None
		self.pages = None
		self.next_page = 0
		self.max_pages = 50

		# Single item thats returned from get_top_stories
	
	def get_nyt_articles(self, pub_date, search_term):
		response = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?q={}&pub_date={}&page={}&sort=newest&api-key={}".format(search_term, pub_date, self.next_page, self.api_key))
		loaded = json.loads(response.content)

		if "fault" in loaded.keys():
			print(loaded)
			return

		if "status" in loaded.keys():
			if loaded["status"]:
				if loaded["response"]:
					if loaded["response"]["docs"]:
						self.generate_queueable_archive_items(loaded["response"]["docs"])
						self.next_page += 1
						self.pages= loaded["response"]["meta"]["hits"]

	def generate_queueable_archive_items(self, items):
		for item in items:
			cont_id = item["_id"][16:]
			session = requests.Session()
			url = item["web_url"]
			if url in self.urls:
				continue

			self.urls.append(url)
			if (len(url) < 1):
				continue

			req = session.get(url)
			soup = BeautifulSoup(req.text, features="html.parser")
			paragraphs = soup.find_all('div', class_='StoryBodyCompanionColumn')
			article = ''
			for p in paragraphs:
				article = article + p.get_text()

			if len(article) < 2:
				continue

			print(item["headline"]["main"])

			self.process_queue.append({
				"url": url,
				"cont_id": cont_id,
				"pub_date": item['pub_date'][:10],
				"title": item["headline"]["main"],
				"article": article
			})

	def set_current_article(self, title, content):
		self.current_article = (title, content)
		

	def build_feature_set(self, kind, sentiment):
		# kind is either 0 for title or 1 for content
		if sentiment > 0:
			self.build_content_features("pos", self.current_article[kind], kind)

		elif sentiment < 0:
			self.build_content_features("neg", self.current_article[kind], kind)

	def build_content_features(self, category, p, kind):
		if p[0]:
			if kind == 1:
				self.content_features.append((p[0], category))

			else:
				self.title_features.append((p[0], category))

		else:
			if kind == 1:
				self.content_features.append((p, category))

			else:
				self.title_features.append((p, category))

	def move_to_classified(self, sentiment):

		if sentiment > 0:
			self.positive_articles.append(self.current_article)

		else:
			self.negative_articles.append(self.current_article)

	def calculate_sentiment(self, tokenized_words):
		# print("Current Art: {}".format(self.current_article))
		pos = self.get_pos(tokenized_words)
		neg = self.get_neg(tokenized_words)
		overall_sentiment = pos + neg

		# print("POS: {}\nNEG: {}".format(pos, neg))

		return pos, neg, overall_sentiment

	def get_pos(self, words):
		# pos_words = []
		pos = 0
		lower = [x.lower() for x in words]

		with open('positive-words.txt') as pf:
			for positive_word in pf.readlines():
				if positive_word.strip() in lower:
					pos += 1
					self.all_words.append((positive_word.strip()))

		return pos

	def get_neg(self, words):
		# neg_words = []
		neg = 0
		lower = [x.lower() for x in words]

		with open('negative-words.txt') as pf:
			for negative_word in pf.readlines():
				if negative_word.strip() in lower:
					neg -= 1
					self.all_words.append((negative_word.strip()))

		return neg

	def build_data_sets(self, kind):
		if kind == "title_features":
			featuresets = [(self.find_features(art), category) for (art, category) in self.title_features]
			
			print(self.title_features)
			# random.shuffle(featuresets)
			# self.training_set = featuresets[:20000]
			# self.testing_set = featuresets[20000:]

		else:
			featuresets = [(self.find_features(art), category) for (art, category) in self.content_features]
			print(featuresets)
			# random.shuffle(featuresets)
			# self.training_set = featuresets[:20000]
			# self.testing_set = featuresets[20000:]


	def naive_bayes_classifier(self):
		classifier = nltk.NaiveBayesClassifier.train(self.training_set)
		# print("Classifier Accuracy Precentage: {}".format(nltk.classify.accuracy(classifier, self.testing_set)) * 100)

		classifier.show_most_informative_features(15)
		most_informative = classifier.most_informative_features()
		counts = Counter(most_informative)

		print(counts)

	def add_parameter_to_save_queue(
		self, 
		cont_id=None,
		date= None,
		url=None, 
		tit_pos_count=None, 
		tit_p_pos=None, 
		tit_neg_count=None, 
		tit_p_neg=None, 
		tit_count=None, 
		cont_pos=None, 
		cont_p_pos=None, 
		cont_neg_count=None, 
		cont_p_neg=None, 
		cont_count=None,
		tit_sent=None,
		cont_sent=None):

		self.save_queue[url] = {
			"id": cont_id,
			"url": url,
			"date": date,
			"tit_pos_count": tit_pos_count,
			"tit_p_pos": tit_p_pos,
			"tit_neg_count": tit_neg_count,
			"tit_p_neg": tit_p_neg,
			"tit_count": tit_count,
			"cont_pos": cont_pos,
			"cont_p_pos": cont_p_pos,
			"cont_neg_count": cont_neg_count,
			"cont_p_neg": cont_p_neg,
			"cont_count": cont_count,
			"tit_sent": tit_sent,
			"cont_sent": cont_sent

		}

	def update_db_tables(self):
		values = []
		query = "INSERT IGNORE INTO articles (id, url, date, tit_pos_count, tit_p_pos, tit_neg_count, tit_p_neg, tit_count, cont_pos, cont_p_pos, cont_neg_count, cont_p_neg, cont_count, tit_sent, cont_sent) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

		for article in self.save_queue:
			art = self.save_queue[article]
			values.append((art["id"], article, art["date"], art["tit_pos_count"], art["tit_p_pos"], art["tit_neg_count"], art["tit_p_neg"], art["tit_count"], art["cont_pos"], art["cont_p_pos"], art["cont_neg_count"], art["cont_p_neg"], art["cont_count"], art["tit_sent"], art["cont_sent"]))
			
		cursor.executemany(query, values)
		db.commit()

		self.save_queue = {}

def get_arts(ts):
	print("Starting App")
	sentiment = SentimentAnalysis()
	readable = date.fromtimestamp(ts).isoformat()[:10]
	
	while sentiment.max_pages > 0:
		sentiment.max_pages -= 1
		ts -= 86400
		sentiment.get_nyt_articles(readable, "politics")
		readable = date.fromtimestamp(ts).isoformat()[:10]

	for article in sentiment.process_queue:
		article_url = article["url"]
		pub_date = article["pub_date"]
		print(pub_date)
		cont_id = article["cont_id"]

		title = article["title"]
		content = article["article"]

		blobbed = TextBlobUtility(title, content[:500])

		title_blob = blobbed.get_title_blob()
		content_blob = blobbed.get_content_blob()

		title_calced = blobbed.get_parsed_sentences_sentiment(blob=title_blob)
		cont_calced = blobbed.get_parsed_sentences_sentiment(blob=content_blob)

		title_sentiment = blobbed.get_blob_sentiment(blob=title_blob)
		cont_sentiment = blobbed.get_blob_sentiment(blob=content_blob)

		sentiment.add_parameter_to_save_queue(
			cont_id=cont_id,
			url=article_url, 
			date=pub_date,
			tit_pos_count=title_calced['pos_count'], 
			tit_p_pos=title_calced['p_pos'], 
			tit_neg_count=title_calced['neg_count'], 
			tit_p_neg=title_calced['p_neg'], 
			tit_count=title_calced['count'], 
			cont_pos=cont_calced['pos_count'], 
			cont_p_pos=cont_calced['p_pos'], 
			cont_neg_count=cont_calced['neg_count'], 
			cont_p_neg=cont_calced['p_neg'], 
			cont_count=cont_calced['count'],
			tit_sent = title_sentiment,
			cont_sent = cont_sentiment 
		)
		# print(blobbed.get_pattern_sentiment())
		# print(blobbed.get_sentences())

		# sentiment.add_to_frequency_queue(tokenized_title, tokenized_content)
		# print("Title Frequency Queue: {}".format(sentiment.title_frequency_queue))

		if len(sentiment.save_queue.keys()) == 10:
			sentiment.update_db_tables()

		

		# title_pos, title_neg, title_sentiment = sentiment.calculate_sentiment(tokenized_title)
		
		# content_pos, content_neg, content_sentiment = sentiment.calculate_sentiment(tokenized_content)
		# sentiment.move_to_classified(title_sentiment+content_sentiment)

		# sentiment.add_parameter_to_save_queue(
		# 	url=article_url, 
		# 	cont_id = cont_id,
		# 	pub_date = pub_date,
		# 	tit_pos=title_pos, 
		# 	tit_neg=title_neg, 
		# 	tit_sent=title_sentiment, 
		# 	cont_pos=content_pos, 
		# 	cont_neg=content_neg, 
		# 	cont_sent=content_sentiment
		# )




	# for article in sentiment.process_queue:

		# title_freq = sentiment.calculate_title_frequency()
		# print("Title Frequency: {}".format(title_freq))
		# cont_freq = sentiment.calculate_content_frequency()
		# print("Content Frequency: {}".format(cont_freq))




		# print(article_sentiment)
	sentiment.update_db_tables()




	# sentiment.calculate_distribution()
	# sentiment.build_data_sets("title_features")
	# sentiment.naive_bayes_classifier()

	# sentiment.build_data_sets("content_features")
	# sentiment.naive_bayes_classifier()

		# features = sentiment.find_features()

		# print(features)

	# print(sentiment.documents)



#### GET DATA FROM DB AND TEACH MACHINE ####

def retrieve_data():
	query = "SELECT id, tit_pos, tit_neg, tit_sent, tit_total, cont_pos, cont_neg, cont_sent, cont_total FROM articles"
	cursor.execute(query)
	records = cursor.fetchall()

	data = []
	

	for record in records:
		if(record[8] > 0):
			this_bias = 1
		else:
			this_bias = 0
		data.append([record[1], record[2], record[3], record[4], record[5], record[6], record[7], this_bias])

	return data


def teach_machine(data):
	model = DecisionModel()
	model.train(data)

ts = time.time()
get_arts(ts)
# data = retrieve_data()
# teach_machine(data)


# The Article Search API returns a max of 10 results at a time. The meta node in the response contains the total number of matches ("hits") 
# and the current offset. Use the page query parameter to paginate thru results (page=0 for results 1-10, page=1 for 11-20, ...). 
# You can paginate thru up to 100 pages (1,000 results). If you get too many results try filtering by date range.

