import csv
import json
import pickle
import requests
import pydotplus
import numpy as np
import pandas as pd
from random import randrange
from IPython.display import Image
from sklearn import preprocessing
from pandas import read_csv, set_option
from sklearn.externals.six import StringIO
from sklearn.feature_selection import chi2, SelectKBest
from numpy import loadtxt, set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model import DecisionModel

API_KEY = ""

class Articles():
	def __init__(self, page=None, api_key=None):
		self.api_key = api_key
		self.content = []
		self.results_count = 0
		self.organized_and_gauged = []
		self.test_article = None
		self.organized_list = []
		self.organized_test_list = []

	def get_everything(self, from_date=None, to_date=None, category=None):
		response = requests.get("https://newsapi.org/v2/everything?q={}&from={}&to={}&pageSize=100&apiKey={}".format(category, from_date, to_date, self.api_key))
		loaded = json.loads(response.content)
		print(loaded)

		for article in loaded["articles"]:
			self.content.append(article)

	def get_top(self, page=None, q=None, date=None):
		if (page == None):
			self.page = 1
		else:
			self.page = page

		response = requests.get("https://newsapi.org/v2/top-headlines?country=us&apiKey={}".format(self.api_key))
		loaded = json.loads(response.content)
		self.content = loaded["articles"]
		self.results_count = loaded["totalResults"]

	def get_single_article(self, category):
		response = requests.get("https://newsapi.org/v2/top-headlines?category={}&pageSize=1&apiKey={}".format(category, self.api_key))
		loaded = json.loads(response.content)
		self.test_article = loaded["articles"]

	def get_single_fox_news_article(self):
		response = requests.get("https://newsapi.org/v2/top-headlines?sources=fox-news&pageSize=100&apiKey={}".format(self.api_key))
		loaded = json.loads(response.content)
		my_idx = randrange(10)
		self.test_article = loaded["articles"][my_idx]

	def get_single_cnn_news_article(self):
		response = requests.get("https://newsapi.org/v2/top-headlines?sources=cnn&pageSize=100&apiKey={}".format(self.api_key))
		loaded = json.loads(response.content)
		my_idx = randrange(10)
		self.test_article = loaded["articles"][my_idx]

	def get_single_breitbart_news_article(self):
		response = requests.get("https://newsapi.org/v2/top-headlines?sources=breitbart-news&pageSize=100&apiKey={}".format(self.api_key))
		loaded = json.loads(response.content)
		my_idx = randrange(10)
		self.test_article = loaded["articles"][my_idx]

	def convert_to_csv(self):
		self.organize_and_gauge()

		with open('article_csv.csv', mode='a') as article_csv:
			fieldnames = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated', 'bias']
			writer = csv.DictWriter(article_csv, fieldnames=fieldnames)

			for item in self.organized_and_gauged:
				if item["title_sentiment"] != 0 or item["content_sentiment"] != 0 or item["description_sentiment"] != 0 or item["opinionated"] != 0:
					writer.writerow({
						"title_sentiment": item["title_sentiment"], 
						"content_sentiment": item["content_sentiment"], 
						"description_sentiment": item["description_sentiment"], 
						"opinionated": item["opinionated"], 
						"bias": item["bias"] })

	def organize_and_gauge(self):
		for article in self.content:
			if (article["description"]):
				description_sentiment, description_pos, description_neg, description_bias = self.gauge_content(article["description"])
			else:
				description_sentiment = 0
				description_pos = 0
				description_neg = 0 
				description_bias = 0

			if (article["title"]):
				title_sentiment, title_pos, title_neg, title_bias = self.gauge_content(article["title"])
			else:
				title_sentiment = 0
				title_pos = 0
				title_neg = 0 
				title_bias = 0

			if (article["content"]):
				content_sentiment, content_pos, content_neg, content_bias = self.gauge_content(article["content"])
			else:
				content_sentiment = 0
				content_pos = 0
				content_neg = 0 
				content_bias = 0

			opinionated = self.get_opinionated(description_sentiment, title_sentiment, content_sentiment)
			bias = self.get_bias(description_sentiment, title_sentiment, content_sentiment)

			self.organized_list.append([title_sentiment, content_sentiment, description_sentiment, opinionated, bias])

			self.organized_and_gauged.append({
				"title_sentiment": title_sentiment,
				"content_sentiment": content_sentiment,
				"description_sentiment": description_sentiment,
				"opinionated": opinionated,
				"bias": bias
			})

	def save_prediction_article(self):
		article = self.test_article

		if (article["description"]):
				description_sentiment, description_pos, description_neg, description_bias = self.gauge_content(article["description"])
		else:
			description_sentiment = 0
			description_pos = 0
			description_neg = 0 
			description_bias = 0

		if (article["title"]):
			title_sentiment, title_pos, title_neg, title_bias = self.gauge_content(article["title"])
		else:
			title_sentiment = 0
			title_pos = 0
			title_neg = 0 
			title_bias = 0

		if (article["content"]):
			content_sentiment, content_pos, content_neg, content_bias = self.gauge_content(article["content"])
		else:
			content_sentiment = 0
			content_pos = 0
			content_neg = 0 
			content_bias = 0

		opinionated = self.get_opinionated(description_sentiment, title_sentiment, content_sentiment)
		bias = self.get_bias(description_sentiment, title_sentiment, content_sentiment)

		print("Hypothesis values: title_sentiment {}, content_sentiment {}, description_sentiment {}, opinionated {}".format(title_sentiment, content_sentiment, description_sentiment, opinionated))
		
		self.organized_test_list.append([title_sentiment, content_sentiment, description_sentiment, opinionated])

		# with open('hypothesis.csv', mode='w') as article_csv:
		# 	fieldnames = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated']
		# 	writer = csv.DictWriter(article_csv, fieldnames=fieldnames)
		# 	writer.writerow({
		# 		"title_sentiment": title_sentiment,
		# 		"content_sentiment": content_sentiment,
		# 		"description_sentiment": description_sentiment,
		# 		"opinionated": opinionated
		# 	})


	def get_pos(self, words):
		pos = 0
		lower = [x.lower() for x in words]

		with open('positive-words.txt') as pf:
			for positive_word in pf.readlines():
				if positive_word.strip() in lower:
					pos += 1
		return pos

	def get_neg(self, words):
		neg = 0
		lower = [x.lower() for x in words]

		with open('negative-words.txt') as pf:
			for negative_word in pf.readlines():
				if negative_word.strip() in lower:
					neg -= 1
		return neg

	def fibonacci(self, n): 
		if n<0: 
			print("Incorrect input") 
		# First Fibonacci number is 0 
		elif n==0:
			return 0
		elif n==1: 
			return 1
		# Second Fibonacci number is 1 
		elif n==2: 
			return 2
		else: 
			return self.fibonacci(n-1)+self.fibonacci(n-2) 

	def gauge_content(self, content):
		pos = self.get_pos(content.split())
		neg = self.get_neg(content.split())
		bias = 1

		sentiment = self.fibonacci(abs(pos + neg))
		if (abs(neg) > pos):
			sentiment = sentiment * -1
			bias = 0

		# print(pos, neg, sentiment)
		return sentiment, pos, neg, bias

	def gauge_description(self, description):
		pass

	def get_opinionated(self, description_sentiment, title_sentiment, content_sentiment):
		return abs(description_sentiment) + abs(title_sentiment) + abs(content_sentiment) 

	def get_bias(self, description_sentiment, title_sentiment, content_sentiment):
		total = (description_sentiment*.05 + title_sentiment*.75 + content_sentiment*.20) / 3
		if total >=0:
			return 1
		else:
			return 0

	# def load_model(self):
	# 	self.model = pickle.load(open("model.sav", "rb"))

	# def save_model(self):
	# 	filename = "model.sav"
	# 	pickle.dump(self.model, open(filename, 'wb'))

art = Articles(api_key=API_KEY)

####### Gets data
for i in range(15):
	date = i + 6
	art.get_everything(from_date="2019-09-{}".format(date), to_date="2019-09-{}".format(date), category="health")

# for i in range(25):
# 	date = i + 5
# 	art.get_everything(from_date="2019-09-{}".format(date), to_date="2019-09-{}".format(date), category="business")

# for i in range(25):
# 	date = i + 5
# 	art.get_everything(from_date="2019-09-{}".format(date), to_date="2019-09-{}".format(date), category="trump")

# for i in range(25):
# 	date = i + 5
# 	art.get_everything(from_date="2019-09-{}".format(date), to_date="2019-09-{}".format(date), category="technology")

# art.get_top()




# Loads existing model

art.convert_to_csv()
# my_list = art.organized_list

model = DecisionModel()
# print("ARITCLES: {}".format(len(my_list)))
# model.train(my_list)
model.train_from_csv()
# model.build_dataframe()
print("Tree Depth: {}".format(model.get_tree_depth()))

print("Leave Count: {}".format(model.get_tree_leaves()))

# model.predict_article("hypothesis.csv")
model.visualize_tree()
# art.get_single_article("technology")

# art.get_single_fox_news_article()
# art.get_single_cnn_news_article()
# art.get_single_breitbart_news_article()

# art.save_prediction_article()

# model.predict_article(art.organized_test_list)
# art.get_single_cnn_news_article()
# art.get_single_breitbart_news_article()

# art.save_prediction_article()


# art.predict()

