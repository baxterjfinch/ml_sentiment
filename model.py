import csv
import json
import pickle
import requests
import pydotplus
import numpy as np
import pandas as pd
from random import randrange
import matplotlib.pyplot as plt
from sklearn import preprocessing, tree
from IPython.display import Image
from pandas import read_csv, set_option
from sklearn.externals.six import StringIO
from sklearn.feature_selection import chi2, SelectKBest
from numpy import loadtxt, set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import mysql.connector as mysql
db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "reojfist1",
    database = "sentiment"
)

cursor = db.cursor()

class DecisionModel():
	def __init__(self):
		# self.model = DecisionTreeClassifier()
		self.model = pickle.load(open("model.sav", "rb"))

	def save_model(self):
		print("Saving and printing visual")
		self.visualize_tree()

		filename = "model.sav"
		pickle.dump(self.model, open(filename, 'wb'))

	# def build_dataframe(self):		
	# 	names = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated', 'bias']
	# 	df = pd.read_csv("article_csv.csv", names=names)
	# 	features = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated']
	# 	X = df[features]
	# 	y = df["bias"]

	# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
	# 	self.model.fit(X_train, y_train)

	# 	dot_data = StringIO()
	# 	export_graphviz(self.model, out_file=dot_data, filled=True, rounded=False,
	# 		special_characters=True, feature_names = features, class_names=["0","1"])
	# 	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	# 	graph.write_png('bitcoin_news.png')
	# 	Image(graph.create_png())

	# 	self.save_model()

	def get_tree_depth(self):
		return self.model.get_depth()

	def get_tree_leaves(self):
		return self.model.get_n_leaves()

	# def train_from_csv(self):
	# 	names = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated', 'bias']
	# 	df = pd.read_csv("article_csv.csv", names=names)
	# 	features = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated']
	# 	X = df[features]
	# 	y = df["bias"]

	# 	self.model.fit(X, y)	

	# 	self.save_model()

	def train(self, articles):

		data = np.array(articles)
		print(data)
		features = ['tit_pos', 'tit_neg', 'tit_sent', 'tit_total', 'cont_pos', 'cont_neg', 'cont_sent']
		names = ['tit_pos', 'tit_neg', 'tit_sent', 'tit_total', 'cont_pos', 'cont_neg', 'cont_sent', 'cont_total']
		df = pd.DataFrame({'tit_pos': data[:, 0], 'tit_neg': data[:, 1],'tit_sent': data[:, 2], 'tit_total': data[:, 3], 'cont_pos': data[:, 4],'cont_neg': data[:, 5], 'cont_sent': data[:, 6], 'cont_total': data[:, 7]})
		X = df[features]
		y = df["cont_total"]

		self.model.fit(X, y)	

		self.save_model()

	def predict_article(self, articles):
		# names = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated', 'bias']
		# df = pd.read_csv(article, names=names)
		# features = ['title_sentiment', 'content_sentiment', 'description_sentiment', 'opinionated']
		# X = df[features]

		data = np.array(articles)
		features = ['tit_pos', 'tit_neg', 'tit_sent', 'tit_total', 'cont_pos', 'cont_neg', 'cont_sent']
		df = pd.DataFrame({'tit_pos': data[:, 0], 'tit_neg': data[:, 1],'tit_sent': data[:, 2], 'tit_total': data[:, 3], 'cont_pos': data[:, 4],'cont_neg': data[:, 5], 'cont_sent': data[:, 6]})
		X = df[features]

		y_pred = self.model.predict(X)
		print("Y PREDICT {}".format(y_pred))

		prob = self.model.predict_proba(X)
		print("Probability: {}".format(prob))

	def visualize_tree(self):
		# tree.plot_tree(self.model)

		features = ['tit_pos', 'tit_neg', 'tit_sent', 'tit_total', 'cont_pos', 'cont_neg', 'cont_sent']

		dot_data = StringIO()
		export_graphviz(self.model, out_file=dot_data, filled=True, rounded=False,
			special_characters=True, feature_names = features, class_names=["0","1"])
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
		graph.write_png('decision_model.png')
		Image(graph.create_png())

