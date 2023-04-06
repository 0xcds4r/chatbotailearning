import tensorflow as tf
import numpy as np
import os
import random
import json
import re
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, TFBertForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import tkinter as tk
import threading
import spacy
import openai
import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class Chatbot:
    def __init__(self):
        print("Welcome to the chatbot! How can I assist you today?")
        self.history = {}  # словарь, где ключ - это чат, значение - это список сообщений в этом чате
        self.current_chat = None  # текущий чат, с которым взаимодействует пользователь
        self.answers = self._load_answers()  # загрузка ответов
        self._internet_search = self._search_online  # функция по умолчанию - поиск ответа в интернете
        self.stop_generation = False  # переменная, которая используется для остановки генерации ответа на запрос
        self.previous_query = None  # предыдущий запрос пользователя
        self.gived_answer_json = False
        self.always_clear_history = False
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        self.stemmer = PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize, stop_words=None)

    def _load_answers(self):
        # загрузка ответов из файла
        answers_path = os.path.join(os.path.dirname(__file__), 'data', 'answers.json')
        with open(answers_path, 'r') as f:
            answers = json.load(f)
        return answers

    def _save_answers(self):
        # сохранение ответов в файл
        answers_path = os.path.join(os.path.dirname(__file__), 'data', 'answers.json')
        with open(answers_path, 'w') as f:
            json.dump(self.answers, f)

    messages_history = []

    def clearHistory(self):
        messages_history = []

    def setAlwaysClearHistory(self, val=True):
        self.always_clear_history = val

    def generate_openai_response(self, prompt):
        # Authenticate with the OpenAI API
        try:
            global messages_history
            messages_history.append({"role": "user", "content": prompt})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301", messages=messages_history, max_tokens=2000
            )

            if self.always_clear_history:
                messages_history = []
            if response and response.choices:
                response_text = response.choices[0].message.content
                return response_text
            else:
                return f"Извини, я не смог сгенерировать ответ на твой запрос."
        except Exception as e:
            messages_history = []
            return self.generate_openai_response(prompt)

    def search_in_chatgpt(self, query):
        # Generate a prompt that includes the query
        prompt = f"{query}"
    
        # Extract the response text from the API response
        text = self.generate_openai_response(prompt)
    
        # Remove any HTML tags from the response
        text = re.sub(r'<[^>]*>', '', text)
    
        # Return the response text
        return text

    def github_search(self, query):
        api_url = 'https://api.github.com/search/repositories'
        params = {
            'q': query,
            'sort': 'stars', # Sort by number of stars
            'order': 'desc', # Sort in descending order
        }
        headers = {'Accept': 'application/vnd.github.v3+json'} # Set the API version
        answer = ""
        with requests.Session() as session:
            response = session.get(api_url, params=params, headers=headers)
            response.raise_for_status() # Raise an exception if the request fails
            results = response.json()['items'] # Get the list of search results
            answer = ""
            for i, result in enumerate(results):
                if i >= 10:
                    break
                answer += f"{i+1}. {result['name']} ({result['html_url']})\n"
        return answer

    def _search_online(self, query):
        # функция для поиска ответа в интернете
        # google_answer = self.google_search(query)
        # github_answer = self.github_search(query)
        # wiki_answer = self.wiki_search(query)
        ai_answer = self.search_in_chatgpt(query)
        answer = ""
        
        # if google_answer is not None and len(google_answer) > 0:
            # answer += "\nGoogle: " + google_answer

        if ai_answer is not None and len(ai_answer) > 0:
            answer += "\n" + ai_answer

        # if github_answer is not None and len(github_answer) > 0:
            # answer += "\n\nInteresting GitHub repositories that you might find useful: \n" + github_answer
        
        self.answers[query.lower()] = answer
        self._save_answers()
        return answer

    def _get_answer(self, query):
        # Tokenize and preprocess the query
        query_tokens = self.preprocess_text(query)

        # Compute the TF-IDF matrix for the questions and the query
        question_texts = list(self.answers.keys())
        question_matrix = self.vectorizer.fit_transform(question_texts)
        query_matrix = self.vectorizer.transform([query])

        # Compute the cosine similarity between the query matrix and the question matrix
        similarity_scores = question_matrix.dot(query_matrix.T).toarray().flatten()

        # Get the indices of the matching answers
        matching_indices = similarity_scores.argsort()[::-1]

        # Get the top 3 matching answers
        top_answers = [self.answers[question_texts[i]] for i in matching_indices[:3]]

        # Generate the answer by combining the top answers
        answer = self.generate_answer(query_tokens, top_answers)

        # Update the given_answer_json flag and return the answer
        if answer and not self.gived_answer_json:
            self.gived_answer_json = True
            return answer
        else:
            self.gived_answer_json = False
            return self._internet_search(query)

    def preprocess_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords and stem the tokens
        tokens = [self.stemmer.stem(token.lower()) for token in tokens if token.lower() not in self.stopwords]

        return tokens

    def tokenize(self, text):
        # Tokenize and stem the text
        tokens = self.preprocess_text(text)

        return tokens

    def generate_answer(self, query_tokens, answers):
        # Compute the frequency of each token in the answers
        answer_counts = Counter()

        for answer in answers:
            answer_tokens = self.preprocess_text(answer)
            answer_counts.update(answer_tokens)

        # Compute the score of each answer based on the frequency of its tokens
        answer_scores = Counter()

        for answer in answers:
            answer_tokens = self.preprocess_text(answer)
            score = sum([answer_counts[token] for token in answer_tokens])
            answer_scores[answer] = score

        # Compute the best answer based on the score of each answer
        best_answer = None
        best_score = 0

        for answer, score in answer_scores.items():
            if score > best_score:
                best_answer = answer
                best_score = score

        return best_answer

    def _get_message(self, text, is_user=True):
        # функция для создания сообщения, где is_user = True, если сообщение от пользователя, и False, если сообщение от нейросети
        return {'text': text, 'is_user': is_user}

    def _get_response(self, query):
        # функция для получения ответа на запрос пользователя
        label = 1
    
        if label == 1:
            return self._get_answer(query)
        else:
            return "I'm sorry, I don't understand. Can you please rephrase your question?"
    
    def add_message(self, message):
        # функция для добавления сообщения в историю чата
        if self.current_chat is None:
            return False
        else:
            self.history[self.current_chat].append(message)
            return True
    
    def create_chat(self, chat_name):
        # функция для создания нового чата
        if chat_name in self.history:
            return False
        else:
            self.history[chat_name] = []
            self.current_chat = chat_name
            return True
    
    def switch_chat(self, chat_name):
        # функция для переключения на другой чат
        if chat_name not in self.history:
            return False
        else:
            self.current_chat = chat_name
            return True
    
    def generate_response(self, query):
        # функция для генерации ответа на запрос пользователя
        self.stop_generation = False
        response = None
    
        while not self.stop_generation:
            response = self._get_response(query)
    
            if len(response) > 0:
                break
    
        message = self._get_message(response, is_user=False)
        self.add_message(message)
    
        return response
    
    def stop_response_generation(self):
        # функция для остановки генерации ответа на запрос
        self.stop_generation = True
    
    def change_previous_query(self, query):
        # функция для изменения предыдущего запроса пользователя
        if self.previous_query is not None:
            self.previous_query = query
            return True
        else:
            return False
    
    def search_online(self, query):
        # функция для поиска ответа в интернете
        self._internet_search = self._search_online
        response = self._internet_search(query)
        message = self._get_message(response, is_user=False)
        self.add_message(message)
        return response
    
    def use_previous_query(self):
        # функция для использования предыдущего запроса пользователя
        if self.previous_query is not None:
            response = self.generate_response(self.previous_query)
            return response
        else:
            return "I'm sorry, there is no previous query to use."
    
    def get_history(self):
        # функция для получения истории текущего чата
        if self.current_chat is None:
            return {}
        else:
            return self.history[self.current_chat]

class ChatbotUI:
    def __init__(self, chatbot):
        self.chatbot = chatbot

        self.root = tk.Tk()
        self.root.title("Chatbot")
        self.api_key = ""

        # Create a frame to hold the chat history and the input field.
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a scrollable chat history text widget.
        self.chatlog_text = tk.Text(self.frame, bg="white", font=("Arial", 12), state=tk.DISABLED)
        self.chatlog_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a scrollbar for the chat history text widget.
        self.scrollbar = tk.Scrollbar(self.frame, command=self.chatlog_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the chat history text widget to use the scrollbar.
        self.chatlog_text.config(yscrollcommand=self.scrollbar.set)

        # Create an input field and a send button.
        self.input_frame = tk.Frame(self.root, bg="#e6e6e6")
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.input_text = tk.Entry(self.input_frame, bg="white", font=("Arial", 12))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Add an input field for the API key
        self.api_key_frame = tk.Frame(self.root, bg="#e6e6e6")
        self.api_key_frame.pack(fill=tk.X, padx=5, pady=5)

        self.api_key_text = tk.Entry(self.api_key_frame, bg="white", font=("Arial", 12))
        self.api_key_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.api_key_button = tk.Button(self.api_key_frame, text="Set API Key", command=self.set_api_key)
        self.api_key_button.pack(side=tk.RIGHT, padx=5, pady=5)

        if len(self.get_api_key()) > 0:
            self.api_key_text.insert(tk.END, self.get_api_key())

    def set_api_key(self):
        self.api_key = self.api_key_text.get()
        # Save the API key to a file
        with open("api_key.txt", "w") as f:
            f.write(self.api_key)

    def get_api_key(self):
        if not self.api_key:
            if os.path.isfile("api_key.txt"):
                with open("api_key.txt", "r") as f:
                    self.api_key = f.read().strip()

        return self.api_key

    def send_message(self):
        message = self.input_text.get().strip()
        self.input_text.delete(0, tk.END)

        if message == '':
            return

        # Disable the input field and the Send button while the chatbot generates the response.
        self.input_text.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)

        # Generate the chatbot's response in a separate thread.
        threading.Thread(target=self.generate_response, args=(message,)).start()

    def generate_question_response(self, message):
        if len(self.get_api_key()) <= 0:
            print("api key is not found!")
            return
        openai.api_key = self.get_api_key()

        ai_question = self.chatbot.search_in_chatgpt(f"Generate one random question on one of the given topics: {message}")
        self.chatbot.clearHistory()
        self.generate_response(ai_question)
        self.chatbot.clearHistory()

    def generate_response(self, message):
        if len(self.get_api_key()) <= 0:
            print("api key is not found!")
            return
        openai.api_key = self.get_api_key()

        if '/genqa' in message:
            message = message.replace(f"/genqa ", "")
            self.input_text.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)
            self.generate_question_response(message)
            self.chatbot.clearHistory()
            self.input_text.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            return

        if '/forqa' in message:
            message = message.replace(f"/forqa ", "")
            i = 0
            self.chatbot.setAlwaysClearHistory(True)
            while i < 10000:
                self.input_text.config(state=tk.DISABLED)
                self.send_button.config(state=tk.DISABLED)
                self.generate_question_response(message)
                self.chatbot.clearHistory()
                i += 1
                time.sleep(0.15)  # add a delay of 0.15 seconds (i.e., 15/100 seconds)
            self.chatbot.setAlwaysClearHistory(False)
            self.input_text.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            return

        if '/train' in message:
            self.chatlog_text.config(state=tk.NORMAL)
            self.chatlog_text.insert(tk.END, "Training started!\nWait for generating questions..\n")

            data = []
            i = 0
            words = message.split()
            message = message.replace(f"/train {words[1]} ", "")
            count = int(words[1])
            print("message: " + message + "\n")
            print("train count: " + str(count) + "\n")
            self.chatbot.setAlwaysClearHistory(True)
            while i < count:
                ai_question = self.chatbot.search_in_chatgpt(f"Generate one random question on one of the given topics: {message}")
                data.append(ai_question)
                i += 1
                self.chatbot.clearHistory()
                time.sleep(0.15)  # add a delay of 0.15 seconds (i.e., 15/100 seconds)

            for question in data:
                response = self.chatbot.generate_response(question)
                
                self.chatlog_text.config(state=tk.NORMAL)
                if self.chatlog_text.index('end-1c') != '1.0':
                    self.chatlog_text.insert(tk.END, "\n")
                self.chatlog_text.insert(tk.END, "You: " + question + "\n")
                self.chatlog_text.insert(tk.END, "Bot: " + response + "\n")
                self.chatlog_text.config(state=tk.DISABLED)
        
                # Scroll to the bottom of the chat history.
                self.chatlog_text.yview(tk.END)
                # clear
                self.chatbot.clearHistory()
            self.chatbot.setAlwaysClearHistory(False)

            # Enable the input field and the Send button.
            self.chatlog_text.config(state=tk.NORMAL)
            self.input_text.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.chatlog_text.insert(tk.END, "Training end!\n")
            self.chatlog_text.yview(tk.END)
            return

        response = self.chatbot.generate_response(message)

        # Update the chat history with the message and the response.
        self.chatlog_text.config(state=tk.NORMAL)
        if self.chatlog_text.index('end-1c') != '1.0':
            self.chatlog_text.insert(tk.END, "\n")
        self.chatlog_text.insert(tk.END, "You: " + message + "\n")
        self.chatlog_text.insert(tk.END, "Bot: " + response + "\n")
        self.chatlog_text.config(state=tk.DISABLED)

        # Scroll to the bottom of the chat history.
        self.chatlog_text.yview(tk.END)

        # Enable the input field and the Send button.
        self.input_text.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        # self.chatbot.clearHistory()

    def run(self):
        self.root.mainloop()

bot = Chatbot()
ui = ChatbotUI(bot)
ui.run()