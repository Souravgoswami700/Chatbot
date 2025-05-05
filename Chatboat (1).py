# Building a chatbot

import sys
import nltk
import string
import random

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton,
    QFileDialog, QVBoxLayout, QWidget, QLineEdit, QLabel
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChatBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stark - Chatbot")
        self.setGeometry(100, 100, 600, 500)

        self.sent_tokens = []
        self.word_tokens = []
        self.lemmer = nltk.stem.WordNetLemmatizer()

        nltk.download('punkt')
        nltk.download('wordnet')

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Chat with Stark! Load a corpus first.")
        layout.addWidget(self.label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message...")
        layout.addWidget(self.user_input)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.chat)
        layout.addWidget(self.send_button)

        self.load_button = QPushButton("Load Corpus")
        self.load_button.clicked.connect(self.load_corpus)
        layout.addWidget(self.load_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_corpus(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text files (*.txt)")
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_doc = f.read().lower()
                if raw_doc.strip():
                    self.sent_tokens = nltk.sent_tokenize(raw_doc)
                    self.word_tokens = nltk.word_tokenize(raw_doc)
                    self.chat_display.append("BOT: Corpus loaded successfully! Let's talk.")
                else:
                    self.chat_display.append("BOT: The selected file is empty.")
            except Exception as e:
                self.chat_display.append(f"BOT: Failed to load corpus. Error: {str(e)}")
        else:
            self.chat_display.append("BOT: No file selected.")

    def LemTokens(self, tokens):
        return [self.lemmer.lemmatize(token) for token in tokens]

    def LemNormalize(self, text):
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        return self.LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    def greet(self, sentence):
        GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
        GREET_RESPONSES = ["hi", "hey", "hello", "hi there", "I am glad! You’re talking to me."]
        for word in sentence.split():
            if word.lower() in GREET_INPUTS:
                return random.choice(GREET_RESPONSES)

    def bot_response(self, user_response):
        if not self.sent_tokens:
            return "Please load a corpus first."

        robo_response = ''
        self.sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=self.LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        if req_tfidf == 0:
            robo_response = "I am sorry! I don’t understand you."
        else:
            robo_response = self.sent_tokens[idx]

        self.sent_tokens.pop()  # Remove user input to avoid duplication
        return robo_response

    def chat(self):
        user_text = self.user_input.text()
        if user_text:
            self.chat_display.append("You: " + user_text)
            if user_text.lower() == "bye":
                self.chat_display.append("BOT: Goodbye! Take care <3")
                self.user_input.setDisabled(True)
                self.send_button.setDisabled(True)
            elif user_text.lower() in ["thanks", "thank you"]:
                self.chat_display.append("BOT: You are welcome!")
            elif self.greet(user_text):
                self.chat_display.append("BOT: " + self.greet(user_text))
            else:
                response = self.bot_response(user_text)
                self.chat_display.append("BOT: " + response)
            self.user_input.clear()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatBotGUI()
    window.show()
    sys.exit(app.exec_())

