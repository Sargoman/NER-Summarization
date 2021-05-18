from flask import Flask,url_for,render_template,request,send_file,redirect,request
from flask_uploads import UploadSet,configure_uploads,ALL,DATA
from werkzeug import secure_filename

# Other Packages
import os
import pandas as pd

# NLP
import spacy
nlp = spacy.load("en_core_web_lg")

# Summarization
from spacy_summarization import text_summarizer
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# function to capture all token_type with token in dataframe 
def token_type(data):
    text = []
    label = []
    for ent in data.ents:
        text.append(ent.text)
        label.append(ent.label_)

    df = pd.DataFrame()
    df['text'] = text
    df['label'] = label
    
    return df
	
# funciton to capture specific token_type
def text_label(label, dfrm):
	text_label_li = []
	for i in range(len(dfrm['label'])):
		if dfrm['label'][i] == label.upper():
			text_label_li.append(dfrm['text'][i])
	return text_label_li

def task_opt_org(data):
	doc = nlp(data)
	dfrm = token_type(doc)
	rel = set(text_label('org', dfrm))
	return rel

def task_opt_name(data):
	doc = nlp(data)
	dfrm = token_type(doc)
	rel = set(text_label('Person', dfrm))
	return rel

def task_opt_place(data):
	doc = nlp(data)
	dfrm = token_type(doc)
	rel = set(text_label('GPE', dfrm))
	return rel

def task_opt_date(data):
	doc = nlp(data)
	dfrm = token_type(doc)
	rel = set(text_label('date', dfrm))
	return rel

# summarization function
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Initialize App
app = Flask(__name__)

# Configuration For Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadedfiles'
configure_uploads(app,files)


@app.route('/')
def index():
	return render_template("index.html")


@app.route('/extract', methods=['GET', 'POST'])
def extract():
	if request.method == 'POST' and 'rawtext' in request.files:
		file = request.files['rawtext']
		filename = secure_filename(file.filename)
		file.save(os.path.join('static/uploadedfiles', filename))

		with open(os.path.join('static/uploadedfiles',filename), 'r+', encoding="utf-8") as f:
			c_text = f.read()
			# SpaCy
			fs_spacy = text_summarizer(c_text)
			# Gensim Summarizer
			fs_gensim = summarize(c_text)
			# NLTK
			fs_nltk = nltk_summarizer(c_text)
			# Sumy
			fs_sumy = sumy_summary(c_text)

			result_org = task_opt_org(c_text)
			result_places = task_opt_place(c_text)
			result_date = task_opt_date(c_text)
			result_name = task_opt_name(c_text)


	return render_template('index.html',c_text=c_text,result_org=result_org, result_places=result_places, result_date=result_date,result_name=result_name, fs_spacy=fs_spacy, fs_gensim=fs_gensim,fs_nltk=fs_nltk,fs_sumy=fs_sumy)



if __name__ == '__main__':
	app.run(debug=True)