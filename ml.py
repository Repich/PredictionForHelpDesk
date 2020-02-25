from sklearn.externals import joblib
import re
from nltk.corpus import stopwords
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import nltk
from gensim.models import Word2Vec
import numpy as np
import gensim.models as kv
import gensim
import logging

def lemmatization(text,morphing = True):
    stop_words = stopwords.words('russian')
    border = text.find('Для решения инцидента заявитель должен предоставить')
    if border > 0:
        text = text[0:border]
    text = text.lower()
    text = re.sub(r"\d+", "", text, flags=re.UNICODE)
    text = re.sub('\W', ' ', text).split()
    if morphing == False:
        return text
    text_new = ''
    x = 0
    previous_word_ne = False
    for item in text:     
        if item == 'не':
            previous_word_ne = True
        
        if len(item) > 2:
            if not (item in stop_words):
                if previous_word_ne:
                    text_new = text_new+'не_'+morph.parse(item)[0].normal_form+' '#'_'+str(morph.parse(item)[0].tag.POS)+' '
                    previous_word_ne = False
                else:
                    text_new = text_new+morph.parse(item)[0].normal_form+' '#'_'+str(morph.parse(item)[0].tag.POS)+' '
                    
    return text_new

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 3:
                continue
            tokens.append(word)
    return tokens
    
def evaluate_prediction(predictions, target, title="Confusion matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    cm = confusion_matrix(target, predictions)    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title + ' Normalized')
    
def predict(vectorizer, classifier, data):
    data_features = vectorizer.transform(data['Description'])
    predictions = classifier.predict(data_features)
    target = data['ShortDescription']
    evaluate_prediction(predictions, target)
    
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])

def find_similar_index(a, A):
    subs = (a[:,None] - A)
    sq_dist = np.einsum('ij,ij->j',subs, subs)
    return sq_dist