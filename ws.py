from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from jsonrpc import JSONRPCResponseManager, dispatcher
import xml.etree.ElementTree as ET
import logging
import ast
import json
from ResultFromHelpDesk import get_single_incident
import ml
import gensim.models as kv
from sklearn.externals import joblib
import pyodbc
import pandas as pd


class ServerJson(object):
    '''
    classdocs
    '''
    kv_inc = 0
    logreg = 0
    logging.basicConfig(filename='../Werkzeug/logs/myapp.log', level=logging.INFO)
    

    def prepare_vector_matrix(self, number):
        cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=MyServer;DATABASE=MyDataBase;UID=MyUser;PWD=MyPassword')
        cursor = cnxn.cursor()
        sql = '''
        SELECT Number, Description, ShortDescription, DateCreate, Status_New 
        FROM [SMP].[dbo].[OP_Request] (nolock)
        where RoleAssigned in ('IT Инж 2 ур 1С', 'IT Инж 2 ур 1С эксперт')
        and ShortDescription like '1C Retail%'
        and Status_New in ('Устранение инцидента', 'Ожидание')'''
        df_result = pd.read_sql_query (sql, cnxn)
        df_result['FixedDescription'] = df_result['Description']
        for index, row in df_result.iterrows():
            text = row['Description']
            text_new = ml.lemmatization(text)
            df_result.iat[index,1]=text_new
        full_tokenized = df_result.apply(lambda r: ml.w2v_tokenize_text(r['Description']), axis=1).values
        df_result['full_tokenized'] = full_tokenized
        return (df_result)

    def prepare_vector_matrix_all(self):
        cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=MyServer;DATABASE=MyDataBase;UID=MyUser;PWD=MyPassword')
        cursor = cnxn.cursor()
        sql = '''
        SELECT Number, Description, ShortDescription, DateCreate, Status_New 
        FROM [SMP].[dbo].[OP_Request] (nolock)
        where RoleAssigned in ('IT Инж 2 ур 1С', 'IT Инж 2 ур 1С эксперт')
        and ShortDescription like '1C Retail%'
        and DateCreate > '2019-01-01\''''
        df_result = pd.read_sql_query (sql, cnxn)
        df_result['FixedDescription'] = df_result['Description']
        for index, row in df_result.iterrows():
            text = row['Description']
            text_new = ml.lemmatization(text)
            df_result.iat[index,1]=text_new
        full_tokenized = df_result.apply(lambda r: ml.w2v_tokenize_text(r['Description']), axis=1).values
        df_result['full_tokenized'] = full_tokenized
#        X_full_word_average = ml.word_averaging_list(self.kv_inc,full_tokenized)
        return (df_result)

    def __init__(self):
        '''
        Constructor
        '''        
        logging.info('Constructor') 
        self.kv_inc = kv.KeyedVectors.load_word2vec_format('../Werkzeug/Data/key_vectors_inc.bin', binary=False)
        self.kv_inc.init_sims(replace=True)
        self.logreg = joblib.load('../Werkzeug/data/logreg.sav') 
#        logging.info('Готовим матрицу со всеми закрытыми инцидентами')
        self.df_result = ServerJson.prepare_vector_matrix_all(self) #type DataFrame
        self.X_full_word_average = ml.word_averaging_list(self.kv_inc,self.df_result['full_tokenized'])
#        self.df_result_all_inc = ServerJson.prepare_vector_matrix_all(self)
        logging.info('Выход из конструктора')
              
    
    @Request.application
    def application(self,request):
        dispatcher["get_text_incident"]						= self.get_text_incident
        dispatcher["get_class_incident"]					= self.get_class_incident
        dispatcher["get_similar_incident"]					= self.get_similar_incident
        dispatcher["get_closed_similar_incident"]			= self.get_closed_similar_incident
        dispatcher["get_closed_similar_incident_by_text"]	= self.get_closed_similar_incident_by_text
        dispatcher["get_class_incident_by_text"]  			= self.get_class_incident_by_text
        response = JSONRPCResponseManager.handle(request.data, dispatcher)

    
        return Response(response.json, mimetype='application/json')
    def main(self):
        run_simple('ms-1capp002', 7777, self.application)
    
    def get_text_incident(self, number):
        result = get_single_incident(int(number))
        if result['result'] == 'False':
            result_message = ' инцидент '+number+' в системе не обнаружен'
        else:
            result_message = result['Description']

        return (result_message)

    def get_class_incident(self, number):

        logging.info('Зашли в функцию')
        inc_text = ServerJson.get_text_incident('', int(number))
        inc_lemmatization = (ml.lemmatization(inc_text))
        logging.info('Сделали лемматизацию')
        inc_tokenize = ml.w2v_tokenize_text(inc_lemmatization)
        logging.info('Сделали токенизацию')
        inc_average_list = ml.word_averaging(self.kv_inc,inc_tokenize)
        logging.info('Вычислили вектор инцидента')
        predicted = self.logreg.predict([inc_average_list])
        return(predicted[0])

    def get_class_incident_by_text(self, inc_text):

        logging.info('Зашли в функцию')
        inc_lemmatization = (ml.lemmatization(inc_text))
        logging.info('Сделали лемматизацию')
        inc_tokenize = ml.w2v_tokenize_text(inc_lemmatization)
        logging.info('Сделали токенизацию')
        inc_average_list = ml.word_averaging(self.kv_inc,inc_tokenize)
        logging.info('Вычислили вектор инцидента')
        predicted = self.logreg.predict([inc_average_list])
        return(predicted[0])


    def get_similar_incident(self,number):


        df_result = ServerJson.prepare_vector_matrix(self, number) #type DataFrame
        X_full_word_average = ml.word_averaging_list(self.kv_inc,df_result['full_tokenized'])
        inc_text = ServerJson.get_text_incident('', int(number))
        inc_lemmatization = (ml.lemmatization(inc_text))
        inc_tokenize = ml.w2v_tokenize_text(inc_lemmatization)
        inc_average_list = ml.word_averaging(self.kv_inc,inc_tokenize)

        df_result['similar_index'] = ml.find_similar_index(inc_average_list,X_full_word_average.T)
        df_order = df_result[df_result['similar_index']<0.3].head(1000)
        return(list(zip(df_order['Number'],df_order['FixedDescription'], df_order['similar_index'], df_order['DateCreate'].astype(str), df_order['Status_New'])))

    def get_closed_similar_incident(self,number):

        df_result = self.df_result
        X_full_word_average = ml.word_averaging_list(self.kv_inc,df_result['full_tokenized'])
        inc_text = ServerJson.get_text_incident('', int(number))
        inc_lemmatization = (ml.lemmatization(inc_text))
        inc_tokenize = ml.w2v_tokenize_text(inc_lemmatization)
        inc_average_list = ml.word_averaging(self.kv_inc,inc_tokenize)

        df_result['similar_index'] = ml.find_similar_index(inc_average_list,X_full_word_average.T)
        df_result = df_result.sort_values('similar_index')
        df_order = df_result[df_result['similar_index']<0.3].head(1000)
        return(list(zip(df_order['Number'],df_order['FixedDescription'], df_order['similar_index'], df_order['DateCreate'].astype(str), df_order['Status_New'])))

    def get_closed_similar_incident_by_text(self,inc_text):

        df_result = self.df_result
        X_full_word_average = self.X_full_word_average#ml.word_averaging_list(self.kv_inc,df_result['full_tokenized'])
        inc_lemmatization = (ml.lemmatization(inc_text))
        inc_tokenize = ml.w2v_tokenize_text(inc_lemmatization)
        inc_average_list = ml.word_averaging(self.kv_inc,inc_tokenize)

        df_result['similar_index'] = ml.find_similar_index(inc_average_list,X_full_word_average.T)
        df_result = df_result.sort_values('similar_index')
        df_order = df_result[df_result['similar_index']<0.3].head(1000)
        return(list(zip(df_order['Number'],df_order['FixedDescription'], df_order['similar_index'], df_order['DateCreate'].astype(str), df_order['Status_New'])))

    
if __name__ == '__main__':
    ServerJson().main()