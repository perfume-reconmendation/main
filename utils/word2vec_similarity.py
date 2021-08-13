import json
from utils.preprocessor import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize, sent_tokenize
import copy
import pickle
from collections import OrderedDict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

def sentence_preprocessing(user_sentence):
  # 사용자 입력 문장 전처리
  _stopwords = set(json.load(open('dataset/stopwords.json', 'r')))
  tokenized_doc = preprocessing(user_sentence, _stopwords)
  return tokenized_doc

'''문서 벡터화'''
def vectors(model_path, document_list):
    # 모델 로드
    from gensim.models import Word2Vec, KeyedVectors
    word2vec_model = KeyedVectors.load_word2vec_format(model_path)

    document_embedding_list = []

    doc2vec = None
    count = 0
    for word in document_list:
        if word in word2vec_model.vocab:
            count += 1
            # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
            if doc2vec is None:
                doc2vec = word2vec_model[word]
            else:
                doc2vec = doc2vec + word2vec_model[word]
    
    if doc2vec is None:
        doc2vec = np.empty(100,)
        doc2vec[:] = 0
        document_embedding_list.append(doc2vec)
    else:
        # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
        doc2vec = doc2vec / count
        document_embedding_list.append(doc2vec)

    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list

'''문서간 코사인 유사도 측정'''
def recommendations(df_path, document_embedding_list, label):

    df = pd.read_csv(df_path)
    same_label_df = df[df['label']==label]

    # 같은 라벨의 문서들의 벡터를 불러와서 사용자 문서 벡터를 append
    dir = 'dataset/docvec_'
    path = dir + str(label)+'.pickle'
    with open(path, 'rb') as f:
      same_laber_vec = pickle.load(f)
    
    same_laber_vec.extend(document_embedding_list)

    # 다른 문서들과의 유사도 측정
    similarity = cosine_similarity([same_laber_vec[-1]], same_laber_vec[0:-1])

    # 전체 cosine유사도 행렬에서 사용자 입력 문장과 가장 유사한 순으로 리뷰 정렬
    sim_scores = list(enumerate(similarity.reshape(-1,1)))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:10]

    # 가장 유사한 리뷰 10개의 인덱스
    per_indices = [i[0] for i in sim_scores]

    # 전체 데이터프레임에서 해당 인덱스의 행만 추출.
    recommend = same_label_df.iloc[per_indices].reset_index(drop=True)

    top3_df = pd.DataFrame(columns=['name','similarity','review','accords'])

    # 데이터프레임으로부터 순차적으로 출력
    recommend_perfume = []
    for index, row in recommend.iterrows():
      if len(recommend_perfume)==3:
        break
      if row['name'] in recommend_perfume:
        continue
      else:
        recommend_perfume.append(row['name'])
        top3_df = top3_df.append({'name':row['name'], 'similarity':round(sim_scores[index][1][0],4), 'review':row['review'], 'accords':row['accords']},ignore_index=True)
      # print('Top {}'.format(len(recommend_perfume)))
      # print('향수 명: ' ,row['name'])
      # print('유사도: ',sim_scores[index][1])
      # print('리뷰: ', row['review'])
      # print()
      # print()
    
    return top3_df.to_json(orient = 'records')

def word2vec_similarity(user_sentence, label):
  df_path = 'dataset/dataset_210626_215600.csv'
  model_path = 'model/w2v_10window'

  final_result = sentence_preprocessing(user_sentence)
  document_embedding_list = vectors(model_path, final_result)
  top3_json = recommendations(df_path, document_embedding_list, label)

  return top3_json


if __name__=="__main__":
    user_sentence = 'The guitarist of the band Sensual and sexy Wearing a shirt and ripped jeans Sweet and drowsy eyes He soaked in sweat in the heat of the stage'
    label=2
    print(word2vec_similarity(user_sentence, label))