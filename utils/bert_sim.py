import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize, sent_tokenize

import copy
import pickle
from sentence_transformers import SentenceTransformer

# !pip install sentence_transformers


def BERT_recommendations(query, label):
    # 쿼리
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    vectors = model.encode(query)

    # 향수 리뷰 정보 데이터
    df = pd.read_csv('data/compact_label' + str(label) + '.csv', usecols=['name', 'accords', 'review', 'Unnamed: 0'])
    # bert 임베딩 벡터
    bert_vec = np.load('bert_vec/bert_vec_label' + str(label) + '.npy', allow_pickle=True)

    #  같은 라벨 내에서 쿼리와 데이터 유사도
    similarity = []
    for i in range(len(bert_vec)):
        sim1 = cosine_similarity([vectors], [bert_vec[i]])
        similarity.append(sim1)

    perfumes = df[['name', 'review', 'accords']]

    # 전체 cosine유사도 행렬에서 사용자 입력 문장과 가장 유사한 순으로 리뷰 정렬
    sim_scores = list(enumerate(similarity))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]

    # 가장 유사한 리뷰 10개의 인덱스
    per_indices = [i[0] for i in sim_scores]

    # 전체 데이터프레임에서 해당 인덱스의 행만 추출. 5개의 행을 가진다.
    recommend = df.iloc[per_indices].reset_index(drop=True)

    result_list = []

    # 데이터프레임으로부터 순차적으로 출력
    recommend_perfume = []
    for index, row in recommend.iterrows():
        if len(recommend_perfume) == 3:
            break
        if row['name'] in recommend_perfume:
            continue
        else:
            recommend_perfume.append(row['name'])
            result = {
                'name': row['name'],
                'similarity': round(float(sim_scores[index][1][0][0]), 4),
                'review': row['review'],
                'accords': row['accords']
            }

            result_list.append(result)

    return result_list

# query = 'I am sitting on the beach with a cool breeze I am surrounded by coconut palm water and I sip a refreshing grapefruit sparkling drink'

# BERT_recommendations(query.lower())
