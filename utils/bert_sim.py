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


# 리뷰 데이터
df = pd.read_csv("data/dataset_210626_215600.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)

# bert 임베딩 벡터
bert_vec = np.load('bert_vec.npy', allow_pickle=True)

# 쿼리
model = SentenceTransformer(
    'sentence-transformers/distiluse-base-multilingual-cased-v1')


def BERT_recommendations(query):
    '''

[{'name': 'Bentley for Men Intense Bentley for men',
  'similarity': array([[0.38496298]], dtype=float32),
  'review': 'Intriguing and sexy but not a massive performer imo. If it was i would wear this a lot more often..',
  'accords': "['woody', 'warm spicy', 'amber', 'rum', 'smoky', 'fresh spicy', 'leather', 'balsamic', 'aromatic', 'cinnamon']"},
 {'name': 'Sauvage Christian Dior for men',
  'similarity': array([[0.36974102]], dtype=float32),
  'review': 'aggressive, provocative, maskulen. a man scent sexy enough to pull the panty down on its own.',
  'accords': "['fresh spicy', 'amber', 'citrus', 'aromatic', 'musky', 'woody', 'herbal', 'warm spicy']"},
 {'name': 'Tuscan Leather Tom Ford for women and men',
  'similarity': array([[0.3611946]], dtype=float32),
  'review': "This perfume makes me feel like I'm being seduced by an Italian man wearing tight acid washed jeans and a tan suede jacket, donning a mustache, and chain smoking cigarettes while waiting at a sweaty subway stop. \r\nIt's sexy, a little dirty, and not for everyone.",
  'accords': "['leather', 'fruity', 'animalic', 'sweet', 'amber', 'smoky']"}]
    '''
    vectors = model.encode(query.lower())

    # 쿼리와 데이터 유사도
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
                'similarity': sim_scores[index][1][0][0],
                'review': row['review'],
                'accords': row['accords']
            }

            result_list.append(result)

    return result_list


# query = 'I am sitting on the beach with a cool breeze I am surrounded by coconut palm water and I sip a refreshing grapefruit sparkling drink'

# BERT_recommendations(query.lower())
