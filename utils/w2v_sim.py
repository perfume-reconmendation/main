import json
import processor
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


def sentence_preprocessing(df_path, tokenized_doc_path, user_sentence, label):
    # 데이터프레임 불러오기
    df = pd.read_csv(df_path)
    indexes = df[df['label'] == label].index

    # 토큰화된 전체 리뷰 불러오기
    with open(tokenized_doc_path, 'rb') as f:
        result = pickle.load(f)

    # 분류된 라벨에 해당하는 리뷰만 추출
    same_label_result = [result[i] for i in indexes]

    # 사용자 입력 문장 전처리
    _stopwords = set(json.load(open('../dataset/stopwords.json', 'r')))
    tokenized_doc = processor.preprocessing(user_sentence, _stopwords)

    # 토큰화된 데이터에 사용자 문장 추가
    final_result = copy.deepcopy(same_label_result)
    final_result.append(tokenized_doc)

    return final_result


'''문서 벡터화'''


def vectors(model_path, document_list):
    # 모델 로드
    from gensim.models import Word2Vec, KeyedVectors
    word2vec_model = KeyedVectors.load_word2vec_format(model_path)

    document_embedding_list = []

    # 각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line:
            if word in word2vec_model.index_to_key:
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
    same_label_df = df[df['label'] == label]

    # 다른 문서들과의 유사도 측정
    similarity = cosine_similarity(
        [document_embedding_list[-1]], document_embedding_list[0:-1])

    # 전체 cosine유사도 행렬에서 사용자 입력 문장과 가장 유사한 순으로 리뷰 정렬
    sim_scores = list(enumerate(similarity.reshape(-1, 1)))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]

    # 가장 유사한 리뷰 10개의 인덱스
    per_indices = [i[0] for i in sim_scores]

    # 전체 데이터프레임에서 해당 인덱스의 행만 추출.
    recommend = same_label_df.iloc[per_indices].reset_index(drop=True)

    top3_df = pd.DataFrame(columns=['name', 'accords', 'similarity', 'review'])

    # 데이터프레임으로부터 순차적으로 출력
    recommend_perfume = []
    for index, row in recommend.iterrows():
        if len(recommend_perfume) == 3:
            break
        if row['name'] in recommend_perfume:
            continue
        else:
            recommend_perfume.append(row['name'])
            top3_df = top3_df.append({'name': row['name'], 'accords': row['accords'],
                                      'similarity': sim_scores[index][1], 'review': row['review']}, ignore_index=True)
        # print('Top {}'.format(len(recommend_perfume)))
        # print('향수 명: ' ,row['name'])
        # print('유사도: ',sim_scores[index][1])
        # print('리뷰: ', row['review'])
        # print()
        # print()

    return top3_df


def word2vec_similarity(user_sentence, label):
    df_path = '../dataset/dataset_210626_215600.csv'
    tokenized_doc_path = '../dataset/tokenized_doc.pickle'
    model_path = './model/w2v_10window'

    final_result = sentence_preprocessing(
        df_path, tokenized_doc_path, user_sentence, label)
    document_embedding_list = vectors(model_path, final_result)
    top3_df = recommendations(df_path, document_embedding_list, label)

    return top3_df.to_dict()


if __name__ == "__main__":
    user_sentence = 'The guitarist of the band Sensual and sexy Wearing a shirt and ripped jeans Sweet and drowsy eyes He soaked in sweat in the heat of the stage'
    label = 2
    print(word2vec_similarity(user_sentence, label))
