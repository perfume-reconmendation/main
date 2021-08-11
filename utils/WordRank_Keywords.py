from collections import OrderedDict
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")


class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def get_keywords(self, number=10):
        """Return top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        dic = dict()
        for i, (key, value) in enumerate(node_weight.items()):
            dic[key] = value
            if i > number:
                break
        return dic

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


def keyword_highlighter(user_sentence, top3_df_dic, model_path):
    """사용자 입력문장과 추천문장에서 키워드 추출"""
    tr4w = TextRank4Keyword()

    from gensim.models import Word2Vec, KeyedVectors
    word2vec_model = KeyedVectors.load_word2vec_format(model_path)

    # 사용자 입력 문장의 키워드 추출
    user_keyword = []
    tr4w.analyze(user_sentence, candidate_pos=['NOUN', 'PROPN', 'ADJ'], window_size=5, lower=False)
    user_keyword = list(tr4w.get_keywords(100).keys())
    user_keyword = [word for word in user_keyword if word in word2vec_model.index_to_key]  # 임베딩 벡터에 없는 단어는 제외
    # print('사용자 문장에서 추출된 키워드 : ' ,user_keyword)

    # 추천 향수 리뷰의 키워드 추출하여 dataframe에 cloumn으로 저장
    top3_df = pd.DataFrame(top3_df_dic)
    top3_keyword = []
    for i in range(0, len(top3_df)):
        tr4w.analyze(top3_df['review'][i], candidate_pos=['NOUN', 'PROPN', 'ADJ'], window_size=5, lower=False)
        keywords = list(tr4w.get_keywords(100).keys())
        keywords = [word for word in keywords if word in word2vec_model.index_to_key]  # 임베딩 벡터에 없는 단어는 제외
        top3_keyword.append(keywords)

    top3_df['keywords'] = top3_keyword

    """하이라이트 컬러 할당"""
    import random
    import colorsys

    custom_palette = []
    for i in range(0, len(user_keyword)):
        r = random.random()
        h, s, l = r, 1, 0.82
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        color = '#%02x%02x%02x' % (r, g, b)
        custom_palette.append(color)

    # user keyword에 랜덤 파스텔 컬러 할당
    user_dict = {word: custom_palette[i] for i, word in enumerate(user_keyword)}
    user_dict

    # 추천 리뷰 keyword에 컬러 할당
    color_list = []
    for i in range(0, len(top3_df)):
        top3_dict = dict.fromkeys(top3_df['keywords'][i])
        index = 0
        for uw in user_dict.keys():
            for tw in top3_dict.keys():
                # 임계값 0.6로 잡아봄
                if word2vec_model.similarity(uw, tw) > 0.65:
                    # 컬러 할당이 안되어있는 상태라면 처음 값 넣어줌
                    if top3_dict[tw] is None:
                        top3_dict[tw] = list(user_dict.items())[index]
                    # 컬러 할당이 되어있는 상태라면 유사도 더 높은 컬러로 넣어줌
                    elif word2vec_model.similarity(uw, tw) > word2vec_model.similarity(top3_dict[tw][0],
                                                                                       tw):  # 이전 user word와 비교
                        top3_dict[tw] = list(user_dict.items())[index]
            index += 1
        color_list.append(top3_dict)

    # 데이터 프레임에 색상 정보 추가
    top3_df['colors'] = color_list

    # return 구조 수정
    # result에는 사용자 문장과 top3문장의 하이라이트 정보를 포함
    result = {'text': [], 'highlight': []}
    result['text'].append(user_sentence)
    result['highlight'].append(user_dict)

    for i in range(0, len(top3_df)):
        th_dic = {'text': top3_df['review'][i]}
        color_dic = {}
        perfume = top3_df['colors'][i]
        for tw in perfume.keys():
            if perfume[tw] is not None:
                color_dic[tw] = perfume[tw][1]
        result['text'].append(top3_df['review'][i])
        result['highlight'].append(color_dic)

    return result


if __name__ == "__main__":
    import word2vec_similarity
    import pprint

    user_sentence = 'The guitarist of the band Sensual and sexy Wearing a shirt and ripped jeans Sweet and drowsy eyes He soaked in sweat in the heat of the stage'
    label = 2
    top3_df_dic = word2vec_similarity.word2vec_similarity(user_sentence, label)
    model_path = './model/w2v_10window'
    pprint.pprint(keyword_highlighter(user_sentence, top3_df_dic, model_path))
