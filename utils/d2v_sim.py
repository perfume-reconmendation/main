def doc2vec(query):
    import pandas as pd
    from tqdm import tqdm
    import os
    from konlpy.tag import Mecab
    import nltk
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    mecab = Mecab()
    df = pd.read_csv('data/dataset_210626_215600.csv')

    model1 = Doc2Vec.load(
        "doc2vec_model/doc2vec_prac_after_preprocessing.model")

    x = model1.docvecs.vectors_docs
    cnt = 0
    for idx, doctag in sorted(model1.docvecs.doctags.items(), key=lambda x: x[1].offset):
        if (cnt % 10000 == 0):
            print(idx, doctag)

    query = mecab.morphs(query.lower())
    new_vector = model1.infer_vector(query)
    # gives you top 10 document tags and their cosine similarity
    sims = model1.docvecs.most_similar([new_vector])

    result_list = []

    for i in range(3):
        result = {
            "name": df.iloc[sims[i][0]].name,
            "similarity": sims[i][1],
            "accords": df.iloc[sims[i][0]].accords,
            "review": df.iloc[sims[i][0]].review
        }

        result_list.append(result)

    return result_list


# doc2vec("A woman I've never met on the street. Lovely and comfortable. The fresh morning air of Paris in the rain. She is wearing a white dress.")
