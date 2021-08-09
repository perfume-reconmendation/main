from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


def load_model(model_path, tokenizer_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

m_model, m_tokenizer = load_model('./src/model', './src/tokenizer')

def test(text, model, tokenizer):
    model.to(device)
    model.eval()

    MAX_LEN = 256

    sentence = text
    sentences = ["[CLS] " + sentence + " [SEP]"]

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(
        input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    test_inputs = torch.tensor(input_ids).to(device)
    test_masks = torch.tensor(attention_masks).to(device)

    outputs = model(test_inputs, token_type_ids=None,
                    attention_mask=test_masks)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    return logits.argmax(1)[0]


if __name__ == "__main__":
    m_model, m_tokenizer = load_model('./src/model', './src/tokenizer')

    test("i feel the wind while looking at the night sky in summer.",
         m_model, m_tokenizer)
    test("it's like having a strong spice in your mouth.", m_model, m_tokenizer,)


def classifier(text):
    '''
    return (0, 1, 2, 3)
    '''
    return test(text, m_model, m_tokenizer)
    # test("it's like having a strong spice in your mouth.", m_model, m_tokenizer)
