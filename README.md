# Perfume Recommendation from Fragrantica

## 개발 환경

- python 3.7
- conda 4.10.1
- MacOS 11.4

## 초기 설정 방법

### 의존성 설치

[링크참고](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

environments.yml 을 이용해 해당 conda 환경에 requirements 를 설치한다.

```shell
conda env create -f environment.yml
conda activate perfume_recommendation
```

이후 개발중에 requirements 를 추가하게 되면 아래의 command 를 통해 `environment.yml` 에 반영하도록 한다.

```shell
conda env export > environment.yml
```

### Dataset 및 모델 다운로드

<https://www.notion.so/sejongai/Dataset-ff4e5acfdf6f444cb2d4d4d892665560>

위의 링크에서 `dataset_210626_215600.csv` 을 다운받아 `/data` 에 배치한다.

<https://drive.google.com/drive/folders/1-V_yRIoQH9C-2iuA7L7F_KeUd1vQ67EA>

위의 링크를 폴더 째로 다운받아서 `src/model` 에 위치시킨다.

<https://drive.google.com/drive/folders/1-5fk-xSCmLJE8SWkt7rI_mxhM2AlDQl8>

위의 링크를 폴더 째로 다운받아서 `src/tokenizer` 에 위치시킨다.

<https://www.notion.so/sejongai/recommend-cc741512dd1f43fd980c88b530c47344>

위의 링크에서 w2v 관련 모델 및 dataset 다운로드 및 배치

아래 그림을 참고해서 dataset 과 모델을 위치시킨다.
![img.png](doc_assets/img_01.png)

### 2021-08-09 추가 사항
<https://www.notion.so/doc2vec-model-df02fdf44d3643de9e3ff09aedf98666>

위의 링크에서 모든 파일을 다운받아 `/doc2vec_model` 에 배치한다.

<https://www.notion.so/sejongai/doc2vec-BERT-rec-555a7f742f35412cba635e8a1c12bd50>

위의 링크에서 
1. BERT에서 'ber_vec_label0.npy' 'ber_vec_label1.npy', 'ber_vec_label2.npy', 'ber_vec_label3.npy' 을 다운받아 `/bert_vec` 에 배치한다.
2. D2V에서  'doc_vec_label0.npy', 'doc_vec_label1.npy', 'doc_vec_label2.npy', 'doc_vec_label3.npy' 을 다운받아 `/doc_vec` 에 배치한다.
3. 공통 사용에서 'compact_label0.csv', 'compact_label1.csv', 'compact_label2.csv', 'compact_label3.csv' 을 다운받아 `/data` 에 배치한다.

### nltk model 및 dataset 다운로드
아래 커맨드를 clone 한 뒤 최초 1회 실행한다.
```shell
python preset.py
```

## Docker

<http://www.science.smith.edu/dftwiki/index.php/Tutorial:_Docker_Anaconda_Python_--_4>
<https://hub.docker.com/r/continuumio/anaconda3/dockerfile>
<https://pythonspeed.com/articles/activate-conda-dockerfile>

도커 아나콘다 이미지 만들기 가이드.

### BUILD
```shell
docker build . -t asia.gcr.io/sai-perfume-recommendation/inferer
```

### DEPLOY
```shell
docker push asia.gcr.io/sai-perfume-recommendation/inferer
```