from __future__ import print_function
# import konlpy.tag
import csv
import numpy as np
import scipy.stats as st
import itertools as it

#from twkorean import TwitterKoreanProcessor

# from gensim.models import Word2Vec
from collections import namedtuple
from gensim.models import doc2vec
from gensim.models import Doc2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

import multiprocessing
from pprint import pprint
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import numpy as np

config = tf.ConfigProto(device_count = {'GPU': 0},log_device_placement=False)

directory = "C:/Users/.../Data/"

def read_comm_csv(category, filename):
    data_table = []
    try:
        with open(directory + category + '/' + filename + '.csv', 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                data_table.append(row)
                print(row)

        return data_table

    except:
        print("CSV reading failed")
        return 0


def print_tokens(tokens, end="\n"):
    if isinstance(tokens, list):
        print("[", end="")
    elif isinstance(tokens, tuple):
        print("(", end="")

    for t in tokens:
        if t != tokens[-1]:
            elem_end = ", "
        else:
            elem_end = ""

        if isinstance(t, (list, tuple)):
            print_tokens(t, end=elem_end)
        else:
            print(t, end=elem_end)

    if isinstance(tokens, list):
        print("]", end=end)
    elif isinstance(tokens, tuple):
        print(")", end=end)


def comm_process(data_table):
    morphs_list = [["tot_ID", "Comment_ID", "Article_ID", "Comment_Date", "Recom", "Unrecom", "Comment_morphs"]]

    for i in range(len(data_table)):
        if i == 0: continue
        parsing_object = data_table[i][6]
        processor = TwitterKoreanProcessor()

        tokens = processor.tokenize(parsing_object)
        sen_morphs_list = []
        for token in tokens:
            if token[1] in ("Noun", "Verb", "Adjective", "Adverb", "KoreanParticle", "Alpha", "Number"):
                sen_morphs_list.append(token[0])

        print(sen_morphs_list)
        morphs_list.append(data_table[i][:6] + sen_morphs_list)  # [data_table[i][8]]+

    return morphs_list


def label_comm_process(data_table):
    morphs_list = [
        ["tot_ID", "Comment_ID", "Article_ID", "Comment_Date", "Recom", "Unrecom", "label", "Comment_morphs"]]

    for i in range(len(data_table)):
        if i == 0: continue
        parsing_object = data_table[i][7]
        processor = TwitterKoreanProcessor()

        tokens = processor.tokenize(parsing_object)
        sen_morphs_list = []
        for token in tokens:
            if token[1] in ("Noun", "Verb", "Adjective", "Adverb", "KoreanParticle", "Alpha", "Number"):
                sen_morphs_list.append(token[0])
        morphs_list.append(data_table[i][:7] + [data_table[i][8]] + sen_morphs_list)
        print(sen_morphs_list)

    return morphs_list


def write_csv(morphs_list, category, filename):
    try:
        with open(directory + category + '/processed_' + filename + '.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for items in morphs_list:
                filewriter.writerow(items)

        print("CSV Write complete")
    except:
        print("Error raised")


def extract_word(data_table):
    tot_word_list = []

    for i in range(len(data_table)):
        if i == 0: continue
        word_list = []
        for morph in data_table[i][8:]:
            if not morph: continue
            word_list.append(morph.split('/')[0])
        tot_word_list.append((word_list, ""))
        # print(word_list)

    return tot_word_list


def morph_extract(data_table):
    tot_word_list = []

    for line in data_table:
        word_list = []
        for morph in line[8:]:
            if not morph: continue
            word_list.append(morph)
        tot_word_list.append((word_list, line[7]))  # or ""

    return tot_word_list


def doc2vec_modeling(train_data, test_data, model, fold, n='0'):
    # doc2vec parameters
    cores = multiprocessing.cpu_count()

    vector_size = 300
    window_size = 15
    word_min_count = 5
    sampling_threshold = 1e-3
    negative_size = 5
    train_epoch = 10
    dm = 1
    worker_count = cores

    # doc2vec 에서 필요한 데이터 형식으로 변경
    TaggedDocument = namedtuple('TaggedDocument', 'words tags')
    tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_data]
    tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_data]
    # print(train_data)

    # 사전 구축
    doc_vectorizer = doc2vec.Doc2Vec(size=vector_size, window=window_size, alpha=0.025, min_alpha=0.025, seed=1234,
                                     workers=worker_count)
    doc_vectorizer.build_vocab(tagged_train_docs)

    # Train document vectors!
    for epoch in range(train_epoch):
        doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
        doc_vectorizer.alpha -= 0.002  # decrease the learning rate
        doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay

    # To save
    doc_vectorizer.save('model/doc2vec_' + model + '_' + str(n) + '_' + str(fold) + 'fold.model')

    # pprint(doc_vectorizer.most_similar('한국'))
    # pprint(doc_vectorizer.similarity('한국', 'ㅋㅋ'))

    # load train data
    doc_vectorizer = Doc2Vec.load('model/doc2vec_' + model + '_' + str(n) + '_' + str(fold) + 'fold.model')

    # 분류를 위한 피쳐 생성
    train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
    train_y = [doc.tags[0] for doc in tagged_train_docs]
    test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
    test_y = [doc.tags[0] for doc in tagged_test_docs]
	
    #classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    if model =='svm':
        #parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000], 'gamma': [0.8, 0.5, 0.2, 0.1, 0.01]}
        parameters = {'C': st.expon(scale=100), 'gamma': st.expon(scale=.1),
                      'kernel': ['rbf'], 'class_weight': ['balanced', None]}
        svc = SVC(decision_function_shape='ovo', random_state = 1234)
        #classifier = GridSearchCV(svc, parameters)
        classifier = RandomizedSearchCV(svc, parameters)

        classifier.fit(train_x, train_y)
        print(classifier.best_params_)

        # 테스트 score 확인
        print(classifier.score(test_x, test_y))
        score = classifier.score(test_x, test_y)

        # save the model to disk
        filename = 'model/finalized_model_' + model + '_' + str(n) + '_' + str(fold) + 'fold.sav'
        pickle.dump(classifier, open(filename, 'wb'))

        return score
        #classifier = SVC(C =100, gamma = 0.05, decision_function_shape='ovo', kernel = 'rbf', random_state =1234)
	
	'''
	모델 수정하여 다시 업로드
	'''
    


def doc2vec_embedding(new_data, best_index, model, fold):
    # doc2vec 에서 필요한 데이터 형식으로 변경
    # run_data = extract_word(new_data)
    run_data = morph_extract(new_data)
    TaggedDocument = namedtuple('TaggedDocument', 'words tags')
    tagged_run_docs = [TaggedDocument(d, [c]) for d, c in run_data]
    print(run_data)

    # load train data
    doc_vectorizer = Doc2Vec.load('model/doc2vec_' + model + '_' + str(best_index) + '_' + str(fold) + 'fold.model')

    # 분류를 위한 피쳐 생성
    run_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_run_docs]
    run_y = [doc.tags[0] for doc in tagged_run_docs]

    # load the model from disk
    filename = 'model/finalized_model_' + str(best_index) + '.sav'

    # 실제 분류 확인
    loaded_model = pickle.load(open(filename, 'rb'))

    result_data = []
    for i in range(len(run_x)):
        line = list(new_data[i][:7])

        if i == 0:
            line.append('predicted')
        else:
            line.append(loaded_model.predict(run_x[i].reshape(1, -1))[0])

        # cross test column
        '''
        if i == 0:
            line.append('labeled')
        else:
            line.append(run_y[i])
        '''

        line.extend(new_data[i][7:])  # [8:])은 cross test 뽑을 때 사용
        result_data.append(line)
        print(line)

    return result_data


def data_processing(category, type):
    data_table = read_comm_csv(category, type)
    if (data_table == 0): return 0

    if "labeled" in type:
        morphs_list = label_comm_process(data_table)
    else:
        morphs_list = comm_process(data_table)

    return morphs_list


def build_model(processed_data_table, model, n):
    processed_data_table.pop(0)
    kf = StratifiedKFold(n_splits=n, random_state=1234)  # stratified n-fold

    y = []
    #category fold
    for item in processed_data_table: y.append(item[3])
    '''
    #category+label fold
    for item in processed_data_table:
        if item[7]=='1': y.append(item[3]+'_2')
        else: y.append(item[3]+'_'+item[7])
    '''
    scores = np.zeros(n)
    for i, (train_index, test_index) in enumerate(kf.split(processed_data_table, y)):
        processed_train = np.array(processed_data_table)[train_index]
        processed_test = np.array(processed_data_table)[test_index]

        tokenized_train_contents = morph_extract(processed_train)
        tokenized_test_contents = morph_extract(processed_test)

        label_0 = []
        label_1 = []
        label_2 = []

        for item in tokenized_train_contents:
            if item[1] == '0':
                label_0.append(item)
            elif item[1] == '1':
                label_1.append(item)
            else:
                label_2.append(item)

        label_12 = label_1
        label_12.extend(label_2)

        tot_upsampled_tokenized_train_contents = []

        if len(label_0) >= len(label_1) + len(label_2):
            # Upsample minority class
            label_12_upsampled = resample(label_12, replace=True, n_samples=len(label_0), random_state=123)

            # Combine majority class with upsampled minority class
            tot_upsampled_tokenized_train_contents = label_0
            tot_upsampled_tokenized_train_contents.extend(label_12_upsampled)

        else:
            # Upsample minority class
            label_0_upsampled = resample(label_0, replace=True, n_samples=len(label_12), random_state=123)

            # Combine majority class with upsampled minority class
            tot_upsampled_tokenized_train_contents = label_12
            tot_upsampled_tokenized_train_contents.extend(label_0_upsampled)

        # print(tot_upsampled_tokenized_train_contents)

        print(model + '_' + str(i) + ' is building now... # of fold: ' + str(n))
        scores[i] = doc2vec_modeling(tot_upsampled_tokenized_train_contents, tokenized_test_contents, model, n, i)

    best_model = scores.tolist().index(max(scores))
    print(scores)
    print("Best Model: " + str(scores[best_model]))

    return best_model, scores[best_model]
	

class Batch:

    def __init__(self,x_data, y_data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = x_data
        self._label = y_data
        self._num_examples = x_data.shape[0]
        pass

    @classmethod
    def data(self):
        return self._data
    '''
    @classmethod
    def label(self):
        return self._label
    '''
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch
            data_new_part =  self._data[start:end]
            label_new_part = self._label[start:end]

            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            return self._data[start:end], self._label[start:end]

if __name__ == "__main__":
    category_list = ["tot"]

    for item in category_list:
        '''
        # creating morphs
        morphs = data_processing(item, "labeled_Sample2")
        print(morphs)
        write_csv(morphs, item, "Sample2")
        '''
        morphs = []
        try:
            with open('tot/processed_labeled_Sample2.csv', 'r', newline='') as csvfile:
                f = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for line in f:
                    print(line)
                    morphs.append(line)
                f.close()
                print("CSV Read complete")
        except:
            print("Error raised")

        # build model
        model_results = [["model", "fold", "best_score", "index"]]
		
        models = ['RF', 'AdaBoost']
        n = [10]
		
        for model in models:
            for nfold in n:
                best_model_index, best_score = build_model(morphs, model, nfold)
                model_results.append([model, nfold, best_score, best_model_index])
		
        # write_csv(model_results, item, "model_score_list")
		
        '''
        #new data parsing
        new_morphs = data_processing(item, "Comment")
        write_csv(new_morphs, item, "Comment")
        '''
		
        # apply model
        # new_data = processed_test
        # new_data =data_processing(item, "raw_Comment_life")
        '''
        new_data = read_comm_csv(item, "processed_Comment_tot")
        result_data = doc2vec_embedding(new_data, best_model_index, model, fold)
        write_csv(result_data, "sample", "label_with_model_tot")
        '''