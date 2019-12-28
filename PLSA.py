import numpy as np
import codecs
import pandas as pd
import jieba
import re


class pLSA:
    def __init__(self, num_topics, corpus, stop_words, max_iteration=30, threshold=10):
        self.threshold = threshold
        self.K = num_topics
        self._corpus = corpus
        self.max_iteration = max_iteration
        self.word2id = {}
        self.id2word = {}
        index = 0
        word_count_list = []
        for doc in corpus:
            word_count = {}
            for word in doc:
                word = word.lower()
                if word not in stop_words and len(word) > 1 and not re.search(r'[0-9]', word):
                    if word not in self.word2id.keys():
                        self.word2id[word] = index
                        self.id2word[index] = word
                        index += 1
                    if word in word_count.keys():
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
            word_count_list.append(word_count)
        # number of docs
        self.M = len(self._corpus)
        # number of words
        self.N = len(self.word2id)
        # observed data
        self.X = np.zeros([self.M, self.N], np.int8)
        for m in range(self.M):
            for word in word_count_list[m]:
                self.X[m, self.word2id[word]] = word_count_list[m][word]
        # p(z_k, d_i)
        self.doc_topic_matrix = np.random.random([self.M, self.K])
        # p(w_j, z_k)
        self.topic_word_matrix = np.random.random([self.K, self.N])
        # p(z_k| w_j, d_i)
        self.Q = np.zeros([self.M, self.N, self.K])

    @ staticmethod
    def _normalize(matrix):
        result_matrix = matrix
        for i in range(result_matrix.shape[0]):
            normalization = result_matrix[i].sum()
            for j in range(result_matrix.shape[1]):
                result_matrix[i, j] = result_matrix[i, j] / normalization
        return result_matrix

    def _EStep(self):
        # calculate P(z_k| w_j,d_i)
        for m in range(self.M):
            for n in range(self.N):
                sigma = 0
                for k in range(self.K):
                    self.Q[m, n, k] = self.doc_topic_matrix[m, k] * self.topic_word_matrix[k, n]
                    sigma = sigma + self.Q[m, n, k]
                if sigma == 0:
                    for k in range(self.K):
                        self.Q[m, n, k] = 0
                else:
                    for k in range(self.K):
                        self.Q[m, n, k] = self.Q[m, n, k]/sigma

    def _MStep(self):
        # update P(w_j|z_k)
        for k in range(self.K):
            sigma = 0
            for n in range(self.N):
                self.topic_word_matrix[k, n] = 0
                for m in range(self.M):
                    self.topic_word_matrix[k, n] += self.X[m, n] * self.Q[m, n, k]
                sigma += self.topic_word_matrix[k, n]
            if sigma == 0:
                for n in range(self.N):
                    self.topic_word_matrix[k, n] = 1.0 / self.N
            else:
                for n in range(self.N):
                    self.topic_word_matrix[k, n] /= sigma

        # update P(z_k|d_m)
        for m in range(self.M):
            for k in range(self.K):
                sigma = 0
                self.doc_topic_matrix[m, k] = 0
                for n in range(self.N):
                    self.doc_topic_matrix[m, k] += self.X[m, n] * self.Q[m, n, k]
                    sigma += self.X[m, n]
                if sigma == 0:
                    self.doc_topic_matrix[m, k] = 1.0 / self.K
                else:
                    self.doc_topic_matrix[m, k] /= sigma

    def _calculateLoglikelihodd(self):
        likelihood = 0
        for m in range(self.M):
            for n in range(self.N):
                theta = 0
                for k in range(self.K):
                    theta += self.topic_word_matrix[k, n] * self.doc_topic_matrix[m, k]
                if theta > 0:
                    likelihood += self.X[m, n] * np.log(theta)
        return likelihood

    def train(self):
        self.doc_topic_matrix = self._normalize(self.doc_topic_matrix)
        self.topic_word_matrix = self._normalize(self.topic_word_matrix)
        old_likelihood = 1
        new_likelihood = 1
        for epoch in range(self.max_iteration):
            self._EStep()
            self._MStep()
            new_likelihood = self._calculateLoglikelihodd()
            print('iteration:{}\tlikelihood:{}'.format(epoch+1, new_likelihood))
            if old_likelihood != 1 and (new_likelihood - old_likelihood) < self.threshold:
                print('training done, final likelihood:{}'.format(new_likelihood))
                break
            old_likelihood = new_likelihood
        else:
            print('training done, final likelihood:{}'.format(new_likelihood))

    def getTopics(self):
        return np.argmax(self.doc_topic_matrix, axis=1)

    def getTopNWords(self, n=5):
        word_id = []
        for i in range(self.topic_word_matrix.shape[0]):
            word_id.append(self.topic_word_matrix[i].argsort()[:n])
        top_word_df = pd.DataFrame(index=['topic{}'.format(x) for x in range(self.K)],
                                   columns=['word{}'.format(x) for x in range(n)])
        for i in range(len(word_id)):
            for j in range(n):
                top_word_df.loc['topic{}'.format(i), 'word{}'.format(j)] = self.id2word[word_id[i][j]]
        return top_word_df


if __name__ == '__main__':
    data = codecs.open('./dataset.txt', encoding='utf-8')
    stop_words = codecs.open('./stopwords.dic', encoding='utf-8')

    data = [x.strip() for x in data]
    stop_words = [x.strip() for x in stop_words]
    doc = [jieba.cut(x) for x in data]
    plsa = pLSA(corpus=doc, num_topics=10, stop_words=stop_words, max_iteration=5)
    plsa.train()
    result = plsa.getTopNWords(n=5)
    print(result)

