import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import requests


class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK, our_model, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print("Using trainable embedding")
            self.w2i, word_emb_val = word_emb
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(
                torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_emb = word_emb
            print("Using fixed embedding")

    def gen_x_batch(self, q, col):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            if self.trainable:
                q_val = [self.w2i.get(x, 0) for x in one_q]
                val_embs.append([1] + q_val + [2])  # <BEG> and <END>
            else:
                # print (i)
                # print ([x.encode('utf-8') for x in one_q])
                # 问题的字转字向量
                q_val = [self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_q]
                # q_val = self.bert_encode(''.join(one_q))
                # print (q_val)
                # print ("#"*60)
                # 加上起始标记位
                val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [
                    np.zeros(self.N_word, dtype=np.float32)])  # <BEG> and <END>
            # exit(0)
            # 问题长度
            val_len[i] = len(q_val) + 2
        # 最长的问题长度
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i, t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            # 因为每个序列长度不一样，所以这里把长度改成一样长的
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i, t, :] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        # 返回问题的字向量和问题的长度
        return val_inp_var, val_len

    def gen_x_batch_bert(self, q, col):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            q_val = self.bert_encode(''.join(one_q))
            val_embs.append(q_val)
            val_len[i] = len(one_q)

        val_inp = torch.from_numpy(np.array(val_embs))
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def gen_col_batch_bert(self, cols):
        name_inp_var = []
        col_name_len = []
        for col in cols:
            c = '[SEP]'.join(col)
            col_name_len.append(len(col))
            name_inp_var.append(self.bert_encode(c))

        val_inp = torch.from_numpy(np.array(name_inp_var))
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var, col_name_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i, t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i, t, :] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

    def bert_encode(self, content):
        import json
        res = requests.get('http://192.168.1.97:8000/bert/?content=' + content).text
        r = json.loads(res).get('encode')

        return r
