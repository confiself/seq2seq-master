#!/usr/bin/env python
import os
import re
import json
from utils import chars
try:
    from utils.chars import tr2sp
except:
    pass


class GenChat(object):
    def __init__(self):
        self.max_encoder_seq_length = 30
        self.max_decoder_seq_length = 30

    @staticmethod
    def create_novel():
        data_path = '/tmp/xhj.csv'
        vocabs = set()
        with open(data_path) as f, open('/tmp/chat/in.txt', 'w') as f_in, open('/tmp/chat/out.txt', 'w') as f_out, \
                open('/tmp/chat/vocabs', 'w') as f_v:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                for word in line:
                    vocabs.add(word)
                line = line.split('\t')
                question = line[0]
                answer = line[1]
                f_in.writelines(' '.join(question) + '\n')
                f_out.writelines(' '.join(answer) + '\n')
            f_v.writelines('<s>' + '\n')
            f_v.writelines('<\s>' + '\n')
            for word in vocabs:
                f_v.writelines(word + '\n')

    @staticmethod
    def get_xiaobing_corpus():
        # data_path = '/tmp/3.log'
        data_path = '/tmp/wei_bo_comments.txt'
        vocabs = set()
        with open(data_path, encoding='utf-8') as f, open('/tmp/chat/in.txt', 'w', encoding='utf-8') as f_in, \
                open('/tmp/chat/out.txt', 'w', encoding='utf-8') as f_out, \
                open('/tmp/chat/vocabs', 'w', encoding='utf-8') as f_v:
            for line in f.readlines():
                w = json.loads(line)
                if w['user_name'] != '小冰':
                    continue
                if 'created_at' in w and w['created_at'] <= '2015-01-01':
                    continue
                text = GenChat.parse_text(w['text'])
                reply_text = GenChat.parse_text(w['reply_text'])
                if not text or not reply_text:
                    continue
                for x in text + reply_text:
                    vocabs.add(x)
                f_in.writelines(' '.join(reply_text) + '\n')
                f_out.writelines(' '.join(text) + '\n')
            f_v.writelines('<s>' + '\n')
            f_v.writelines('</s>' + '\n')
            for word in vocabs:
                f_v.writelines(word + '\n')

    @staticmethod
    def parse_text(text):
        """
        1 只能是中文或英文常用符号，提取方式为text字段</a>:到最后， reply_text到最后  <span class=\"url-icon\">
        2 "user_name": "小冰"
        3 含有冰/小娜/领养词的不要，超过30个字的不要，[]去掉此间内容，若为4个字也不要，可能是成语
        """
        def _is_valid_word(word):
            if '\u4E00' <= word <= '\u9FFF':
                return True
            if word in ',.?!~、？，。！“”：； …_()（）-|':
                return True
            return False

        keys = ('冰', '小娜', '领养')
        matcher = re.search('</a>:(.*?)(<span class=\"url-icon\">|$)', text)
        if not matcher:
            return None
        words = matcher.group(1)
        if len(words) > 30:
            return None
        if [x for x in keys if x in words]:
            return None
        words = re.sub('\[.*?\]', '', words).strip()
        words = chars.tr2sp(words)
        if not all([_is_valid_word(x) for x in words]):
            return None
        return words


class GenPoetCorpus(object):
    def __init__(self):
        self.embed_size = 100
        self.corpus_file = os.path.join('/opt/app/data/poet/couplet.train')
        self.max_encoder_seq_length = 10
        self.max_decoder_seq_length = 10
        self.corpus_num = 0

    @staticmethod
    def create_novel():
        data_path = '/tmp/xhj.csv'
        vocabs = set()
        with open(data_path) as f, open('/tmp/chat/in.txt', 'w') as f_in, open('/tmp/chat/out.txt', 'w') as f_out, \
                open('/tmp/chat/vocabs', 'w') as f_v:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                for word in line:
                    vocabs.add(word)
                line = line.split('\t')
                question = line[0]
                answer = line[1]
                f_in.writelines(' '.join(question) + '\n')
                f_out.writelines(' '.join(answer) + '\n')
            f_v.writelines('<s>' + '\n')
            f_v.writelines('<\s>' + '\n')
            for word in vocabs:
                f_v.writelines(word + '\n')

    @staticmethod
    def create_seq_corpus(poet_file_path, poet_target_path, vocab):
        """创建供seq训练的语料
        1 in:标题由前后最多共6个字，+当前字组成
        2 out:当前诗句
        :return:
        """

        def _get_title(_poet, index):
            start_index = 0
            stop_index = min(len(_poet), 6)
            if index >= 6:
                stop_index = min(len(_poet), index + 3)
                start_index = stop_index - 6
                start_index = max(0, start_index)
            _title = [_poet[_index][0] for _index in range(start_index, stop_index)] + [_poet[index][0]]
            return u' '.join(_title)

        poet_target_path_in = poet_target_path + '.in'
        poet_target_path_out = poet_target_path + '.out'
        with open(poet_file_path) as f, open(poet_target_path_in, 'a') as f_in, \
                open(poet_target_path_out, 'a') as f_out:
            data = json.load(f)
            for poet in data:
                sentences = []
                for paragraph in poet['paragraphs']:
                    paragraph = paragraph.replace(' ', '')
                    paragraph_sp = tr2sp(paragraph)
                    _sentences = re.split(r'(?:[，。,！？])', paragraph_sp)
                    _sentences = filter(lambda x: x.strip(), _sentences)
                    _sentences = filter(lambda x: len(x) == 7, _sentences)
                    if len(_sentences) % 2 != 0:
                        continue
                    sentences += _sentences
                if len(sentences) < 2:
                    continue
                for i, line in enumerate(sentences):
                    title = _get_title(sentences, i)
                    cur_line = ' '.join(sentences[i])
                    for word in sentences[i]:
                        vocab.add(word)
                    f_in.writelines(title + '\n')
                    f_out.writelines(cur_line + '\n')

    @staticmethod
    def process_couplet():
        in_path = '/opt/app/data/poet/couplet/train/in.txt'
        out_path = '/opt/app/data/poet/couplet/train/out.txt'
        target_path = '/opt/app/data/poet/couplet.train'
        error_count = 0
        with open(in_path) as f_in, open(out_path) as f_out, open(target_path, 'w') as f_w:
            in_lines = f_in.readlines()
            out_lines = f_out.readlines()
            for up_corpus, down_corpus in zip(in_lines, out_lines):
                up_corpus = up_corpus.replace(' ', '').strip()
                down_corpus = down_corpus.replace(' ', '').strip()
                up_corpus = up_corpus
                down_corpus = down_corpus
                up_corpus = re.split(r'(?:[，。,：！、；？])', up_corpus)
                down_corpus = re.split(r'(?:[，。,：、；！？])', down_corpus)
                up_corpus = filter(lambda x: x.strip(), up_corpus)
                down_corpus = filter(lambda x: x.strip(), down_corpus)
                assert len(up_corpus) == len(down_corpus)
                for up, down in zip(up_corpus, down_corpus):
                    if len(up) != len(down):
                        error_count += 1
                        continue
                    f_w.writelines(up.strip() + '\t' + down.strip() + '\n')

if __name__ == '__main__':
    GenChat.get_xiaobing_corpus()
