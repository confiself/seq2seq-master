#!/usr/bin/env python
import threading
from model import Model
from pypinyin import lazy_pinyin
import random
import json
import re
import traceback
import os


class ModelInstance(Model):
    _instance_lock = threading.Lock()

    def __init__(self, vocab_file, model_dir, beam_width):
        super(ModelInstance, self).__init__(None, None, None, None, vocab_file,
                                            num_units=768, layers=4, dropout=0.2,
                                            batch_size=32, learning_rate=0.0001,
                                            output_dir=model_dir,
                                            restore_model=True, init_train=False, init_infer=True,
                                            decode_method='beam', beam_width=beam_width)

        self.high_fre_words = []
        high_fre_words_path = os.path.join(os.path.dirname(vocab_file), 'high_fre_words')
        if os.path.exists(high_fre_words_path):
            with open(high_fre_words_path, encoding='utf-8') as f:
                self.high_fre_words = [x.strip() for x in f.readlines() if x.strip()]

    def get_random_high_fre_word(self):
        if not self.high_fre_words:
            return None
        index = random.randint(0, len(self.high_fre_words) - 1)
        return self.high_fre_words[index]

    @classmethod
    def get_instance(cls, instance, vocab_file, model_dir, beam_width):
        if not hasattr(cls, instance):
            with cls._instance_lock:
                if not hasattr(cls, instance):
                    setattr(cls, instance, ModelInstance(vocab_file, model_dir, beam_width))
        return getattr(cls, instance)


class CoupletModel(object):
    """
    输入不超过15个字
    使用beam_search:30
    根据对仗叠词，选择最合适的三个对联，随机给出
    可能需要更进一步训练
    日志管理操作，能定时清理日志
    
    处理逻辑如下：
    1 提取标点符号位置，去掉标点，组成预测序列，若超过15个字或为空返回错误信息
    2 预测
    3 计算各结果的编码信息比较编码相同数和非重复字数，排序，最多取3个
    4 随机返回最相似中的一个
    5 插入标点
    """

    def __init__(self):
        vocab_file = '/opt/app/data/couplet/vocabs'
        model_dir = '/opt/app/data/couplet/output'
        self.punctuation = '，。：！、；？ '
        self.encode_seq_alpha = 'abcdefghijklmnopqrst'
        # 存放标点符号的位置
        self.punctuation_dict = {}
        self.text = ''
        self.infer_words = ''
        self.model = ModelInstance.get_instance('couplet_model', vocab_file, model_dir, 30)

    def _encode_seq(self, words):
        """按照顺序编码 abcdefg...
        :param words:
        :return:
        """
        seq_dict = {}
        seq = ''
        for word in words:
            if word in self.punctuation:
                seq += word
            elif word in seq_dict:
                seq += seq_dict[word]
            else:
                code = self.encode_seq_alpha[len(seq_dict)]
                seq_dict[word] = code
                seq += code
        return seq

    def _get_best_output(self, outputs):
        ori_code = self._encode_seq(self.infer_words)
        output_scores = {}
        for output in outputs:
            code = self._encode_seq(output)
            unique_words_num = str(len(set(output)))
            # 部分结果长度不匹配
            try:
                same_code_num = str(sum([1 if code[i] == ori_code[i] else 0 for i in range(len(code))]))
            except:
                continue
            if len(same_code_num) == 1:
                same_code_num = '0' + same_code_num
            if len(unique_words_num) == 1:
                unique_words_num = '0' + unique_words_num
            output_scores[output] = same_code_num + unique_words_num
        output_scores_list = sorted(output_scores.items(), key=lambda x: x[1], reverse=True)
        best_result = [output_scores_list[0][0]]
        for i in range(1, 3):
            if i < len(output_scores_list) and output_scores_list[i][1] == output_scores_list[0][1]:
                best_result.append(output_scores_list[i][0])
        if len(best_result) == 1:
            return best_result[0]
        index = random.randint(0, len(best_result) - 1)
        return best_result[index]

    def _extract_punctuation(self):
        infer_words = ''
        for i, word in enumerate(self.text):
            if word in self.punctuation:
                self.punctuation_dict[i] = word
            else:
                infer_words += word
        return infer_words

    def _add_punctuation(self, output):
        text = ''
        output = list(output)
        for i in range(len(output) + len(self.punctuation_dict)):
            if i in self.punctuation_dict:
                text += self.punctuation_dict[i]
            else:
                text += output.pop(0)
        return text

    def predict(self, text):
        try:
            resp = self._predict(text)
        except:
            traceback.print_exc()
            resp = {'error_code': 1002, 'error_msg': '预测失败，内部异常'}
        return json.dumps(resp, ensure_ascii=False)

    def _predict(self, text):
        # 空格替换为逗号
        self.text = re.sub('[{}]+'.format(self.punctuation), '，', text)
        self.text = self.text.strip('，')
        if len(self.text) == 0 or len(self.text) > 15:
            return {'error_code': 1001, 'error_msg': '输入长度过长或为空'}
        self.infer_words = self.text
        outputs = self.model.infer(self.infer_words)
        if not outputs:
            return {'error_code': 1002, 'error_msg': '预测失败，内部异常'}
        outputs = outputs[0]
        output = self._get_best_output(outputs)
        assert len(self.text) == len(output)
        return {'error_code': 0, 'error_msg': '',
                'couplet': {'up': self.text, 'down': output}}


class PoetModel(object):
    """
    根据叠词最少排序，按照押韵，给出最佳组合
    处理逻辑
    1 生成预测序列组(原句+最后一个字)
    2 分别预测
    3 排序，选择叠词最少的句子
    4 统计韵母序列的数目，找出最押韵的字母
    5 随机选择押韵字母对应的诗

    """
    _instance_lock = threading.Lock()

    def __init__(self):
        vocab_file = '/opt/app/data/poet/vocabs'
        model_dir = '/opt/app/data/poet/output'
        self.punctuation = '，。：！、；？ '
        self.pinyin_finals = {'a', 'o', 'e', 'i', '', 'v', 'ai', 'ei', 'ui',
                              'ao', 'o', 'i', 'ie', 've', 'er', 'an', 'en',
                              'in', 'un', 'vn', 'ang', 'eng', 'ing', 'ong'}
        self.model = ModelInstance.get_instance('poet_model', vocab_file, model_dir, 200)

    def predict(self, text):
        try:
            resp = self._predict(text)
        except Exception as e:
            print(e)
            traceback.print_exc()
            resp = {'error_code': 1002, 'error_msg': '内部异常'}
        return json.dumps(resp, ensure_ascii=False)

    @staticmethod
    def _filter_most_words(outputs):
        words_count_dict = {x: len(set(x)) for x in outputs}
        words_count_list = sorted(words_count_dict.items(), key=lambda x: x[1], reverse=True)
        words = []
        for i, item in enumerate(words_count_list):
            if item[1] == words_count_list[0][1]:
                words.append(item[0])
        return words

    def _yun_mu(self, _pinyin):
        if _pinyin[-3:] in self.pinyin_finals:
            return _pinyin[-3:]
        elif _pinyin[-2:] in self.pinyin_finals:
            return _pinyin[-2:]
        elif _pinyin[-1:] in self.pinyin_finals:
            return _pinyin[-1:]
        return _pinyin[-3:]

    def _get_best_poet(self, outputs_list):
        """
        1 找出所有序列
        :param outputs_list:
        :return:
        """
        yun_mu_counts = {}
        outputs_yun_mu_list = []
        for outputs in outputs_list:
            # 单序列韵母去重
            yun_mu_set_single = set()
            outputs_yun_mu_dict = {}
            for output in outputs:
                pinyin = lazy_pinyin(output[-1])[0]
                yun_mu = self._yun_mu(pinyin)
                yun_mu_set_single.add(yun_mu)
                if yun_mu in outputs_yun_mu_dict:
                    outputs_yun_mu_dict[yun_mu].append(output)
                else:
                    outputs_yun_mu_dict[yun_mu] = [output]
            outputs_yun_mu_list.append(outputs_yun_mu_dict)

            # 统计韵母在各序列中出现的次数
            for _yun_mu in yun_mu_set_single:
                if _yun_mu not in self.pinyin_finals:
                    continue
                if _yun_mu in yun_mu_counts:
                    yun_mu_counts[_yun_mu] += 1
                else:
                    yun_mu_counts[_yun_mu] = 1

        # 找出最佳韵母对应的诗，以及重复字数最少的
        yun_mu_counts_list = sorted(yun_mu_counts.items(), key=lambda x: x[1], reverse=True)
        best_yun_mu = None
        if yun_mu_counts_list:
            best_yun_mu = yun_mu_counts_list[0][0]
        poet = []

        words_dict = set()

        def most_words(_outputs):
            if not words_dict:
                return _outputs
            max_num = -1
            best_output = None
            for _output in _outputs:
                most_num = len([x for x in _output if x not in words_dict])
                if most_num > max_num:
                    max_num = most_num
                    best_output = _output
            return [best_output]

        for i, outputs in enumerate(outputs_list):
            if best_yun_mu and best_yun_mu in outputs_yun_mu_list[i] and (i % 2 != 0 or i == 0):
                yun_mu_outputs = outputs_yun_mu_list[i][best_yun_mu]
                if len(yun_mu_outputs) >= 2:
                    outputs = yun_mu_outputs
            outputs = most_words(outputs)
            index = random.randint(0, len(outputs) - 1)
            poet.append(outputs[index])
            for word in outputs[index]:
                words_dict.add(word)
        return poet

    @staticmethod
    def _repair_word(output_list, text):
        """防止第一个字不是藏头情况出现(绝大部分是第一个字)
        :param output_list:
        :param text:
        :return:
        """
        for i, word in enumerate(text):
            outputs = output_list[i]
            outputs = [word + x[1:] for x in outputs]
            output_list[i] = outputs
        for i in range(len(output_list)):
            for j in range(len(output_list[i])):
                words = output_list[i][j]
                while len(words) < 7:
                    words += words[-1]
                if len(words) > 7:
                    words = words[:7]
                output_list[i][j] = words

    @staticmethod
    def is_chinese(word):
        if '\u4E00' <= word <= '\u9FFF':
            return True
        return False

    def to_high_fre_words(self, poet):
        """
        替换乱码字符为高频词
        :param poet:
        :return:
        """
        for i, words in enumerate(poet):
            for word in words:
                if self.is_chinese(word):
                    continue
                high_fre_word = self.model.get_random_high_fre_word()
                if not high_fre_word:
                    continue
                words = words.replace(word, high_fre_word)
                poet[i] = words

    def _predict(self, text):
        text = re.sub('[{}]+'.format(self.punctuation), '', text)
        if len(text) == 0 or len(text) > 8:
            return {'error_code': 1001, 'error_msg': '输入长度过长或为空'}
        # 组成预测序列

        infer_words = []
        for word in text:
            infer_words.append(text + word)
        outputs_list = self.model.infer(infer_words)
        self._repair_word(outputs_list, text)
        outputs_list = [self._filter_most_words(x) for x in outputs_list]
        poet = self._get_best_poet(outputs_list)
        poet = [x.replace(' ', '') for x in poet]
        self.to_high_fre_words(poet)
        assert all([len(x) == 7 for x in poet])
        return {'error_code': 0, 'error_msg': '', 'poet': poet}

