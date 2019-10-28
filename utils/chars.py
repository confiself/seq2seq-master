#!/usr/bin/env python

from . import langconv

punctuation = "！＂＃＄％＆＇（）＊＋，－.／：；＜＝＞？＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏｡。"

punctuation_half_ori = ("""!"#$%&'()*+,-.·/:;<=>?@[\]^_`{|}~
                    ｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏｡。""")

# 为语音播报准备 +-*/.><=
punctuation_half_voice = """!"#$%&'(),:;?@[\]^_`{|}~｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏｡。"""

punctuation_half = ("""',;[\]^`{|}
                    ｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏｡。""")

CN_MAPPING = ('零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
              '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九')
DIGIT_MAPPING = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                 '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
P0 = ('', '十', '百', '千',)


def is_alpha_num(word):
    if '\u0030' <= word <= '\u0039':
        return True
    elif '\u0041' <= word <= '\u005a' or '\u0061' <= word <= '\u007a':
        return True


def tr2sp(sentence):
    """将sentence中的繁体字转为简体字
    :param sentence: unicode
    :return :unicode
    """
    sentence = langconv.Converter('zh-hans').convert(sentence)
    return sentence


def sp2tr(sentence):
    """将sentence中的简体字转为繁体字
    :param sentence: unicode
    :return :unicode
    """
    sentence = langconv.Converter('zh-hant').convert(sentence)
    return sentence


def is_alphabet(uchar):
    if ('\u0041' <= uchar <= '\u005a') or \
            ('\u0061' <= uchar <= '\u007a'):
        return True
    else:
        return False

