#!/usr/bin/env python
# encoding: utf-8
import unittest
import time
from .models import PoetModel, CoupletModel
import json
import logging


class TestCouplet(unittest.TestCase):
    def test_req(self):
        couplet = CoupletModel()
        resp = couplet.predict('松子打孙子   。。。。  松子落孙子落')
        logging.debug(json.dumps(resp, ensure_ascii=False))
        resp = couplet.predict('松子。  ，，，打孙子')
        logging.debug(json.dumps(resp, ensure_ascii=False))

        resp = couplet.predict('十口心思，思国思君思社稷')
        logging.debug(json.dumps(resp, ensure_ascii=False))
        resp = couplet.predict('松子打孙子，松子落孙子落')
        logging.debug(json.dumps(resp, ensure_ascii=False))


class TestPoet(unittest.TestCase):

    def test_req(self):
        poet = PoetModel()
        pre_time = time.time()
        resp = poet.predict('fdj，啊')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')
        pre_time = time.time()
        resp = poet.predict('胡争辉')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')

        pre_time = time.time()
        resp = poet.predict('招商银行')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')

        pre_time = time.time()
        resp = poet.predict('招商银行')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')

        pre_time = time.time()
        resp = poet.predict('招商银行')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')

        pre_time = time.time()
        resp = poet.predict('招商银行')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')

        pre_time = time.time()
        resp = poet.predict('我为秋香')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')

        pre_time = time.time()
        resp = poet.predict('你好')
        print(json.dumps(resp, ensure_ascii=False))
        print(time.time() - pre_time, 's')

        resp = poet.predict('你')
        print(json.dumps(resp, ensure_ascii=False))

        resp = poet.predict('你是谁，啊')
        print(json.dumps(resp, ensure_ascii=False))

    def test_input(self):
        while True:
            poet = PoetModel()
            text = input('>')
            resp = poet.predict(text)
            print(json.dumps(resp, ensure_ascii=False))
