#! coding:utf-8
from flask import Flask, jsonify
from model import Model

app = Flask(__name__)

vocab_file = '/tmp/chat/vocabs'
model_dir = '/tmp/chat/output_dir'

m = Model(
        None, None, None, None, vocab_file,
        num_units=768, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)


@app.route('/chat/<in_str>')
def chat_couplet(in_str):
    if len(in_str) == 0 or len(in_str) > 50:
        output = u'您的输入太长了'
    else:
        output = m.infer(' '.join(in_str))
    print('输入：{}'.format(in_str))
    print('回复：{}'.format(output[0][0]))
    return jsonify({'output': output})

app.run()

