#! coding:utf-8
from flask import Flask, jsonify
from model import Model

app = Flask(__name__)

vocab_file = '/opt/app/data/poet_seq/vocabs'
model_dir = '/opt/app/data/poet_seq/output'

m = Model(
        None, None, None, None, vocab_file,
        num_units=768, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)


@app.route('/chat/couplet/<in_str>')
def chat_couplet(in_str):
    if len(in_str) == 0 or len(in_str) > 50:
        output = u'您的输入太长了'
    else:
        output = m.infer(' '.join(in_str))
        output = '|'.join(output)
    print(u'上联：%s；下联：%s' % (in_str, output))
    return jsonify({'output': output})

app.run()

