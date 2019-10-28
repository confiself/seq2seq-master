from flask import Flask
from deploy.models import CoupletModel, PoetModel
app = Flask(__name__)


@app.route('/couplet/<in_str>')
def couplet(in_str):
    _couplet_model = CoupletModel()
    return _couplet_model.predict(in_str)


@app.route('/poet/<in_str>')
def poet(in_str):
    poet_model = PoetModel()
    return poet_model.predict(in_str)

if __name__ == '__main__':
    app.run()
