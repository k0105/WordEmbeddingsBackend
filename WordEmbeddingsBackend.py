from flask import Flask, request, jsonify
from flask.ext.restful import Resource, Api, reqparse
import gensim, os
from gensim.models.word2vec import Word2Vec as w
# from gensim import utils, matutils
import argparse
from time import gmtime, strftime
import base64

app = Flask(__name__)
api = Api(app)

def __init__(self, dirname):
    self.dirname = dirname


def __iter__(self):
    for fname in os.listdir(self.dirname):
        for line in open(os.path.join(self.dirname, fname)):
            yield line.split()

# This REST backend was inspired by https://github.com/3Top/word2vec-api/blob/master/word2vec-api.py

@app.route('/')
def hello_world():
    return 'Welcome to the Distributional Semantics Backend. Please enter a subdirectory to proceed'

# http://localhost:4123/wordemb/ping
class Ping(Resource):
    def get(self):
        return strftime("%Y-%m-%d %H:%M:%S", gmtime())

# http://localhost:4123/wordemb/doesntmatch?set=cat%20dog%20horse%20rabbit%20tiger%20bus
class DoesntMatch(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('set', type=str, required=True, help="Set of words is a mandatory parameter",
                            action='append')
        try:
            return Model.model.doesnt_match(parser.parse_args()['set'][0].split())
        except ValueError as err:
            return 'Value error, {0}'.format(err)

# http://localhost:4123/wordemb/mostsimilar?setPos=king%20woman&setNeg=man
class MostSimilar(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('setPos', type=str, required=True, help="", action='append')
        parser.add_argument('setNeg', type=str, required=True, help="", action='append')
        try:
            return Model.model.most_similar(parser.parse_args()['setPos'][0].split(),
                                            parser.parse_args()['setNeg'][0].split())
        except ValueError as err:
            return 'Value error, {0}'.format(err)


class MostSimilarCosmul(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('setPos', type=str, required=True, help="", action='append')
        parser.add_argument('setNeg', type=str, required=True, help="", action='append')
        try:
            return Model.model.most_similar_cosmul(parser.parse_args()['setPos'][0].split(),
                                                   parser.parse_args()['setNeg'][0].split())
        except ValueError as err:
            return 'Value error, {0}'.format(err)


class NSimilarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('set0', type=str, required=True, help="", action='append')
        parser.add_argument('set1', type=str, required=True, help="", action='append')
        try:
            return Model.model.n_similarity(parser.parse_args()['set0'][0].split(),
                                            parser.parse_args()['set1'][0].split())
        except ValueError as err:
            return 'Value error, {0}'.format(err)


class Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word0', type=str, required=True, help="", action='append')
        parser.add_argument('word1', type=str, required=True, help="", action='append')
        try:
            return Model.model.similarity(parser.parse_args()['word0'][0],
                                          parser.parse_args()['word1'][0])
        except ValueError as err:
            return 'Value error, {0}'.format(err)


class Model(Resource):
    @staticmethod
    def get(self):
        return 'word2vec model'


if __name__ == '__main__':

    #global model

    # Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument('--port', help='Port, default is 4123')
    p.add_argument("--word2vecload", help="Use load_word2vec_format() instead of load()")
    args = p.parse_args()

    # Setup server
    host = 'localhost'
    port = int(args.port) if args.port else 4123
    path = '/wordemb'

    print(os.path.dirname(os.path.realpath(__file__)))

    # Prepare gensim
    load_w2v = True if args.word2vecload else False
    if load_w2v:
        # Note: The following model is huge and will take a model or two to load
        Model.model = gensim.models.Word2Vec.load_word2vec_format('model/model.txt', binary=False)
    else:
        Model.model = gensim.models.Word2Vec.load('model/model.model')
    Model.model.init_sims(replace=True)  # We don't want to modify the model, so trim unnecessary memory

    # elearning plus computer minus bLearning equals multimedia
    # print(model.most_similar(positive=['elearning', 'computer'], negative=['bLearning']))
    # "Personalised learning" minus "teacher"" equals European Multiple MOOC Aggregator", just "personalization" or
    # "adaptation techniques":
    # print(model.most_similar(positive=['personalised_learning'], negative=['teacher']))
    # print(model['blended_learning'])
    # print(model.doesnt_match("elearning blended_learning e_learning computer".split()))

    api.add_resource(Ping, path+'/ping')
    api.add_resource(DoesntMatch, path + '/doesntmatch')
    api.add_resource(MostSimilar, path + '/mostsimilar')
    api.add_resource(MostSimilarCosmul, path + '/mostsimilarcosmul')
    api.add_resource(NSimilarity, path + '/nsimilarity')
    api.add_resource(Similarity, path + '/similarity')

    app.run(host=host, port=port)
