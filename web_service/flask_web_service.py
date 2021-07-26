from flask import Flask, jsonify, request
from waitress import serve

from classification_service.classify import classify_test_dataset, get_classified_genres,get_titles_from_genre

app = Flask(__name__)
# app.config['CORS_HEADER'] = 'Content-Type'

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Credentials'] = 'true'
    header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, PATCH, DELETE'
    return response

@app.route('/classify_N_persist', methods=['POST'])
def classify_N_persist():
    """Method will classify the test dataset and persist only the """
    try:
        classify_test_dataset()
    except Exception:
        result = {'response':False}
    else:
        result = {'response': True}

    return jsonify(result)




# @app.route('/classified_genres', methods=['GET'])
# def classified_genres():
#     pass
#     Xy_test = get_classified_genres()
#     return jsonify(result)
#
# @app.route('/titles_from_genre', methods=['GET'])
# def titles_from_genre():
#     pass
#     return jsonify(result)

def start_web_service():
    host =  '127.0.0.1' #'0.0.0.0'
    port = 8080
    serve(app, host=host, port=port)


if __name__ == '__main__':
     start_web_service()
