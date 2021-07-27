from flask import Flask, jsonify, request
from waitress import serve

from classification_service.classify import classify_test_dataset, get_classified_genres, get_titles_from_genre

app = Flask(__name__)

app.config['CORS_HEADER'] = 'Content-Type'

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
    except Exception as e:
        msg = f"classify_N_persist failed with exception: {e}"
        print(msg)
        result = {'response': False, 'msg': msg}
    else:
        msg = f"classify_N_persist worked successfully"
        print(msg)
        result = {'response': True, 'msg': msg}

    return jsonify(result)


@app.route('/classified_genres', methods=['GET'])
def classified_genres():
    try:
        pred_genre = get_classified_genres()
        result = {'response': True, 'msg': pred_genre.to_dict()}

    except Exception as e:
        msg = f"classified_genres failed with exception: {e}"
        print(msg)
        result = {'response': False, 'msg': msg}

    return jsonify(result)


@app.route('/titles_from_genre', methods=['GET'])
def titles_from_genre():
    known_genres = ['folk', 'soul and reggae', 'punk', 'dance and electronica',
                    'metal', 'pop', 'classic pop and rock', 'jazz and blues']

    # validate the genre name
    try:
        if 'genre' in request.args and request.args.get('genre') in known_genres:
            genre = request.args.get('genre')

            # run function to call SQL query
            title_genre = get_titles_from_genre(genre)
            result = {'response': True, 'genre': genre,'msg': title_genre.to_dict()}

        else: # return invalid response
            result = {'response': False, 'msg':'invalid input'}

    except Exception as e: # for Exception cases where database does not exist, SQL query fail, etc...
        msg = f"titles_from_genre failed with exception: {e}"
        print(msg)
        result = {'response': False, 'msg': msg}

    return jsonify(result)


def start_web_service():
    serve(app, host='0.0.0.0', port=8080)


if __name__ == '__main__':
    start_web_service()
