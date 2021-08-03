import json
import os

from flask import Flask, Response, send_file
from flask import request
from flask import jsonify
from flask import render_template
from flask_cors import CORS
from flask_dropzone import Dropzone
from src.qa import get_best_answer_and_paragraph
from src.retriever import get_documents_and_scores, update_chapters_embeddings, reset_files, reset_embeddings

_path = os.path.dirname(__file__)
_data_path = os.path.join(_path, '../cache/')

bot_app = Flask('Bot', template_folder=os.path.join(_path, '../templates'))
CORS(bot_app)
dropzone = Dropzone(bot_app)


def root_dir():
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)


def create_new_text_from_components(dialogue, query, answer):
    prompt = dialogue
    prompt += 'Q: ' + query.strip() + '\n'
    prompt += 'A: ' + answer.strip() + '\n'
    return prompt


@bot_app.route('/api/bot', methods=['POST'])
def get_bot_reply():
    if request.method != 'POST':
        return []
    data = json.loads(request.data)
    dialogue = data['text']
    query = data['bobline']
    if '?' not in query:
        query = query.strip() + '?'

    try:
        results = get_documents_and_scores(dialogue, query)
        if not results:
            raise RuntimeError()

    except RuntimeError as e:
        return jsonify({'text': create_new_text_from_components(dialogue=dialogue,
                                                                query=query,
                                                                answer='unknown'),
                        'paragraph': 'NO DOCUMENT HAS BEEN LOADED YET!'})

    answer, paragraph = get_best_answer_and_paragraph(results, dialogue, query)
    return jsonify({'text': create_new_text_from_components(dialogue=dialogue,
                                                            query=query,
                                                            answer=answer),
                    'paragraph': paragraph})


@bot_app.route('/', defaults={'path': 'index.html'})
@bot_app.route('/static/<path>')
def get_resource(path):
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    if 'index.html' in path:
        return render_template('index.html')

    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    if mimetype != "text/html":
        path = 'static/' + path
    complete_path = os.path.join(root_dir(), path)
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)


@bot_app.route('/images/<path>')
def get_image(path):
    path = 'images/' + path
    filename = os.path.join(root_dir(), path)
    return send_file(filename, mimetype='image/gif')


@bot_app.route('/transfer_file', methods=['POST'])
def transfer_file():
    file_paths = json.load(open(os.path.join(_path, '../cache/files.json')))

    if request.method == 'POST':
        f = request.files.get('file')
        file_path = os.path.join(_data_path, f.filename)
        f.save(file_path)
        file_paths.append(file_path)

    json.dump(file_paths, open(os.path.join(
        _path, '../cache/files.json'), 'w'))

    update_chapters_embeddings()

    return render_template('index.html')


if __name__ == '__main__':
    reset_files()
    reset_embeddings()
    port = 8010
    bot_app.run(debug=True, port=port, host='0.0.0.0', use_reloader=False)
