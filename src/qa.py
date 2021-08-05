import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

_question_prompt = '\nQ: '
_answer_prompt = '\nA: '

_path = os.path.dirname(__file__)
_forbidden_words = set([item.strip().lower()
                        for item in open(os.path.join(_path, '../data/bad-words.txt')).readlines()])

_logger = logging.getLogger(__file__)
_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
_model = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(os.path.join(_path, '../models/save_small6'), map_location=device)
_model.load_state_dict(checkpoint['model_state_dict'])
_model = _model.to(device)


def get_text_up_to_question_number(text, number):
    pos = text.find(_answer_prompt)
    for _ in range(number):
        pos = text.find(_answer_prompt, pos + 1)
    return text[0:pos + 1]


def get_answers_number(text):
    return text.count(_answer_prompt)


def get_answer_number(text, number):
    pos = text.find(_answer_prompt)
    for _ in range(number):
        pos = text.find(_answer_prompt, pos + 1)
    end = text.find('\n', pos + len(_answer_prompt))
    return text[pos + len(_answer_prompt):end]


def get_question_number(text, number):
    pos = text.find(_question_prompt)
    for _ in range(number):
        pos = text.find(_question_prompt, pos + 1)
    end = text.find('\n', pos + len(_question_prompt))
    return text[pos + len(_question_prompt):end]


def get_all_answers(dev_dict, dev_index):
    answers = [[item['input_text']
                for item in dev_dict['data'][dev_index]['answers']]]
    answers += [[item['input_text'] for item in dev_dict['data']
    [dev_index]['additional_answers'][str(index)]] for index in range(3)]
    return [list(set([answers[j][i] for j in range(len(answers))])) for i in range(len(answers[0]))]


def generate_answer(text, query, dialogue, length):
    text = text.strip()
    query = query.strip()
    dialogue = dialogue.strip()

    prompt = 'In the text below two people are discussing a story.\n\n'
    prompt += 'Story:\n' + text + '\n\n'
    prompt += 'Discussion:\n'
    prompt += dialogue
    prompt += '\nQ: ' + query + '\n'
    tokens = _tokenizer.encode(prompt, return_tensors='pt')
    tokens_length = tokens.shape[1]
    out_tokens = _model.generate(
        tokens.to(device),
        max_length=tokens_length + length,
        temperature=0,
        pad_token_id=50256
    )
    generated_text = _tokenizer.decode(out_tokens[:, tokens_length:][0], skip_special_tokens=True)
    score = float(_model(out_tokens, labels=out_tokens)[0])
    start = 0
    end = generated_text.find('\n', start + 1)
    if end == -1:
        end = len(generated_text)
    answer = generated_text[start:end + 1].split('A:')[-1].strip()
    if len(set(answer.split()) & _forbidden_words) > 0:
        _logger.warning("A forbidden word was caught in the answer!")
        answer = 'unknown'
    return answer, score


def get_best_answer_and_paragraph(results, dialogue, query):
    answers_and_scores = []
    for paragraph, similarity in results:
        answer, perplexity = generate_answer(paragraph, query, dialogue, length=50)
        score = 0.02 * perplexity + (1 - similarity)
        answers_and_scores.append((answer, score, paragraph))

    answers_and_scores = sorted(answers_and_scores, key=lambda x: x[1])
    final_answer = 'unknown'
    final_paragraph = ''
    index = 0
    while final_answer == 'unknown':
        final_answer = answers_and_scores[index][0]
        final_paragraph = answers_and_scores[index][2]
        index += 1

    return final_answer, final_paragraph
