from flask import Flask, render_template, request, jsonify
import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
app = Flask(__name__)

template = """Your name is Senor, you are a legal assistent, greet people properly and provide them with legal assistence for their {question} provide short and apt answers."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Set up the OpenAI API key
repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"max_length": 256, "temperature": 0.5})
llm_chain = LLMChain(prompt=prompt, llm=llm)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if question:
        response = llm_chain.run(question)
        return jsonify({'response': response})
    else:
        return jsonify({'response': ''})


if __name__ == '__main__':
    app.run(debug=True)
