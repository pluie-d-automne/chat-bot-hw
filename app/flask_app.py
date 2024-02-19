from flask import Flask, render_template, request
import random

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('base.html')

@app.route("/get")
def get_homer_response():    
    userText = request.args.get('msg')    
    return random.choice(['Hello', '🙈'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')