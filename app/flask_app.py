from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    image = 'static/images/homer_word_cloud.png'
    return render_template('base.html', main_image = image)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')