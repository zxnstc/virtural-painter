# 最终运行文件
from flask import Flask, render_template
import VirtualPainter

app = Flask(__name__)


@app.route('/')
def background():
    return render_template("homepage.html")


@app.route('/execute')
def camera():
    return VirtualPainter.main()


@app.route('/thank')
def finish():
    return render_template("thanks.html")


if __name__ == '__main__':
    # db.create_all()
    app.run(debug=True)
