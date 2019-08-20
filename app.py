from flask import Flask, render_template, redirect, url_for
from api_fetch import harvest, manipulate
from model_train import trainer

app = Flask(__name__)

# Setting up routes to rule the world
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/update/<which>")
def new_info(which="a"):
    harvest(which)
    return redirect(url_for("home"))

@app.route("/manip")
def manip_info():
    manipulate()
    return redirect(url_for("home"))

@app.route("/predict")
def crystal_ball():
    trainer()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)