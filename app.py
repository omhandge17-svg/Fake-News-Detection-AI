from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open("model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = None

    if request.method == "POST":
        news = request.form["news"]

        news_vector = vectorizer.transform([news])

        result = model.predict(news_vector)
        probability = model.predict_proba(news_vector)
        confidence = round(max(probability[0]) * 100, 2)

        if result[0] == 0:
            prediction = "Fake News"
        else:
            prediction = "Real News"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)