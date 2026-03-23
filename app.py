import html
import re

from flask import Flask, render_template, request

from search_engine import MiniSearchEngine

app = Flask(__name__)
engine = MiniSearchEngine("corpus.json")


def highlight_terms(text, query):
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    unique_tokens = sorted(set(tokens), key=len, reverse=True)
    escaped_text = html.escape(text)

    for token in unique_tokens:
        if not token.strip():
            continue
        pattern = re.compile(rf"(?i)\b({re.escape(token)})\b")
        escaped_text = pattern.sub(r"<mark>\1</mark>", escaped_text)

    return escaped_text


@app.route("/", methods=["GET"])
def index():
    query = request.args.get("q", "").strip()
    payload = None

    if query:
        payload = engine.search(query)
        for result in payload["results"]:
            result["highlighted_text"] = highlight_terms(result["text"], query)

    stats = engine.get_stats()
    return render_template("index.html", query=query, payload=payload, stats=stats)


if __name__ == "__main__":
    app.run(debug=True)
