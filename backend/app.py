import os
import random
import yaml
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.corpus import load_index, build_index
from src.retrieval import get_embedder
from src.generation import get_generator
from src.pipelines import rag_fusion, hyde, crag, graph_rag
from src.data_loader import load_examples

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load config
CONFIG_PATH = "config/config.yaml"
if not os.path.exists(CONFIG_PATH):
    # Fallback to example if user hasn't created it yet
    CONFIG_PATH = "config/config.example.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Global instances
CORPUS = None
EMBEDDER = None
GENERATOR = None

def init_app():
    global CORPUS, EMBEDDER, GENERATOR
    
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    api_key = config.get("gemini_api_key", "")
    generation_model = config.get("generation_model", "gemini-1.5-flash")
    index_path = config.get("index_path", "index/crag_index")
    dataset_path = config.get("dataset_path", "dataset/crag_task_1_and_2_dev_v4.jsonl/crag_task_1_and_2_dev_v4.jsonl")
    index_build_limit = config.get("index_build_limit", 500)

    # 1. Load or build index
    if Path(index_path).exists():
        CORPUS = load_index(index_path, embedding_model)
    else:
        print("[backend] Index not found. Building index...")
        CORPUS = build_index(dataset_path, embedding_model, index_path, limit=index_build_limit)

    # 2. Setup embedder and generator
    EMBEDDER = get_embedder(embedding_model)
    GENERATOR = get_generator(api_key, generation_model)
    print("[backend] Initialized successfully.")

@app.route("/api/query", methods=["POST"])
def query_pipeline():
    pipelines = {
        "rag_fusion": rag_fusion,
        "hyde": hyde,
        "crag": crag,
        "graph_rag": graph_rag
    }
    try:
        data = request.json
        query = data.get("query", "")
        pipeline_name = data.get("strategy", data.get("pipeline", "CRAG")).lower()
        top_k = int(data.get("top_k", 5))

        if not query:
            return jsonify({"error": "No query provided"}), 400

        pipeline = pipelines.get(pipeline_name)
        if not pipeline:
            return jsonify({"error": f"Invalid pipeline name: {pipeline_name}"}), 400

        print(f"Running {pipeline_name} for query: {query}")
        result = pipeline.run(query, CORPUS, EMBEDDER, GENERATOR, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns a few sample queries from the dataset."""
    dataset_path = config["dataset_path"]
    try:
        # Load first 100 and pick 5 random
        examples = list(load_examples(path=dataset_path, limit=100))
        samples = random.sample(examples, min(5, len(examples)))
        return jsonify([{"query": s["query"], "id": s["interaction_id"]} for s in samples])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

init_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
