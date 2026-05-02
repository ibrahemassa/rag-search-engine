from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import RESULTS_LIMIT, cosine_similarity, load_movies

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", docs: list[dict]=None):
        self.model_name = model_name
        self.sen_transformer = SentenceTransformer(model_name)
        if docs:
            self.docs = docs
            self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]

            self.text_embeddings = self.sen_transformer.encode(self.texts)

    def embed_image(self, img_path):
        img = Image.open(img_path)
        img_embedding = self.sen_transformer.encode(img)
        return img_embedding

    def search_with_image(self, img_path: str):
        img = Image.open(img_path)
        img_embedding = self.embed_image(img_path)
        similarities: list[tuple] = [(i, cosine_similarity(img_embedding, text_embedding)) for i, text_embedding in enumerate(self.text_embeddings)]
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = [{**(self.docs[i]), "similarity": similarity} for i, similarity in similarities[:RESULTS_LIMIT]]


        return results

def verify_image_embedding(img_path):
    mul_search = MultimodalSearch()
    img_embedding = mul_search.embed_image(img_path)
    print(f"Embedding shape: {img_embedding.shape[0]} dimensions")

def search_with_image_command(img_path):
    movies = load_movies()
    mul_search = MultimodalSearch(docs=movies)
    results = mul_search.search_with_image(img_path)
    return results



