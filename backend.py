from transformers import BertTokenizer, BertModel
import torch
import faiss
import random

def load_language_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

def create_embedding(meals, tokenizer, model):
    embedding_dim = model.config.hidden_size
    meal_embeddings = torch.zeros(len(meals), embedding_dim)

    for i, meal in enumerate(meals):
        inputs = tokenizer(meal, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs, return_dict=True)

        last_hidden_states = outputs.last_hidden_state
        pooled_output = last_hidden_states.mean(dim=1)
        meal_embeddings[i] = pooled_output

    return meal_embeddings

def build_faiss_index(meal_embeddings):
    index = faiss.IndexFlatL2(meal_embeddings.shape[1])
    index.add(meal_embeddings.detach().numpy())
    return index

def recommend_meal(preference, budget, tokenizer, model, index, meals, prices):
    inputs = tokenizer(preference, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs, return_dict=True)
    last_hidden_states = outputs.last_hidden_state
    pooled_output = last_hidden_states.mean(dim=1)

    preference_embedding = torch.zeros(1, build_faiss_index.meal_embeddings.shape[1])
    preference_embedding[0] = pooled_output
    _, similar_meals = index.search(preference_embedding.detach().numpy(), 5)

    meals_within_budget = [meals[i] for i in similar_meals[0] if prices[i] <= budget]
    if not meals_within_budget:
        return "Sorry, couldn't find meals within your budget."

    recommended_meal = random.choice(meals_within_budget)
    recommended_price = prices[meals.index(recommended_meal)]

    return f"Recommended Meal: {recommended_meal}, Price: {recommended_price}"
