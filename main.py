from frontend import create_interface
from backend import load_language_model, create_embedding, build_faiss_index, recommend_meal

# Load language model
tokenizer, model = load_language_model()

# Define meal and price vectors
meals = [ 
    "Spaghetti Bolognese", "Chicken Alfredo", "Vegetarian Pizza", "Salmon Salad",
    "Beef Stir-Fry", "Margherita Pizza", "Caesar Salad", "Shrimp Scampi",
    "Chicken Parmesan", "Vegetable Lasagna", "Grilled Salmon", "Caprese Salad",
    "Penne Vodka", "Chicken Caesar Wrap", "BBQ Ribs", "Mushroom Risotto",
    "Teriyaki Chicken Bowl", "Greek Salad", "Lobster Pasta", "Tofu Stir-Fry",
    "Steak Fajitas", "Cobb Salad", "Pesto Pasta", "Hawaiian Pizza",
    "Spinach and Feta Stuffed Chicken", "Shrimp Pad Thai", "Avocado Toast",
    "Chicken Tikka Masala", "Butternut Squash Risotto", "BBQ Pizza"
    ]
prices = [
    10.0, 15.0, 12.0, 18.0,
    14.0, 11.0, 9.0, 20.0,
    16.0, 13.0, 19.0, 11.0,
    15.0, 10.0, 22.0, 17.0,
    18.0, 14.0, 20.0, 12.0,
    23.0, 14.0, 13.0, 12.0,
    19.0, 18.0, 14.0, 16.0,
    20.0, 15.0
    ]

# Create embeddings and build Faiss index
meal_embeddings = create_embedding(meals, tokenizer, model)
index = build_faiss_index(meal_embeddings)

# Create Gradio interface
recommendation_fn = lambda p, b: recommend_meal(p, b, tokenizer, model, index, meals, prices)
iface = create_interface(recommendation_fn)

# Launch Gradio interface
iface.launch(share=True)
