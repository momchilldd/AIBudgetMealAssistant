import gradio as gr

def create_interface(recommend_meal):
    #Detailed Interface
    explanation_text = gr.HTML("""
        <p>Explore a world of delicious meals with our AI Meal Assistant. 
        Simply enter your meal preference and budget to get personalized recommendations.</p>
    """)

    meal_preference_input = gr.Textbox(
        label="Meal Preference",
        placeholder="Enter your preference",
        style="width: 400px; height: 50px; font-size: 16px;"
    )

    budget_slider = gr.Slider(
        minimum=1,
        maximum=50,
        default=25,
        label="Budget",
        style="width: 400px; height: 50px; font-size: 16px;"
    )

    recommendation_output = gr.Textbox(
        "Hello, please set a meal preference and a budget first.",
        label="Recommendation",
        style="width: 400px; height: 100px; font-size: 16px;"
    )

    # Create the Gradio Interface
    iface = gr.Interface(
        fn=recommend_meal,
        inputs=[meal_preference_input, budget_slider],
        outputs=recommendation_output,
        live=True,
        title="AI Meal Assistant",
    )

    # Configure interface layout and style
    iface.add_component(explanation_text)
    iface.component("Meal Preference").css("margin-bottom", "20px")
    iface.component("Budget").css("margin-bottom", "20px")
    iface.component("Recommendation").css("margin-top", "20px")

    # Style adjustments for a cleaner look
    iface.style(
        inputs_font_size=18,
        outputs_font_size=18,
        live_label_font_size=18,
        title_font_size=24,
        live_label="Live Recommendation"
    )

    # Launch the Gradio Interface
    iface.launch(share=True)

 
