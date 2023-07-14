# Verdictify Decision Helper

This code provides a web application called "Verdictify" that helps users make purchasing decisions for products. It uses YouTube video transcripts and user-defined preferences to generate a verdict and provide feedback on whether to buy a product. 

## Key Features
- Scrapes product data from an Amazon product URL or user-defined product name.
- Allows users to select preferences for budget-friendliness, customer satisfaction, brand reputation, and innovation.
- Retrieves YouTube video IDs related to the product and displays video details with checkboxes.
- Generates transcripts for selected YouTube videos and combines them with product reviews.
- Uses AI models for natural language processing and conversation to extract relevant information and provide a verdict.
- Calculates a verdict based on user preferences and displays a summary of the product, positive/negative reviews, key features, and the final verdict.
- Displays the final verdict in a user-friendly interface.

## Instructions
1. Install dependencies from requirements.txt
2. Create a file config/apikey.py to add the following API keys:
    openaiapikey = ""
    googleapikey = ""
3. Run the app using streamlit run app-a.py
4. Enter an Amazon product URL or product name.
5. Set your preferences for budget-friendliness, customer satisfaction, brand reputation, and innovation.
6. Select YouTube videos related to the product.
7. Click the "Verdictify" button to generate transcripts and combine them with product reviews.
8. View the final verdict and analysis provided by the application.

Note: Make sure to provide the necessary API keys and dependencies as mentioned in the code above.

Enjoy using Verdictify to make informed purchasing decisions!
