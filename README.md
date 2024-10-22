### Repository Description for Mercari Price Suggestion Challenge

This repository contains my approach to the Kaggle Mercari Price Suggestion Challenge, where I explored natural language processing (NLP) techniques and implemented a system inspired by GPT's encoder architecture to solve the problem of predicting product prices based on various features.

#### Problem Overview:
The challenge involves predicting the price of products listed on Mercari, a Japanese e-commerce platform. The dataset includes a variety of features related to the product, such as:
- `name`: The product name, often brief but descriptive.
- `item_description`: A longer, detailed description of the product.
- `category_name`: The category path, representing a hierarchical categorization (e.g., "Women/Tops & Blouses/Blouses").
- `brand_name`: The brand of the product.
- `item_condition_id`: A numeric representation of the product's condition.
- `price`: The target variable, representing the price to predict.

#### Approach:
To tackle this challenge, I designed a model that processes and encodes these features using a custom encoder system inspired by the GPT architecture, especially focusing on the attention mechanism for handling text data.

**Key Steps:**
1. **Preprocessing**: 
   - Preprocessed the textual data, including product names, descriptions, and category names.
   - Tokenized and embedded these texts to transform them into numerical representations that could be fed into the model.
   
2. **Encoder Architecture**:
   - Implemented a multi-head self-attention mechanism to process the textual inputs (product name and description) to capture the important contextual information.
   - Applied a transformer-like encoder architecture to process the hierarchical structure of product categories.

3. **Feature Integration**:
   - Combined outputs from the encoder with additional structured features like `item_condition_id` and `brand_name` to provide a holistic view of each product.

4. **Model Training**:
   - Trained the model using a deep learning approach with fully connected layers, tuned to predict the final product price.
   - Optimized the model using the Adam optimizer and employed techniques like batch normalization and dropout to prevent overfitting.

5. **Challenges & Optimizations**:
   - Implemented custom residual normalization wrappers and layer normalization to ensure smooth gradient flow.
   - Used `AdjustDim` layers to handle dimensionality issues during concatenation of different feature outputs.
   - Dealt with masking issues during the attention mechanism and incorporated solutions to handle various input masks for different features.

#### Results:
The model was trained and tested on a subset of the data due to computational limitations. It effectively learned to predict product prices based on the given features, providing a strong baseline for further improvements and experiments.

#### Future Work:
- Further tuning the model with more advanced NLP techniques and improving feature engineering.
- Expanding the model to handle larger datasets more efficiently.
- Exploring additional architectures, such as BERT-based encoders, to improve accuracy.

This repository showcases a practical implementation of transformer-based architectures in a real-world pricing prediction problem, offering insights into how attention mechanisms and NLP techniques can be leveraged for structured data problems.
