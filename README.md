## Repository Description for Mercari Price Suggestion Challenge

This repository contains my approach to the Kaggle Mercari Price Suggestion Challenge, where I explored natural language processing (NLP) techniques and implemented a system inspired by GPT's encoder architecture to solve the problem of predicting product prices based on various features.

### Problem Overview:
The challenge involves predicting the price of products listed on Mercari, a Japanese e-commerce platform. The dataset includes a variety of features related to the product, such as:

- **name**: The product name, often brief but descriptive.
- **item_description**: A longer, detailed description of the product.
- **category_name**: The category path, representing a hierarchical categorization (e.g., "Women/Tops & Blouses/Blouses").
- **brand_name**: The brand of the product.
- **item_condition_id**: A numeric representation of the product's condition.
- **price**: The target variable, representing the price to predict.

### Approach:
To tackle this challenge, I designed a model that processes and encodes these features using a custom encoder system inspired by the GPT architecture, especially focusing on the attention mechanism for handling text data.

---

## How to Use:

1. **Download the Dataset:**
   The `train.tsv` file contains the training data needed for this project. Follow these steps to download it:

   - Navigate to the [Mercari Price Suggestion Challenge Data page](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data).
   - Find the `train.tsv` file and download it.
   - Save it in a `data/` directory within your project, as shown below:

     ```
     project_root/
     ├── data/
     │   └── train.tsv
     ├── notebooks/
     │   └── data_analysis.ipynb
     └── src/
         └── main.py
     ```

2. **Install the Dependencies:**
   Install the necessary Python libraries using the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Model:**
   After downloading the dataset and installing dependencies, you can run the `train.py` script to preprocess the data, train the model, and generate predictions:

   ```bash
   python train.py
   ```

4. **Train the Model (Optional):**
   The `train.py` file includes code for training the model using the processed data. You can adjust the parameters and train the model by uncommenting the `model.fit` section in the script.

   The following is the key structure of the code for reference:

   ```python
   model.fit(x_train_dict, y_train, epochs=100, batch_size=128, validation_split=0.2)
   model.save('model.keras')
   ```

---

### Downloading the `train.tsv`:

1. **Access the Dataset:**
   - Navigate to the [Mercari Price Suggestion Challenge Data page](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data).
   
2. **Download the File:**
   - Locate the `train.tsv` file and download it to your local machine.

3. **Save the File:**
   - Store it in a directory named `data` within your project folder to maintain organization.

```bash
project_root/
├── data/
│   └── train.tsv
├── notebooks/
│   └── data_analysis.ipynb
└── src/
    └── main.py
```

You can reference this file in your code by using the following path:

```python
data_path = 'data/train.tsv'
df = pd.read_csv(data_path, sep='\t')
```

This ensures the data is loaded correctly for model training and evaluation.
