import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

df=pd.read_csv ("data/IMLP4_TASK_03-products.csv")

# drop all rows with missing values
df=df.dropna()

# standardization of column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.strip('_')

# Convert all values to lowercase and strip extra spaces
df['category_label'] = df['category_label'].astype(str).str.lower().str.strip()

# standardization Category Label
mapping = {
    'fridge freezers': 'fridge',
    'fridges': 'fridge',
    'freezers': 'fridge',
    'mobile phones':'mobile phone',
    'cpus':'cpu'
}
df['category_label'] = df['category_label'].replace(mapping)


# Convert column category_label type to 'category'
df['category_label'] = df['category_label'].astype('category')

# Removing irrelenavt columns
df=df.drop(columns=['product_id','merchant_id','product_code','number_of_views','merchant_rating','listing_date'])

# Adding new column with lenght of product title
df['product_title_length'] = df['product_title'].str.len()

# Adding new column with info if title content numbers or caracters
df['has_numbers'] = df['product_title'].str.contains(r'\d', regex=True)
df['has_special_chars'] = df['product_title'].str.contains(r'[^a-zA-Z0-9\s]', regex=True)
df['title_content_check'] = (
    df['has_numbers'].map({True: 'numbers', False: ''}) +
    df['has_special_chars'].map({True: ' special_chars', False: ''})
).str.strip()

# Adding column maximum word length per title
df['max_word_length'] = df['product_title'].apply(lambda x: max([len(word) for word in str(x).split()]) if x else 0)

# Features and label
X = df[["product_title", "product_title_length", "title_content_check","max_word_length"]]
y = df["category_label"]

# Preprocessor: TF-IDF for text, MinMaxScaler for numeric feature
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "product_title"),
        ("content", TfidfVectorizer(), "title_content_check"),
        ("length", MinMaxScaler(), ["product_title_length"]),
        ("max_length", MinMaxScaler(), ["max_word_length"])
    ]
)
# define pipeline with the best model
pipeline=Pipeline([
        ("preprocessing", preprocessor),
        ("classifier",RandomForestClassifier())
    ])

# Train the model on the all dataset
pipeline.fit (X,y)

# Save the model to a file
joblib.dump(pipeline,"model/train_the_best_model.pkl")
print ("Model trained and saved")