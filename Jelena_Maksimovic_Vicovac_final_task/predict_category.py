import joblib
import pandas as pd
 
# Load the saved model
model = joblib.load("model/train_the_best_model.pkl")
 
print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")
 
while True:
    title = input(" Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break
 
    # lenght of title
    product_title_length = len(title)

    # checking if title contain numbers
    has_numbers = False
    for char in title:
        if char.isdigit():
            has_numbers = True
            break

    # checking if title contain special caracters
    has_special_chars = False
    for char in title:
        if not char.isalnum() and not char.isspace():
            has_special_chars = True
            break

    # adding new columns title_content_check
    if has_numbers and has_special_chars:
        title_content_check = "numbers special_chars"
    elif has_numbers:
        title_content_check = "numbers"
    elif has_special_chars:
        title_content_check = "special_chars"
    else:
        title_content_check = ""

    # Max lenght of word in title
    words = title.split()

    if len(words) > 0:
        max_word_length = len(words[0])
        for word in words:
            if len(word) > max_word_length:
                max_word_length = len(word)
    else:
        max_word_length = 0

    # Creating DataFrame
    input_df = pd.DataFrame({
        "product_title": [title],
        "product_title_length": [product_title_length],
        "title_content_check": [title_content_check],
        "max_word_length": [max_word_length]
    })

    # Prediction
    prediction = model.predict(input_df)

    print("Predicted category:", prediction[0])
    print()
 