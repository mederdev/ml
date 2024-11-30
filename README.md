# Machine Learning in Practice

This application performs text analysis and predicts labels for the input text. The main steps include:

1. **Importing Libraries**: The script imports necessary libraries such as `pandas`, `re`, `nltk`, `streamlit`, `matplotlib`, and `sklearn`.
2. **Loading Data**: Reads data from the `fba_inventory.csv` file.
3. **Text Cleaning**: Cleans the text data in the `product_name` column by removing URLs, non-alphabetic characters, and converting text to lowercase.
4. **Tokenization**: Tokenizes the cleaned text.
5. **Stop Words Removal**: Removes Russian stop words from the tokenized text.
6. **Train-Test Split**: Splits the data into training and testing sets.
7. **TF-IDF Vectorization**: Converts text data into TF-IDF features.
8. **Model Training**: Trains a Multinomial Naive Bayes model on the training data.
9. **Prediction**: Predicts labels for the test data.
10. **Streamlit Interface**: Provides a web interface for text input and analysis.
11. **Data Visualization**: Displays the distribution of labels in the training data.

## Installation

1. Clone the repository:
  ```sh
  git clone <repository_url>
  cd <repository_directory>
  ```

2. Install the required packages:
  ```sh
  pip install -r requirements.txt
  ```

3. Download NLTK data:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

## Usage

1. Run the Streamlit application:
  ```sh
  streamlit run analyzing.py
  ```

2. Open the provided URL in your web browser.

3. Enter the text you want to analyze in the text area and click the "Analyze" button to get the prediction.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
