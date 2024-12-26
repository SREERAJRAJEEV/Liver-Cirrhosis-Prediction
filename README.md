
# Liver Cirrhosis Stage Prediction

This project is a machine learning application that predicts the stage of liver cirrhosis based on user input. The model was built using a dataset with 25,000 entries and is my first machine learning project.

The application allows users to input various health parameters, and the model will predict the stage of liver cirrhosis based on these inputs. This project uses **Streamlit** for the frontend and **scikit-learn** for the machine learning model.

## Project Structure

The project is organized as follows:

```
liver_cirrhosis_prediction/
├── app.py                  # Streamlit application to interact with the model
├── data_processing.py      # Script for preprocessing the input data
├── liver_cirrhosis.csv     # Dataset used to train the model 
├── liver_cirrhosis_model.pkl # Pre-trained machine learning model
├── model_training/         # Folder for model training scripts 
├── requirements.txt        # List of required dependencies
```

## Installation

To run this project locally, follow the steps below:

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/liver_cirrhosis_prediction.git
cd liver_cirrhosis_prediction
```

### 2. Set up a Virtual Environment

It is recommended to use a virtual environment to manage the dependencies for this project.

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows:

```bash
.venv\Scriptsctivate
```

- On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Once the dependencies are installed, you can run the Streamlit application using the following command:

```bash
streamlit run app.py
```

This will start the application, and you can access it in your browser at `http://localhost:8501`.

## How it Works

1. **Model Training**: The model is trained on a dataset containing 25,000 entries, which is used to predict the stage of liver cirrhosis based on various health parameters.
2. **User Input**: The Streamlit app provides an interactive interface for users to input health-related data (e.g., Bilirubin, Albumin, SGOT, etc.).
3. **Prediction**: After the user enters the data, the model predicts the stage of liver cirrhosis, which is displayed on the app.

## Requirements

The following Python packages are required to run the project:

- streamlit
- pandas
- scikit-learn
- joblib

You can install all the dependencies by running:

```bash
pip install -r requirements.txt
```

## Notes

- This is my first machine learning project, and it serves as a demonstration of how to use machine learning models for real-world applications.
- The model was trained using a dataset with 25,000 entries, and the accuracy and predictions are based on this data.
- The application uses **Streamlit** for creating a simple, interactive web interface.

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to contribute, report issues, or provide suggestions for improvement!
