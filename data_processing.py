import pandas as pd
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Age'] = data['Age'] / 365.25

    data['Status'] = data['Status'].map({'C': 0, 'CL': 1, 'D': 2})
    data['Drug'] = data['Drug'].map({'D-penicillamine': 0, 'placebo': 1})
    data['Sex'] = data['Sex'].map({'M': 0, 'F': 1})
    data['Ascites'] = data['Ascites'].map({'N': 0, 'Y': 1})
    data['Hepatomegaly'] = data['Hepatomegaly'].map({'N': 0, 'Y': 1})
    data['Spiders'] = data['Spiders'].map({'N': 0, 'Y': 1})
    data['Edema'] = data['Edema'].map({'N': 0, 'S': 1, 'Y': 2})

    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()

    data_numeric = data.select_dtypes(include=[float, int])
    correlation_matrix = data_numeric.corr()
    print(correlation_matrix)

    # we remove the weak entites (less important/relevant) columns.
    data = data.drop(columns=['Age', 'Cholesterol', 'Platelets', 'Tryglicerides'])

    return data
