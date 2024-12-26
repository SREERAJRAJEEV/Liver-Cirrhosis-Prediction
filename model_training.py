from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
from data_processing import preprocess_data
def train_model():

    data = preprocess_data('liver_cirrhosis.csv')

    #split to features and target variable
    X = data.drop(columns=['Stage'])
    y = data['Stage']

    #split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #we use RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    #predict!
    y_pred = model.predict(X_test)

    #checking perfomance
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    #hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    #optimizing the model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    print(f"Best Model Accuracy: {accuracy_score(y_test, y_pred_best)}")
    print("Best Model Classification Report:")
    print(classification_report(y_test, y_pred_best))

    #cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average cross-validation score: {cv_scores.mean()}")

    #save
    joblib.dump(best_model, 'liver_cirrhosis_model.pkl')


if __name__ == "__main__":
    train_model()
