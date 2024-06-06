import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
import tkinter as tk
from tkinter import ttk, messagebox
# Загрузка данных
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Предварительная обработка данных
def preprocess_data(df):
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'] = imputer.fit_transform(df[['Fare']])
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return X, y
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    # Построение графиков обучения
    plt.figure()
    plt.title(f'{name} - ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc_score(y_test, y_proba):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()



    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest']),
            ('lgbm', models['LightGBM']),
            ('svc', models['SVM']),
            ('knn', models['KNN'])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    results['Ensemble'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    # Построение графика для ансамбля
    plt.figure()
    plt.title('Ensemble - ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'Ensemble (area = {roc_auc_score(y_test, y_proba):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()

    def on_train_button_click():
        method = method_combobox.get()
        if method not in models and method != 'Ensemble':
            messagebox.showerror("Ошибка", "Пожалуйста, выберите метод обучения")
            return
        if method == 'Ensemble':
            y_pred = ensemble.predict(X_test)
            y_proba = ensemble.predict_proba(X_test)[:, 1]
        else:
            model = models[method]
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        result_label.config(text=f"Accuracy: {acc:.2f}\nROC AUC: {roc_auc:.2f}")

    root = tk.Tk()
    root.title("Модель обучения - Титаник")
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    ttk.Label(frame, text="Выберите метод обучения:").grid(row=0, column=0, sticky=tk.W)
    method_combobox = ttk.Combobox(frame, values=list(models.keys()) + ['Ensemble'])
    method_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E))
    train_button = ttk.Button(frame, text="Обучить", command=on_train_button_click)
    train_button.grid(row=1, column=0, columnspan=2, pady=10)
    result_label = ttk.Label(frame, text="Результаты будут показаны здесь", padding="10")
    result_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
    root.mainloop()
