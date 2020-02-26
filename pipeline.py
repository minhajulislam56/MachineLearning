from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

iris_df = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_df.data, iris_df.target, test_size=0.45, random_state=0)

# print(len(x_train))
# print(len(x_test))

pipe_lr = Pipeline([
    ('scaler1', StandardScaler()),
    ('pca1', PCA(n_components=2)),
    ('lr_classifier', LogisticRegression(random_state=0))
])
pipe_dt = Pipeline([
    ('scaler2', StandardScaler()),
    ('pca2', PCA(n_components=2)),
    ('dt_classifier', DecisionTreeClassifier())
])
pipe_rf = Pipeline([
    ('scaler3', StandardScaler()),
    ('pca3', PCA(n_components=2)),
    ('rf_classifier', RandomForestClassifier())
])
pipelines = [pipe_lr, pipe_dt, pipe_rf]

pipe_dc = {
    0: "logistic regression",
    1: "decision tree",
    2: "random forest"
}
for x in pipelines:
    x.fit(x_train, y_train)

for i, model in enumerate(pipelines):
    print("{} test accuracy {}".format(pipe_dc[i], model.score(x_test, y_test)))

# Finding the best accuracy
mx = -1
for i, model in enumerate(pipelines):
    if model.score(x_test, y_test) > mx:
        mx=model.score(x_test, y_test)
        clf_name = pipe_dc[i]

print("best accuracy is {}".format(clf_name))






