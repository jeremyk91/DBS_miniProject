from joblib import load
import os
import pandas as pd

from database.sqlite_db import conn

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
clf = load(BASE_DIR + '/model_files/KNN_clf.joblib')


def classify_test_dataset():
    """Classify the test dataset """
    test_df = pd.read_csv(BASE_DIR + "/Data/test.csv")
    trackTitle = test_df[['trackID', 'title']]

    y_pred = clf.predict(test_df)
    y_pred = pd.Series(y_pred, name='pred_genre')  # convert from np.array to pd.Series
    trackTitle_y_pred = pd.concat([trackTitle, y_pred], axis=1)  # concat the relevant columns tgt
    trackTitle_y_pred.to_sql(name='Xy_test',
                             con=conn,
                             if_exists='replace', # replace so that the classify_test_dataset() method can run multiple times
                             )


def get_classified_genres():
    """Return the entire classified test dataset"""

    query = "SELECT DISTINCT pred_genre FROM Xy_test"
    pred_genre = pd.read_sql(sql=query, con=conn)
    return pred_genre


def get_titles_from_genre(genre: str) -> pd.DataFrame:
    """Return only the titles from the specified genre"""

    query = f"""
            SELECT title from Xy_test WHERE pred_genre = '{genre}'
            """
    title_genre = pd.read_sql(sql=query, con=conn)
    return title_genre


# if __name__ == '__main__':
#     classify_test_dataset()
#     get_classified_genres()
#     get_titles_from_genre('metal')
