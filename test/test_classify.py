from unittest import TestCase

class TestClassify(TestCase):
    def test_get_classified_genres(self):
        from classification_service.classify import get_classified_genres
        pred_genre_df = get_classified_genres()

        known_genres = ['folk', 'soul and reggae', 'punk', 'dance and electronica',
                        'metal', 'pop', 'classic pop and rock', 'jazz and blues']

        for idx, row in pred_genre_df.iterrows():
            self.assertIn(row['pred_genre'], known_genres)



    def test_get_titles_from_genre(self):
        from classification_service.classify import get_titles_from_genre

        genre_to_test = 'pop'  # changing to anything other than 'pop' will cause test to fail
        title_df = get_titles_from_genre(genre_to_test)
        title_list = title_df['title'].tolist()

        # Make some assertions based on some titles that we know are 'pop'
        self.assertIn('Tu Nombre Es Traicion', title_list)
        self.assertIn('I May Smoke Too Much', title_list)
