# sacred import
from sacred import Ingredient
path_ingredient = Ingredient('path')

@path_ingredient.config
def cfg():
    root              = '/home/tran/workspace/new_human_protein_atlas_image_classfication/'
    submit            = 'submit/'
    train_data        = 'input/train_all/'
    test_data         = 'input/test_jpg/'
    sample_submission = 'input/sample_submission.csv'
    kaggle_csv        = 'input/train.csv'
    external_csv      = 'input/external_data/train.csv'
