import pandas as pd

def voting(predicts):
    nvoters = len(predicts)
    predicts = [item for pred in predicts for item in pred]
    predict_set = set(predicts)
    final_pred = [v for v in predict_set if predicts.count(v) > nvoters / 4]
    final_pred = ' '.join([str(v) for v in final_pred])
    return final_pred

'''
submit_0:
resnet50_bcelog_bestloss_submission.csv
by huutrinh
1fold, 27 epoch, 5tta, fillNa
0.573


submit_1
resnet34_new_weight_bestloss_submission.csv
by dizzyvn
resnet34-5fold-oldExternalPreprocess-RecalcWeightEachEpoch-TTA3
0.571


submit_2:
resnet34_bcelog_bestloss_submission.csv
by huutrinh
5fold-standard loss weight-high val f1-withTTA
0.568

submit_3:
resnet34_new_weight_bestloss_submission.csv
by dizzyvn
2fold, recalc-loss-weight, new-f1, noTTA
0.571


submit_4:
resnet34_bcelog_bestloss_submission.csv
by huutrinh
newf1score, tta, fillNa, 1fold
0.548


submit_5:
resnet50_bcelog_bestloss_submission.csv
by huutrinh
1fold, 27 epoch, 3tta, fillNa
0.569


submit_6:
seresnet50_bcelog_bestloss_submission.csv
by huutrinh
[new_data]1fold, 30 epoch, 5tta, fillNa
0.544

submit_7:
bninception_kfold_recalc_weight_bestloss_submission.csv
by dizzyvn
5fold of bninception (old data?)
0.547

submit_8:
seresnet50_bcelog_bestf1_submission.csv
by huutrinh
add submission details
0.541

submit_9:
seresnet50_bcelog_bestloss_bestf1_submission.csv
by huutrinh
old data, 1 fold
0.562

submit_10:
seresnet50_bcelog_bestloss_submission.csv
by huutrinh
5fold 15 epochs, 5tta, fillna [old data]
0.586

'''

 

# files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_2.csv', './tmp/submit_3.csv', './tmp/submit_4.csv', './tmp/submit_5.csv', \
#         './tmp/submit_6.csv', './tmp/submit_7.csv', './tmp/submit_8.csv'] #0.585 thres=4, 0.576 thres=5

# files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_2.csv', './tmp/submit_3.csv', './tmp/submit_4.csv', './tmp/submit_5.csv', \
#         './tmp/submit_6.csv', './tmp/submit_7.csv', './tmp/submit_9.csv'] #0.581 thres=4, 0.584 thres=5

# files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_2.csv', './tmp/submit_3.csv', './tmp/submit_5.csv', \
#         './tmp/submit_6.csv', './tmp/submit_9.csv'] #0.578 thres=3, 0.582 thres=4

# files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_2.csv', './tmp/submit_3.csv', './tmp/submit_10.csv', './tmp/submit_5.csv', \
#          './tmp/submit_9.csv'] #0.583 thres=4, 0.582 thres=3

# files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_10.csv'] #thres2: 0.578

# files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_2.csv', './tmp/submit_3.csv', './tmp/submit_10.csv', './tmp/submit_5.csv', \
#         './tmp/submit_6.csv', './tmp/submit_7.csv', './tmp/submit_9.csv'] #0.586 thres4, 0.588 thres5

# files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_2.csv', './tmp/submit_3.csv', './tmp/submit_10.csv', './tmp/submit_5.csv', \
#         './tmp/submit_10.csv', './tmp/submit_7.csv', './tmp/submit_9.csv'] #0.589 thres4, 0.584 thres 5

files = ['./tmp/submit_0.csv', './tmp/submit_1.csv', './tmp/submit_2.csv', './tmp/submit_3.csv', './tmp/submit_10.csv', './tmp/submit_5.csv', \
        './tmp/submit_10.csv', './tmp/submit_10.csv', './tmp/submit_9.csv'] #0.589 thres4
submit_dfs = [pd.read_csv(f) for f in files]
final_df = submit_dfs[0] # dummy

all_predicts = [[[int(v) for v in row.split(' ')]
                 for row in df['Predicted']]
                for df in submit_dfs]
nrow = len(final_df)
for row in range(nrow):
    row_predicts = [predicts[row] for predicts in all_predicts]
    final_df['Predicted'][row] = voting(row_predicts)
final_df.to_csv('./tmp/voting_submit.csv', index=False)