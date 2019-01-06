import os
import torch

from models.model import*
from config import config


def load_model():
    models = []
    # model1
    path1 = "../working/seresnet50_bcelog_fold_0_model_best_loss.pth.tar"
    model1 = get_net()
    model1.load_state_dict(torch.load(path1)["state_dict"]) 
    model1.cuda()
    
    # model2
    path2 = "../working/seresnet50_bcelog_fold_0_model_best_f1.pth.tar"
    model2 = get_net()
    model2.load_state_dict(torch.load(path2)["state_dict"]) 
    model2.cuda()
    
    # add model
    models.append(model1)
    models.append(model2)
    
    return models


def test(test_loader, model):
    # evaluation model
    model.eval()

    test_pred = []
    for i, (inputs, _) in enumerate(tqdm(test_loader)):
        probs = []
        for input in inputs:
            with torch.no_grad():
                image_var = input.cuda(non_blocking=True)
                y_pred = model(image_var)
                prob = y_pred.sigmoid().cpu().data.numpy()
                probs.append(prob)
        test_pred.append(np.vstack(probs).mean(axis=0))

    return test_pred


def makesubmission(test_pred, test_loader):
    print('Aggregating predictions')
    sample_submission_df = pd.read_csv(config.sample_submission)
    submissions = []

    for i, (_, _) in enumerate(tqdm(test_loader)):
        pred = np.where(test_pred[i, :].ravel() > config.thresold)[0]
        if len(pred) == 0: pred = [np.argmax(test_pred[i, :].ravel())]
        subrow = ' '.join(list([str(i) for i in pred]))
        submissions.append(subrow)

    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/{}_bestloss_submission.csv'.format(config.model_name), index=None)

def predict_overall(models):
    # load test csv file
    test_files = pd.read_csv(config.sample_submission)
    
    # load test dataset
    test_gen = HumanDataset(test_files, config.test_data, augument=False, mode="test", tta=config.n_tta)
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=2)
    
    preds = []
    for model in models:
        # test with public dataset
        preds.append(test(test_loader, model))
        
    # compute test predict overall
    test_pred = np.mean(preds, axis=0)
    
    # make submit file
    makesubmission(test_pred, test_loader)


if __name__ == '__main__':
    # load models
    models = load_model()
    
    # predict & and create submit file
    test_pred = predict_overall(models)
    
    
    