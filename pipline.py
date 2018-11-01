from common import *
from helper import *
from kenerl import KernelSettings
from sklearn.model_selection import RepeatedKFold
from model_parameter import ModelParameter
from improvement_generator import ImprovedDataGenerator
from track_history import TrackHistory 
"""
Building a baseline model
"""
train_files = listdir("../input/human-protein-atlas-image-classification/train")
test_files = listdir("../input/human-protein-atlas-image-classification/test")
percentage = np.round(len(test_files) / len(train_files) * 100)
print("The test set size turns out to be {} % compared to the train set.".format(percentage))

# K-Fold Cross-Validation
splitter = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
partitions = []
for train_idx, test_idx in splitter.split(train_labels.index.values):
    partition = {}
    partition["train"] = train_labels.Id.values[train_idx]
    partition["validation"] = train_labels.Id.values[test_idx]
    partitions.append(partition)
    print("TRAIN:", train_idx, "TEST:", test_idx)
    print("TRAIN:", len(train_idx), "TEST:", len(test_idx))

print(partitions[0]["train"][0:5])

# parameter setting
parameter = ModelParameter(train_path)

# preprocessing
preprocessor = ImagePreprocessor(parameter)

# looking preprocessing
example = images[0,0]
preprocessed = preprocessor.preprocess(example)
print(example.shape)
print(preprocessed.shape)
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(example, cmap="Greens")
ax[1].imshow(preprocessed.reshape(parameter.scaled_row_dim,parameter.scaled_col_dim), cmap="Greens")

# datasets
partition = partitions[0]
labels = train_labels
print("Number of samples in train: {}".format(len(partition["train"])))
print("Number of samples in validation: {}".format(len(partition["validation"])))

# training the baseline on the first cv-fold
training_generator = DataGenerator(partition['train'], labels, parameter, preprocessor)
validation_generator = DataGenerator(partition['validation'], labels, parameter, preprocessor)
predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)

# run computation and store results as csv
if kernelsettings.fit_baseline == True:
    model = BaseLineModel(parameter)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    history = model.learn()
    #model.save("baseline_model.h5")
    proba_predictions = model.predict(predict_generator)
    baseline_proba_predictions = pd.DataFrame(proba_predictions, columns=train_labels.drop(
        ["Target", "number_of_targets", "Id"], axis=1).columns)
    baseline_proba_predictions.to_csv("baseline_predictions.csv")
# if you already have done a baseline fit once, 
# you can load predictions as csv and further fitting is not neccessary:
else:
    baseline_proba_predictions = pd.read_csv("../input/protein-atlas-eab-predictions/baseline_predictions.csv", index_col=0)


validation_labels = train_labels.loc[train_labels.Id.isin(partition["validation"])]
print(validation_labels.shape)
print(baseline_proba_predictions.shape)
print(baseline_proba_predictions.tail())

proba_predictions = baseline_proba_predictions.values
hot_values = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).values.flatten()
one_hot = (hot_values.sum()) / hot_values.shape[0] * 100
zero_hot = (hot_values.shape[0] - hot_values.sum()) / hot_values.shape[0] * 100

fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.distplot(proba_predictions.flatten() * 100, color="DodgerBlue", ax=ax[0])
ax[0].set_xlabel("Probability in %")
ax[0].set_ylabel("Density")
ax[0].set_title("Predicted probabilities")
sns.barplot(x=["label = 0", "label = 1"], y=[zero_hot, one_hot], ax=ax[1])
ax[1].set_ylim([0,100])
ax[1].set_title("True target label count")
ax[1].set_ylabel("Percentage")

# to which targets do the high and small predicted probabilities belong to?
mean_predictions = np.mean(proba_predictions, axis=0)
std_predictions = np.std(proba_predictions, axis=0)
mean_targets = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).mean()
labels = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).columns.values
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.barplot(x=labels,
            y=mean_predictions,
            ax=ax[0])
ax[0].set_xticklabels(labels=labels,
                      rotation=90)
ax[0].set_ylabel("Mean predicted probability")
ax[0].set_title("Mean predicted probability per class over all samples")
sns.barplot(x=labels,
           y=std_predictions,
           ax=ax[1])
ax[1].set_xticklabels(labels=labels,
                      rotation=90)
ax[1].set_ylabel("Standard deviation")
ax[1].set_title("Standard deviation of predicted probability per class over all samples")

# show result 
fig, ax = plt.subplots(1,1,figsize=(20,5))
sns.barplot(x=labels, y=mean_targets.values, ax=ax)
ax.set_xticklabels(labels=labels,
                      rotation=90)
ax.set_ylabel("Percentage of hot (1)")
ax.set_title("Percentage of hot counts (ones) per target class")

# 
feature = "Cytosol"
plt.figure(figsize=(20,5))
sns.distplot(baseline_proba_predictions[feature].values[0:-10], color="Purple")
plt.xlabel("Predicted probabilites of {}".format(feature))
plt.ylabel("Density")
plt.xlim([0,1])

# improvement
wishlist = ["Nucleoplasm", "Cytosol", "Plasma membrane"]

# adding score metric
"""
https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
"""
def f1(y_true, y_pred):
    y_pred = keras.round(y_pred)
    tp = K.sum(keras.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(keras.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(keras.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(keras.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + keras.epsilon())
    r = tp / (tp + fn + keras.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

# plug and play 
parameter = ModelParameter(train_path, num_classes=len(wishlist), n_epochs=5, batch_size=64)
labels = train_labels

training_generator = ImprovedDataGenerator(partition['train'], labels,
                                           parameter, preprocessor, wishlist)
validation_generator = ImprovedDataGenerator(partition['validation'], labels,
                                             parameter, preprocessor, wishlist)
predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)

# run computation and store results as csv
if kernelsettings.fit_improved_baseline == True:
    model = ImprovedModel(parameter)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    history = model.learn()
    proba_predictions = model.predict(validation_generator)
    #model.save("improved_model.h5")
    improved_proba_predictions = pd.DataFrame(proba_predictions, columns=wishlist)
    improved_proba_predictions.to_csv("improved_predictions.csv")
# if you already have done a baseline fit once, 
# you can load predictions as csv and further fitting is not neccessary:
else:
    improved_proba_predictions = pd.read_csv("../input/protein-atlas-eab-predictions/improved_predictions.csv", index_col=0)

if kernelsettings.fit_improved_baseline == True:
    plt.figure(figsize=(20,5))
    plt.plot(model.history.losses)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.savefig("loss_of_improved_model", format="png")

# show result 
fig, ax = plt.subplots(3,1,figsize=(25,15))
sns.distplot(improved_proba_predictions.values[:,0], color="Orange", ax=ax[0])
ax[0].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[0]))
ax[0].set_xlim([0,1])
sns.distplot(improved_proba_predictions.values[:,1], color="Purple", ax=ax[1])
ax[1].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[1]))
ax[1].set_xlim([0,1])
sns.distplot(improved_proba_predictions.values[:,2], color="Limegreen", ax=ax[2])
ax[2].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[2]))
ax[2].set_xlim([0,1])
