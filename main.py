from fastapi import FastAPI, Request, Form, Body, BackgroundTasks, Depends, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional

from MAF2023.DataSet import RawDataSet, aifData
from MAF2023.Metric import DataMetric, ClassificationMetric
from MAF2023.Algorithms.Preprocessing import Disparate_Impact_Remover, Learning_Fair_Representation, RW
from MAF2023.Algorithms.Inprocessing import Gerry_Fair_Classifier, Meta_Fair_Classifier, Prejudice_Remover
from MAF2023.Algorithms.Postprocessing import Calibrated_EqOdds, EqualizedOdds, RejectOption
from MAF2023.Algorithms.sota import FairBatch, FairFeatureDistillation, FairnessVAE, KernelDensityEstimator, \
    LearningFromFailure

from sklearn import svm
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import torch
import random
from torch import nn
from torch import optim

from sample import AdultDataset, GermanDataset, CompasDataset, PubFigDataset

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
metrics = None
miti_result = None


# Main page
@app.get("/")
async def main(request: Request):
    context = {
        'request': request
    }
    return templates.TemplateResponse('index.html', context)


# Data selection
@app.get("/data")
async def data_selection(request: Request):
    global metrics
    global miti_result
    metrics = Metrics()
    miti_result = Mitigation()

    context = {
        'request': request
    }
    return templates.TemplateResponse('data_select.html', context)


@app.post("/file", response_class=RedirectResponse)
async def upload_file(file: UploadFile):
    df = pd.read_csv(file.file)
    df.to_csv("custom.csv", index=False)
    return "/original"


class Metrics:
    def __init__(self):
        self.result = None


    def get_metrics(self, dataset, tsne):
        print("train model start")
        # 2. Get classification metrics
        privilege = {key: value[0] for key, value in
                     zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)}
        print("Privileged values: ", privilege)
        unprivilege = {key: value[0] for key, value in
                       zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)}
        print("Unprivileged values: ", unprivilege)

        print("T-SNE option value:", tsne)
        print("T-SNE option type:", type(tsne))

        # For T-SNE
        if tsne == "on":
            print("T-SNE analysis start")
            priv_val = dataset.privileged_protected_attributes[0][0]
            unpriv_val = dataset.unprivileged_protected_attributes[0][0]

            df = dataset.convert_to_dataframe()[0]
            df_priv = df.loc[df[dataset.protected_attribute_names[0]] == priv_val]
            df_unpriv = df.loc[df[dataset.protected_attribute_names[0]] == unpriv_val]
            ds_priv = aifData(df=df_priv, label_name=dataset.label_names[0],
                              favorable_classes=[dataset.favorable_label],
                              protected_attribute_names=dataset.protected_attribute_names,
                              privileged_classes=dataset.privileged_protected_attributes)
            ds_unpriv = aifData(df=df_unpriv, label_name=dataset.label_names[0],
                                favorable_classes=[dataset.favorable_label],
                                protected_attribute_names=dataset.protected_attribute_names,
                                privileged_classes=dataset.privileged_protected_attributes)

            # Sampling
            sample_size = 50

            priv_sample = random.sample(ds_priv.features.tolist(), k=sample_size)
            priv_sample = np.array(priv_sample)
            unpriv_sample = random.sample(ds_unpriv.features.tolist(), k=sample_size)
            unpriv_sample = np.array(unpriv_sample)

            # T-SNE analysis
            tsne_priv = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(
                priv_sample)
            tsne_unpriv = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(
                unpriv_sample)
            tsne_priv = tsne_priv.tolist()
            tsne_unpriv = tsne_unpriv.tolist()
            print("T-SNE analysis end")
        else:
            tsne_priv = [[1, 1]]
            tsne_unpriv = [[0, 0]]

        ## train model
        model = svm.SVC(random_state=777)
        model.fit(dataset.features, dataset.labels.ravel())

        ## predict
        pred = model.predict(dataset.features)

        ## metric
        data_metric = DataMetric(dataset, privilege=privilege, unprivilege=unprivilege)
        cls_metric = ClassificationMetric(dataset=dataset,
                                          privilege=privilege, unprivilege=unprivilege,
                                          prediction_vector=pred, target_label_name=dataset.label_names[0])
        perfm = cls_metric.performance_measures()

        print("train model end")

        # 3. Make result json
        context = {
            "data": {
                "protected": dataset.protected_attribute_names[0],
                "privileged": {
                    "num_negatives": data_metric.num_negative(privileged=True),
                    "num_positives": data_metric.num_positive(privileged=True),
                    "TSNE": tsne_priv
                },
                "unprivileged": {
                    "num_negatives": data_metric.num_negative(privileged=False),
                    "num_positives": data_metric.num_positive(privileged=False),
                    "TSNE": tsne_unpriv
                },
                "base_rate": round(data_metric.base_rate(), 3),
                "statistical_parity_difference": round(data_metric.statistical_parity_difference(), 3),
                "consistency": round(data_metric.consistency(), 3)
            },
            "performance": {
                "recall": round(perfm['TPR'], 3),
                "true_negative_rate": round(perfm['TNR'], 3),
                "false_positive_rate": round(perfm['FPR'], 3),
                "false_negative_rate": round(perfm['FNR'], 3),
                "precision": round(perfm['PPV'], 3),
                "negative_predictive_value": round(perfm['NPV'], 3),
                "false_discovery_rate": round(perfm['FDR'], 3),
                "false_omission_rate": round(perfm['FOR'], 3),
                "accuracy": round(perfm['ACC'], 3),
            },
            "classify": {
                "error_rate": round(cls_metric.error_rate(), 3),
                "average_odds_difference": round(cls_metric.average_odds_difference(), 3),
                "average_abs_odds_difference": round(cls_metric.average_abs_odds_difference(), 3),
                "selection_rate": round(cls_metric.selection_rate(), 3),
                "disparate_impact": round(cls_metric.disparate_impact(), 3),
                "statistical_parity_difference": round(cls_metric.statistical_parity_difference(), 3),
                "generalized_entropy_index": round(cls_metric.generalized_entropy_index(), 3),
                "theil_index": round(cls_metric.theil_index(), 3),
                "equal_opportunity_difference": round(cls_metric.equal_opportunity_difference(), 3)
            }
        }

        self.result = context

    def get_state(self):
        return self.result


metrics = Metrics()


class Mitigation:
    def __init__(self):
        self.result = None
        self.error_message = None

    def get_metrics(self, dataset, method_id):
        print("Mitigation start")
        try:
            if method_id == 1:  # Disparate impact remover
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                fair_mod = Disparate_Impact_Remover(rep_level=0.5,
                                                    sensitive_attribute=dataset_train.protected_attribute_names[0])
                transf_dataset = fair_mod.fit_transform(dataset_train)

                # Train
                model = svm.SVC(random_state=777)
                model.fit(transf_dataset.features, transf_dataset.labels.ravel())

                # Prediction
                pred = model.predict(dataset_test.features)

            elif method_id == 2:  # Learning fair representation
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                fair_mod = Learning_Fair_Representation(unprivileged_groups=[unprivilege[0]],
                                                        privileged_groups=[privilege[0]])
                transf_dataset = fair_mod.fit_transform(dataset_train)
                transf_dataset.labels = dataset_train.labels

                # Train
                model = svm.SVC(random_state=777)
                model.fit(transf_dataset.features, transf_dataset.labels.ravel())

                # Prediction
                pred = model.predict(dataset_test.features)

            elif method_id == 3:  # Reweighing
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                fair_mod = RW(unprivileged_groups=unprivilege, privileged_groups=privilege)
                transf_dataset = fair_mod.fit_transform(dataset_train)
                transf_dataset.labels = dataset_train.labels

                # Train
                model = svm.SVC(random_state=777)
                model.fit(transf_dataset.features, transf_dataset.labels.ravel())

                # Prediction
                pred = model.predict(dataset_test.features)

            # elif method_id == 4:  # Adversarial debiasing
            # pass
            elif method_id == 5:  # Gerry fair classifier
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                gfc = Gerry_Fair_Classifier()
                gfc.fit(dataset_train)

                # Train
                transf_dataset = gfc.predict(dataset_test)

                # Prediction
                pred = transf_dataset.labels

            elif method_id == 6:  # Meta fair classifier
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                mfc = Meta_Fair_Classifier()
                mfc = mfc.fit(dataset_train)

                # Train
                transf_dataset = mfc.predict(dataset_test)

                # Prediction
                pred = transf_dataset.labels

            elif method_id == 7:  # Prejudice remover
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                pr = Prejudice_Remover()
                pr.fit(dataset_train)

                # Train
                transf_dataset = pr.predict(dataset_test)

                # Prediction
                pred = transf_dataset.labels

            elif method_id == 8:  # Fair batch
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                protected_label = dataset_train.protected_attribute_names[0]
                protected_idx = dataset_train.feature_names.index(protected_label)
                biased = dataset_train.features[:, protected_idx]

                # RawDataSet
                train_data = RawDataSet(x=dataset_train.features, y=dataset_train.labels, z=biased)

                protected_idx_test = dataset_test.feature_names.index(protected_label)
                biased_test = dataset_test.features[:, protected_idx_test]
                test_data = RawDataSet(x=dataset_test.features, y=dataset_test.labels, z=biased_test)

                # Prediction
                batch_size = 256
                alpha = 0.1
                fairness = 'eqodds'
                model, cls2val, _ = FairBatch.train(train_data, batch_size, alpha, fairness)
                pred = FairBatch.evaluation(model, test_data, cls2val)

                # Transformed dataset
                # transf_dataset = dataset_test.copy(deepcopy=True)
                # transf_dataset.labels = np.array(pred).reshape(len(pred), -1)

            elif method_id == 9:
                # Fair feature distillation (Image only)
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names,
                                                                   dataset['aif_dataset'].privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names,
                                                                     dataset[
                                                                         'aif_dataset'].unprivileged_protected_attributes)]

                # Flatten the images
                fltn_img = np.array([img.ravel() for img in dataset['image_list']], dtype='int')

                # Split the dataset
                dataset_train, dataset_test = dataset['aif_dataset'].split([0.7], shuffle=True)

                protected_label = dataset_train.protected_attribute_names[0]
                protected_idx = dataset_train.feature_names.index(protected_label)
                biased = dataset_train.features[:, protected_idx]

                # RawDataSet
                rds = RawDataSet(x=fltn_img, y=dataset['target'], z=dataset['bias'])
                # train_data = RawDataSet(x=train_img, y=train_target, z=train_bias)
                # test_data = RawDataSet(x=test_img, y=test_target, z=test_bias)

                # Train
                n_epoch = 20
                batch_size = 64
                learning_rate = 0.01
                image_shape = (3, 64, 64)
                ffd = FairFeatureDistillation.FFD(rds, n_epoch, batch_size, learning_rate, image_shape)
                ffd.train_teacher()
                ffd.train_student()

                # Prediction
                pred = ffd.evaluation()

                # Make aifData for test
                test_X = ffd.test_dataset.X.reshape(len(ffd.test_dataset), -1).cpu().detach().numpy()
                test_y = ffd.test_dataset.y.cpu().detach().numpy()
                test_z = ffd.test_dataset.z.cpu().detach().numpy()
                df = pd.DataFrame(test_X)
                df[protected_label] = test_z
                df[dataset['aif_dataset'].label_names[0]] = test_y

                dataset_test = aifData(df=df,
                                       label_name=dataset['aif_dataset'].label_names[0],
                                       favorable_classes=[dataset['aif_dataset'].favorable_label],
                                       protected_attribute_names=dataset['aif_dataset'].protected_attribute_names,
                                       privileged_classes=dataset['aif_dataset'].privileged_protected_attributes)


            elif method_id == 10:  # Fair VAE (Image only)
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names,
                                                                   dataset['aif_dataset'].privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names,
                                                                     dataset[
                                                                         'aif_dataset'].unprivileged_protected_attributes)]

                # Flatten the images
                fltn_img = np.array([img.ravel() for img in dataset['image_list']], dtype='int')

                # Split the dataset
                dataset_train, dataset_test = dataset['aif_dataset'].split([0.7], shuffle=True)

                protected_label = dataset_train.protected_attribute_names[0]
                protected_idx = dataset_train.feature_names.index(protected_label)
                biased = dataset_train.features[:, protected_idx]

                # RawDataSet
                rds = RawDataSet(x=fltn_img, y=dataset['target'], z=dataset['bias'])
                # train_data = RawDataSet(x=dataset_train.features, y=dataset_train.labels.ravel(), z=biased)
                # test_data = RawDataSet(x=dataset_test.features, y=dataset_test.labels.ravel(), z=biased)

                # Train
                z_dim = 20
                batch_size = 32
                num_epochs = 20
                image_shape = (3, 64, 64)
                fvae = FairnessVAE.FairnessVAE(rds, z_dim, batch_size, num_epochs, image_shape=image_shape)
                fvae.train_upstream()
                fvae.train_downstream()

                # Prediction
                pred = fvae.evaluation()

                # Make aifData for test
                test_X = fvae.test_dataset.feature.reshape(len(fvae.test_dataset), -1).cpu().detach().numpy()
                test_y = fvae.test_dataset.target.cpu().detach().numpy()
                test_z = fvae.test_dataset.bias.cpu().detach().numpy()
                df = pd.DataFrame(test_X)
                df[protected_label] = test_z
                df[dataset['aif_dataset'].label_names[0]] = test_y

                dataset_test = aifData(df=df,
                                       label_name=dataset['aif_dataset'].label_names[0],
                                       favorable_classes=[dataset['aif_dataset'].favorable_label],
                                       protected_attribute_names=dataset['aif_dataset'].protected_attribute_names,
                                       privileged_classes=dataset['aif_dataset'].privileged_protected_attributes)


            elif method_id == 11:  # Kernel density_estimation
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                # dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                protected_label = dataset.protected_attribute_names[0]
                protected_idx = dataset.feature_names.index(protected_label)
                biased = dataset.features[:, protected_idx]

                # RawDataSet
                rds = RawDataSet(x=dataset.features, y=dataset.labels.ravel(), z=biased)
                # train_data = RawDataSet(x=dataset_train.features, y=dataset_train.labels.ravel(), z=biased)
                # test_data = RawDataSet(x=dataset_test.features, y=dataset_test.labels.ravel(), z=biased)

                # Train
                fairness_type = 'DP'
                batch_size = 64
                n_epoch = 20
                learning_rate = 0.01
                kde = KernelDensityEstimator.KDEmodel(rds, fairness_type, batch_size, n_epoch, learning_rate)
                kde.train()

                # Prediction
                pred = kde.evaluation(all_data=False)

                # Data Make
                test_X = kde.test_data.X.reshape(len(kde.test_data), -1).cpu().detach().numpy()
                test_y = kde.test_data.y.cpu().detach().numpy()
                test_z = kde.test_data.z.cpu().detach().numpy()
                df = pd.DataFrame(test_X)
                df[protected_label] = test_z
                df[dataset.label_names[0]] = test_y

                dataset_test = aifData(df=df,
                                       label_name=dataset.label_names[0], favorable_classes=[dataset.favorable_label],
                                       protected_attribute_names=[dataset.protected_attribute_names[0]],
                                       privileged_classes=dataset.privileged_protected_attributes)



            elif method_id == 12:  # Learning from failure (Image only)
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names,
                                                                   dataset['aif_dataset'].privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names,
                                                                     dataset[
                                                                         'aif_dataset'].unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset['aif_dataset'].split([0.7], shuffle=True)

                protected_label = dataset_train.protected_attribute_names[0]
                protected_idx = dataset_train.feature_names.index(protected_label)
                biased = dataset_train.features[:, protected_idx]

                # RawDataSet
                train_data = RawDataSet(x=np.delete(dataset_train.features, protected_idx, axis=1), y=dataset_train.labels.ravel(), z=biased)
                test_data = RawDataSet(x=np.delete(dataset_test.features, protected_idx, axis=1), y=dataset_test.labels.ravel(), z=biased)

                # Train
                n_epoch = 20
                batch_size = 64
                learning_rate = 0.01
                image_shape = (3, 64, 64)
                lff = LearningFromFailure.LfFmodel(train_data, n_epoch, batch_size, learning_rate, image_shape)
                lff.train()

                # Prediction
                _, _, _, pred = lff.evaluate(lff.model_d)

            elif method_id == 13:  # Calibrated equalized odds
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                # Train
                model = svm.SVC(random_state=777)
                model.fit(dataset_train.features, dataset_train.labels.ravel())

                # Prediction
                pred = model.predict(dataset_test.features)
                dataset_test_pred = dataset_test.copy(deepcopy=True)
                dataset_test_pred.labels = np.array(pred).reshape(len(pred), -1)

                # Post-processing
                cpp = Calibrated_EqOdds([unprivilege[0]], [privilege[0]])
                cpp = cpp.fit(dataset_test, dataset_test_pred)

                # Re-prediction
                pred_dataset = cpp.predict(dataset_test_pred)
                pred = pred_dataset.scores

            elif method_id == 14:  # Equalized odds
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                # Train
                model = svm.SVC(random_state=777)
                model.fit(dataset_train.features, dataset_train.labels.ravel())

                # Prediction
                pred = model.predict(dataset_test.features)
                dataset_test_pred = dataset_test.copy(deepcopy=True)
                dataset_test_pred.labels = np.array(pred).reshape(len(pred), -1)

                # Post-processing
                eqodds = EqualizedOdds([unprivilege[0]], [privilege[0]])
                eqodds = eqodds.fit(dataset_test, dataset_test_pred)

                # Re-prediction
                pred_dataset = eqodds.predict(dataset_test_pred)
                pred = pred_dataset.scores

            elif method_id == 15:  # Reject option
                # Make privileged group and unprivileged group
                privilege = [{key: value[0]} for key, value in
                             zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
                unprivilege = [{key: value[0]} for key, value in
                               zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

                # Split the dataset
                dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

                # Train
                model = svm.SVC(random_state=777)
                model.fit(dataset_train.features, dataset_train.labels.ravel())

                # Prediction
                predict = model.predict(dataset_test.features)
                dataset_test_pred = dataset_test.copy(deepcopy=True)
                dataset_test_pred.labels = np.array(predict).reshape(len(predict), -1)

                # Post-processing
                ro = RejectOption([unprivilege[0]], [privilege[0]])
                ro = ro.fit(dataset_test, dataset_test_pred)

                # Re-prediction
                pred_dataset = ro.predict(dataset_test_pred)
                pred = pred_dataset.scores

            else:
                print("ERROR!!")

            ## metric
            transf_metric = ClassificationMetric(dataset=dataset_test,
                                                 privilege=privilege, unprivilege=unprivilege,
                                                 prediction_vector=pred, target_label_name=dataset_test.label_names[0])
            perfm = transf_metric.performance_measures()
            print("Mitigation end")

            # 3. Make result
            context = {
                "performance": {
                    "recall": round(perfm['TPR'], 3),
                    "true_negative_rate": round(perfm['TNR'], 3),
                    "false_positive_rate": round(perfm['FPR'], 3),
                    "false_negative_rate": round(perfm['FNR'], 3),
                    "precision": round(perfm['PPV'], 3),
                    "negative_predictive_value": round(perfm['NPV'], 3),
                    "false_discovery_rate": round(perfm['FDR'], 3),
                    "false_omission_rate": round(perfm['FOR'], 3),
                    "accuracy": round(perfm['ACC'], 3)
                },
                "classify": {
                    "error_rate": round(transf_metric.error_rate(), 3),
                    "average_odds_difference": round(transf_metric.average_odds_difference(), 3),
                    "average_abs_odds_difference": round(transf_metric.average_abs_odds_difference(), 3),
                    "selection_rate": round(transf_metric.selection_rate(), 3),
                    "disparate_impact": round(transf_metric.disparate_impact(), 3),
                    "statistical_parity_difference": round(transf_metric.statistical_parity_difference(), 3),
                    "generalized_entropy_index": round(transf_metric.generalized_entropy_index(), 3),
                    "theil_index": round(transf_metric.theil_index(), 3),
                    "equal_opportunity_difference": round(transf_metric.equal_opportunity_difference(), 3)
                }
            }

            self.result = context


        except Exception as e:
            error_message = f"An error occurred during mitigation: {str(e)}"
            self.result = {"error": error_message}

    def get_state(self):
        return self.result


miti_result = Mitigation()


# Original Metrics
# Request: form data (Data id)
# Response: Bias metrics (json)
@app.post("/original", response_class=RedirectResponse)
async def original_metrics(
        request: Request, background_tasks: BackgroundTasks,
        data_name: Optional[str] = Form(None),
        tsne: Optional[str] = Form(None)
):
    global metrics
    global miti_result
    metrics = Metrics()
    miti_result = Mitigation()
    metrics.data_name = data_name
    miti_result.data_name = data_name

    # 1. Get data metrics
    if data_name == 'compas':
        data = CompasDataset()
    elif data_name == 'german':
        data = GermanDataset()
    elif data_name == 'adult':
        data = AdultDataset()
    elif data_name == 'pubfig':
        pubfig = PubFigDataset()
        if not os.path.isdir('./Sample/pubfig'):
            pubfig.download()
            return 'There is no image data on your local. We will download pubfig dataset images from source. Please wait a lot of times. After downloaing the images, you can check images on ./Sample/pubfig directory'
        dataset = pubfig.to_dataset()
        data = dataset['aif_dataset']
    else:  # Custom file: data_name = filename
        data_name = 'custom'
        df = pd.read_csv("custom.csv")
        data = aifData(df=df, label_name='Target', favorable_classes=[1],
                       protected_attribute_names=['Bias'], privileged_classes=[[1]])
        # os.remove("custom.csv")

    background_tasks.add_task(metrics.get_metrics, data, tsne)
    return '/original/{}'.format(data_name)

import asyncio

@app.get("/original/{data_name}")
@app.post("/original/{data_name}")
async def check_metrics(request: Request, data_name: str, background_tasks: BackgroundTasks):
    async def metrics_loading():
        while not metrics.result:
            await asyncio.sleep(1)

    background_tasks.add_task(metrics_loading)

    if not metrics.result:
        # Render a loading HTML page
        return templates.TemplateResponse('metrics_loading.html', {'request': request})
    else:
        context = {
            "request": request,
            "data_name": data_name,
            "data": {
                "protected": metrics.result['data']['protected'],
                "privileged": {
                    "num_negatives": metrics.result['data']['privileged']['num_negatives'],
                    "num_positives": metrics.result['data']['privileged']['num_positives'],
                    "TSNE": metrics.result['data']['privileged']['TSNE']
                },
                "unprivileged": {
                    "num_negatives": metrics.result['data']['unprivileged']['num_negatives'],
                    "num_positives": metrics.result['data']['unprivileged']['num_positives'],
                    "TSNE": metrics.result['data']['unprivileged']['TSNE']
                },
                "base_rate": metrics.result['data']['base_rate'],
                "statistical_parity_difference": metrics.result['data']['statistical_parity_difference'],
                "consistency": metrics.result['data']['consistency']
            },
            "performance": {
                "recall": metrics.result['performance']['recall'],
                "true_negative_rate": metrics.result['performance']['true_negative_rate'],
                "false_positive_rate": metrics.result['performance']['false_positive_rate'],
                "false_negative_rate": metrics.result['performance']['false_negative_rate'],
                "precision": metrics.result['performance']['precision'],
                "negative_predictive_value": metrics.result['performance']['negative_predictive_value'],
                "false_discovery_rate": metrics.result['performance']['false_discovery_rate'],
                "false_omission_rate": metrics.result['performance']['false_omission_rate'],
                "accuracy": metrics.result['performance']['accuracy'],
            },
            "classify": {
                "error_rate": metrics.result['classify']['error_rate'],
                "average_odds_difference": metrics.result['classify']['average_odds_difference'],
                "average_abs_odds_difference": metrics.result['classify']['average_abs_odds_difference'],
                "selection_rate": metrics.result['classify']['selection_rate'],
                "disparate_impact": metrics.result['classify']['disparate_impact'],
                "statistical_parity_difference": metrics.result['classify']['statistical_parity_difference'],
                "generalized_entropy_index": metrics.result['classify']['generalized_entropy_index'],
                "theil_index": metrics.result['classify']['theil_index'],
                "equal_opportunity_difference": metrics.result['classify']['equal_opportunity_difference']
            }
        }
        return templates.TemplateResponse('metrics.html', context=context)

# Check metrics status
@app.get("/check_metrics_status", response_model=dict)
async def check_metrics_status():
    # Assume data_name is a global variable that is updated in original_metrics
    data_name = getattr(metrics, 'data_name', None)
    return {"metricsReady": bool(metrics.result), "data_name": data_name}


# Select a algorithm
@app.get("/algorithm/{data_name}")
async def select_algorithm(request: Request, data_name: str):
    context = {
        "request": request,
        "data_name": data_name
    }
    return templates.TemplateResponse("algorithm_select.html", context=context)


# Mitigation Result
# Request: form data (Algorithm id, Data id)
# Response: Comparing metrics (json)
@app.post("/mitigation/{data_name}", response_class=RedirectResponse)
async def compare_metrics(request: Request, background_tasks: BackgroundTasks, data_name: str, algorithm: int = Form(...)):
    # 1. Load original metrics (with task_id)

    # 2. Get mitigated result
    if data_name == 'compas':
        data = CompasDataset()
    elif data_name == 'german':
        data = GermanDataset()
    elif data_name == 'adult':
        data = AdultDataset()
    elif data_name == 'pubfig':
        pubfig = PubFigDataset()
        data = pubfig.to_dataset()
    else:  # Custom file: data_name = filename
        df = pd.read_csv("custom.csv")
        data = aifData(df=df, label_name='Target', favorable_classes=[1],
                       protected_attribute_names=['Bias'], privileged_classes=[[1]])
        os.remove("custom.csv")

    # 3. Make result json
    background_tasks.add_task(miti_result.get_metrics, dataset=data, method_id=algorithm)
    miti_result.method_id = algorithm

    return f"/mitigation/{data_name}/{algorithm}"


# Check mitigation status
@app.get("/check_mitigation_status", response_model=dict)
async def check_mitigation_status():
    # Assume data_name is a global variable that is updated in original_metrics
    data_name = getattr(miti_result, 'data_name', None)
    method_id = getattr(miti_result, 'method_id', None)
    return {"metricsReady": bool(miti_result.result), "data_name": data_name, "method_id": method_id}



@app.post("/mitigation/{data_name}/{algo_id}")
@app.get("/mitigation/{data_name}/{algo_id}")
async def get_mitigated_result(request: Request, data_name: str, algo_id: int, background_tasks: BackgroundTasks):
    async def metrics_loading():
        while not miti_result.result:
            await asyncio.sleep(1)

    background_tasks.add_task(metrics_loading)

    if miti_result.result is None:
        return templates.TemplateResponse('compare_loading.html', {'request': request})
    elif 'error' in miti_result.result:
        context = {'request': request, 'error_message': miti_result.result['error']}
        return templates.TemplateResponse('compare_error.html', context=context)
    else:
        context = {
            'request': request,
            'data_name': data_name,
            'algo_id': algo_id,
            'original': metrics.result,
            'mitigated': miti_result.result
        }
        return templates.TemplateResponse('compare.html', context=context)