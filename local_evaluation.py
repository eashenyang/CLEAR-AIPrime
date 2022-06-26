"""
=========
IMPORTANT
=========

THE CONTENTS OF THIS FILE WILL BE REPLACED DURING EVALUATION.
ANY CHANGES MADE TO THIS FILE WILL BE DROPPED DURING EVALUATION.

THIS FILE IS PROVIDED ONLY FOR YOUR CONVINIENCE TO TEST THE CODE LOCALLY.

"""

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from evaluation_utils.clear_evaluator import CLEAREvaluator
from evaluation_utils.base_predictor import BaseCLEARPredictor
from evaluation_utils.CLEAR10 import CLEAR10IMG
from torch.utils.data import DataLoader

# Import participant's prediction class
from evaluation_setup import load_models, data_transform

class CLEARPredictor(BaseCLEARPredictor):
    def __init__(self, bucket_num=10, use_gpu=True):
        self.bucket_num = bucket_num
        self.use_gpu = use_gpu
        self.models = [None] * self.bucket_num

    def prediction_setup(self, models_path):
        print("[DEBUG] Loading models...")
        self.models = load_models(models_path)
        assert(len(self.models)) == self.bucket_num

    def prediction(self, image_file_path: str):
        # Data Loader
        print("[DEBUG] Loading and Transforming Test Data...")
        transform = data_transform()
        test_data = [CLEAR10IMG(image_file_path, i, form="all", debug=False, transform=transform) for i in range(self.bucket_num)]
        test_loaders = [DataLoader(test_data[i], shuffle=False, num_workers=4) for i in range(len(test_data))]

        # Inference
        print("[DEBUG] Inference...")
        R = np.zeros((self.bucket_num,)*2)  # accuracy matrix
        for i, model in enumerate(self.models):
            for j, test_loader in enumerate(test_loaders):
                print('Evaluate timestamp %d model on bucket %d' % (i, j))
                if self.use_gpu: 
                    model.to('cuda')
                R[i, j] = self.test(model, test_loader)
            del model

        return R

    def test(self, model, test_loader):
        total_test_acc = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                if self.use_gpu:
                    xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
                y_pred = model(xb)
                _, preds = torch.max(y_pred.data, 1)
                total_test_acc += torch.sum(preds == yb.data)
        avg_test_acc = total_test_acc / len(test_loader.dataset)
        print('Test Accuracy: {:.2f}'.format(avg_test_acc))
        
        return avg_test_acc.cpu().numpy().squeeze()


def main():
    parser = argparse.ArgumentParser(description="Local Evaluation Test")
    parser.add_argument(
        "--dataset-path",
        required=True,
        dest="dataset_path",
        help="Path to dir containing extracted dataset",
    )
    args = parser.parse_args()
    
    evaluator = CLEAREvaluator(
        test_data_path=args.dataset_path,
        models_path="models",
        predictions_file_path="predictions.txt",
        predictor=CLEARPredictor(),
    )
    evaluator.evaluation()  # Make predictions and calculate accuracy matrix

    # Compute four metrics and plot accuracy matrix at accuracy_matrix.png by default
    scores = evaluator.scoring(prediction_file_path="predictions.txt")
    print("Weighted Average Score: %.3f" % scores['score'])
    print("Next-Domain Accuracy: %.3f" % scores['score_secondary'])
    print("In-Domain Accuracy: %.3f" % scores['meta']['in_domain_accuracy'])
    print("Backward Transfer Accuracy: %.3f" % scores['meta']['backward_transfer'])
    print("Forward Transfer Accuracy: %.3f" % scores['meta']['forward_transfer'])


if __name__ == "__main__":
    main()
