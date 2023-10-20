import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pdf2image import convert_from_path
import hdbscan


def extract_features(img_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = convert_from_path(img_path)[0]
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()  # Set the model to evaluation mode.
    resnet50_features = torch.nn.Sequential(*list(resnet50.children())[:-1])
    with torch.no_grad():
        features = resnet50_features(img_tensor)
        features = features.squeeze(-1).squeeze(-1)
        flattened_features = features.flatten()
        return flattened_features

def cluster(features):

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(features)
    return cluster_labels



if __name__ == "__main__":
    img_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/pdf/"
    image_files = [f for f in os.listdir(img_path) if f.endswith('.pdf')][:100]
    feature_list = [extract_features(os.path.join(img_path, img_file)) for img_file in image_files]
    cluster(feature_list)
