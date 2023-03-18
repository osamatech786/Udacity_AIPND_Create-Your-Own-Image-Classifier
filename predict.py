import argparse
from functions import *
import numpy as np
import json

def predict():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--input_img', default='flowers/test/1/image_06764.jpg')
    parser.add_argument('--checkpoint', default='checkpoint.pth')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', default=False, action='store_true')

    pa_obj = parser.parse_args()
    image_path = pa_obj.input_img
    nums_outs = pa_obj.top_k
    category_names = pa_obj.category_names
    device = "gpu" if pa_obj.gpu else "cpu"
    checkpoint_path = pa_obj.checkpoint

    print("Loading Data")
    training_loader, testing_loader, validation_loader, _ = load_data()
    print("Loading Checkpoint")
    model = load_checkpoint(checkpoint_path, device)
    print("Prediciting")
    probabs = predict(image_path, model, num_outs, device)
    with open(category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)

    labels = [cat_to_name[str(index + 1)] for index in np.array(probabs[1][0].cpu().numpy())]
    probability = np.array(probabs[0][0].cpu().numpy())
    for i in range(0, nums_outs):
        print(f'{labels[i]}, probability: {probability[i]*100}%')

if __name__ == '__main__':
    predict()