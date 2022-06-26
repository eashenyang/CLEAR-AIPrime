from statistics import mode
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import timm


def load_models(models_path):
    model_files = os.listdir(models_path)
    model_files = [os.path.join(models_path, ff) for ff in model_files if (ff.endswith('.pth') or ff.endswith('.pt'))]
    model_files.sort()
    assert len(model_files) == 10

    loaded_models = [None] * 10
    for i in range(10):
        loaded_models[i] = timm.create_model('ecaresnet50t', pretrained=False, num_classes=100)
        dictt = torch.load(model_files[i])
        new_dict={}
        for key in dictt:
            if 'module' in key:
                new_dict[key[7:]]=dictt[key]
            else:
                new_dict[key] = dictt[key]
        dictt = new_dict
        loaded_models[i].load_state_dict(dictt)

    # loaded_models = [None] * 10
    # for i in range(10):
    #     loaded_models[i] = models.resnet18(False)
    #     loaded_models[i].load_state_dict(torch.load(model_files[i]))
    
    return loaded_models
    

def data_transform():
    # Data Loader
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(320), 
        transforms.ToTensor(),            
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transform
