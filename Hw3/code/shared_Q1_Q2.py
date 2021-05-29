# %% Imports
import torch
import torchvision.transforms as transforms

def Preprocess(input_image,device):
    preprocess=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
    # perform pre-processing
    input_tensor=preprocess(input_image)
    input_batch=input_tensor.unsqueeze(0) # create a mini-batch of size 1 as expected by the model
    input_batch=input_batch.to(device)
    return input_batch

def DeepLabFeedForward(input_batch,model):
    with torch.no_grad():
        output=model(input_batch)['out'][0]
    output_predictions=output.argmax(0)
    return output_predictions

def DeepLabSegmentation(input_image,model,device,return_type):
    input_batch = Preprocess(input_image,device)
    output_predictions = DeepLabFeedForward(input_batch,model)
    if return_type:
        output_predictions = output_predictions.cpu().numpy()
    else:
        output_predictions
    return output_predictions











