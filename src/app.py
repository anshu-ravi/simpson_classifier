import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torchvision.transforms.functional import InterpolationMode
import gradio as gr



map_characters = {0: 'abraham_grampa_simpson', 
                1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
                3: 'charles_montgomery_burns', 4: 'chief_wiggum', 
                5: 'comic_book_guy', 6: 'edna_krabappel', 7: 'homer_simpson', 
                8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
                11: 'marge_simpson', 12: 'milhouse_van_houten', 
                13: 'moe_szyslak', 14: 'ned_flanders', 15: 'nelson_muntz', 
                16: 'principal_skinner', 17: 'sideshow_bob'}



def predict_simpson_character(image): 

    # Define transformations
    size_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = size_transform(image)

    # Add an extra batch dimension
    image = image.unsqueeze(0)

    # Load the pretrained ResNet-50 model
    resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Modify the final fully connected layer
    num_classes = 18
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    saved_state_dict = torch.load("model3.pth", map_location=torch.device('cpu'))
    resnet.load_state_dict(saved_state_dict)

    # Generate predictions
    resnet.eval()
    with torch.no_grad():
        output = resnet(image)
        _, predicted = output.max(1)

    prediction = map_characters[int(predicted[0])]
    prediction = prediction.replace('_', " ").title()

    return prediction


example_images = ['examples/pic_0012.jpg', 'examples/pic_0013.jpg', 'examples/pic_0025.jpg', 'examples/pic_0036.jpg']

ifc = gr.Interface(fn=predict_simpson_character,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(), 
             title = "Simpson Character Classifier", 
             description = "This classifier was trained on 18 famous Simpson characters. Upload an image of a Simpson character to get predictions", 
             examples=example_images)

if __name__ == "__main__":
    ifc.launch()
