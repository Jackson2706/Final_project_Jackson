import torchvision
import torch.onnx

# Load the pre-trained SSD model
ssd_model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
print(ssd_model.classes)
# # Set the model to evaluation mode
# ssd_model.eval()
#
# # Define sample input dimensions
# input_shape = (1, 3, 300, 300)  # (batch_size, channels, height, width)
#
# # Create a sample input tensor
# sample_input = torch.randn(*input_shape)
# print("SSD model exported to ONNX format !")
# # Export the model to ONNX format
# onnx_file_path = "/home/jackson/Desktop/Final_project_Jackson/aicore/detection/SSD_onnx/SSD300/config/ssd300.onnx"
# torch.onnx.export(ssd_model, sample_input, onnx_file_path, opset_version=11,  input_names=['images'], output_names=['labels', 'scores', 'boxes'],)
#
# print("SSD model exported to ONNX format successfully!")
# print()