import torch
import os
import cv2
from yolact import Yolact

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = 'yolact_base_54_800000.pth'
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    net.to(device)

    output_onnx = os.path.splitext(trained_model)[0] + '.onnx'
    inputs = torch.randn(1, 3, 550, 550).to(device)
    print('convert',output_onnx,'begin')
    torch.onnx.export(net, inputs, output_onnx, verbose=False, opset_version=12, input_names=['image'],
                      output_names=['loc', 'conf', 'mask', 'proto'])
    print('convert', output_onnx, 'to onnx finish!!!')

    try:
        dnnnet = cv2.dnn.readNet(output_onnx)
        print('read sucess')
    except:
        print('read failed')
        dnnnet = cv2.dnn.readNet(output_onnx)