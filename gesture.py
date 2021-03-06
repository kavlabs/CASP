from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import cv2
import time
import numpy as np
import picamera
from picamera.array import PiRGBArray
import socket
from PIL import Image
import tflite_runtime
from tflite_runtime.interpreter import Interpreter

d = {"g1":"hi","g2":"","g3":"help me!","g4":"good morning","g5":"Thank You","g6":"awesome!","g7":"NO!","g8":"YES!","g9":"ok","g10":"I need medical attention immediately!"}
# d1 = {"g3":"help me!","g4":"good morning","g6":"awesome!","g7":"NO!","g8":"YES!"}
d1 = {"0":"help me!","1":"good morning","2":"NO!","3":"awesome!","4":"OK!","5":"YES!","6":""}
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def main():
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
    args = parser.parse_args()

    labels = load_labels(args.labels)

    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host = socket.gethostbyname(socket.gethostname())
    print(host)
    port = 9081
    s.bind((host, port))

    print('Starting server on', host, port)
    print('The Web server URL for this would be http://%s:%d/' % (host, port))

    s.listen(5)

    c, (client_host, client_port) = s.accept()
    print('Got connection from', client_host, client_port)
    backSub = cv2.createBackgroundSubtractorKNN()
#     backSub = cv2.createBackgroundSubtractorMOG2()
    top,right,bottom,left = 20, 150, 400, 450
    count = 0

    with picamera.PiCamera(resolution=(640, 480)) as camera:
#     with picamera.PiCamera(framerate=30) as camera:
        camera.start_preview()
#         rawcap = PiRGBArray(camera, size=(640,480))
        try:
          stream = io.BytesIO()
          for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            stream.seek(0)
#             image1 = frame.array
#             cv2.imshow("kk",image1)
#             key = cv2.waitKey(1) & 0xFF
#             rawCapture.truncate(0)
#             if key == ord("q"):
#                 break
            image1 = np.array(Image.open(stream).convert('RGB'))
#             cv2.imwrite(r"/home/pi/Desktop/casp/img%d.jpg"%count,image1)
#             image1 = cv2.imread(image2)
            roi = image1[top:bottom, right:left]
#             cv2.imwrite(r"/home/pi/Desktop/casp/roi%d.jpg"%count,roi)
            count+=1
            image2 = backSub.apply(cv2.flip(roi,1))
#             image2 = cv2.resize(image2,(30,30))
#             cv2.imshow("kk",image2)
#             image3 = backSub.apply(roi)
            image2 = cv2.resize(image2,(width,height))
            image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(r"/home/pi/Desktop/casp/img%d.jpg"%count,image2)
            img = np.expand_dims(image2, axis=0)
            img = (np.float32(img)-127.5)/127.5
#             cv2.imshow("kk",image)
            start_time = time.time()
            results = classify_image(interpreter, img)
            elapsed_ms = (time.time() - start_time) * 1000
            label_id, prob = results[0]
            print(label_id)
            stream.seek(0)
            stream.truncate()
            try:
                kk = d1[str(label_id)]
            except:
                kk = ''
            c.sendall(str.encode("HTTP/1.0 200 OK\n",'iso-8859-1'))
            c.sendall(str.encode('Content-Type: text/html\n', 'iso-8859-1'))
            c.send(str.encode('\r\n'))
            str1 = b"<html><body style='background-color:black;'><style>.center {margin: 0;position: absolute;top: 50%;left: 50%;-ms-transform: translate(-50%, -50%);transform: translate(-50%, -50%);}</style><font color='red' size='+7'><div class='center'><h1><center>"
            str2 = b"</center></h1></div></body></font></html>"
            str3 = str.encode(kk)
            c.send(str1+str3+str2)
            time.sleep(0.1)
            c.send(b'<script type="text/javascript">document.body.innerHTML = "";</script>')
        finally:
            camera.stop_preview()
    c.close()

if __name__ == '__main__':
    main()
