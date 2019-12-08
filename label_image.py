from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import socket
import argparse
import numpy as np
import time
from PIL import Image
import tensorflow as tf
import cv2

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

k = ['ok.jpg','ok1.jpg','ok2.jpg','rock.jpg','hi.jpg','g3.jpg','g4.jpg','g5.jpg','g7.jpg']
d = {"g1":"hi","g2":"","g3":"help me!","g4":"good morning","g5":"Thank You","g6":"awesome!","g7":"NO!","g8":"YES!","g9":"ok","g10":"I need medical attention immediately!"}
d1 = {"g3":"help me!","g4":"good morning","g6":"awesome!","g7":"NO!","g8":"YES!"}
if __name__ == '__main__':
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  host = socket.gethostbyname(socket.gethostname())
  port = 9082
  s.bind((host, port))

  print('Starting server on', host, port)
  print('The Web server URL for this would be http://%s:%d/' % (host, port))

  s.listen(5)

  c, (client_host, client_port) = s.accept()
  print('Got connection from', client_host, client_port)
  while True:
    # print(">>>>>>>>>>>>>>",value)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, image = cap.read()
    # print(image)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='ok.jpg',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    args = parser.parse_args()
    m = time.time()
    interpreter = tf.lite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = input_details[0]['dtype'] == np.float32

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    # img = Image.open(args.image).resize((width, height))
    # print(np.shape(img))
    img = cv2.resize(image, (width, height))
    # print(np.shape(img))

    input_data = np.expand_dims(img, axis=0)

    if floating_model:
      input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    try:
      kk = d1[labels[top_k[0]]]
    except:
      kk = ''
    c.sendall(str.encode("HTTP/1.0 200 OK\n",'iso-8859-1'))
    c.sendall(str.encode('Content-Type: text/html\n', 'iso-8859-1'))
    c.send(str.encode('\r\n'))
    c.send(b"<html><body style='background-color:black;'><style>.center {margin: 0;position: absolute;top: 50%;left: 50%;-ms-transform: translate(-50%, -50%);transform: translate(-50%, -50%);}</style><font color='red' size='+7'><div class='center'><h1><center>"+str.encode(f'{kk}')+b"</center></h1></div></body></font></html>")
    time.sleep(0.5)
    c.send(b'<script type="text/javascript">document.body.innerHTML = "";</script>')
    cap.release()
    cv2.destroyAllWindows()
  c.close()