import random
import socket
import time
import cv2

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# host = socket.getfqdn()
# host = '127.0.0.1'
host = socket.gethostbyname(socket.gethostname())
port = 9082
s.bind((host, port))

print('Starting server on', host, port)
print('The Web server URL for this would be http://%s:%d/' % (host, port))

s.listen(5)

c, (client_host, client_port) = s.accept()
print('Got connection from', client_host, client_port)
# c.recv(1000)
while True:
    model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph1.pb',
                                          'models/kk.pbtxt')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    global kk
    ret, image = cap.read()
    image_height, image_width, _ = image.shape
    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()

    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .5:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            # kk = str(str(class_id) + " " + str(detection[2])  + " " + class_name)
            kk = str(str(detection[2]*100) + "%"  + " " + class_name)
            print(kk)
            c.sendall(str.encode("HTTP/1.0 200 OK\n",'iso-8859-1'))
            c.sendall(str.encode('Content-Type: text/html\n', 'iso-8859-1'))
            c.send(str.encode('\r\n'))
            c.send(b"<html><body style='background-color:black;'><font color='red'><h1>"+str.encode(f'{kk}')+b"</h1></body></font></html>")
            time.sleep(0.5)
            c.send(b'<script type="text/javascript">document.body.innerHTML = "";</script>')
    cap.release()
    out.release()
    cv2.destroyAllWindows()

c.close()