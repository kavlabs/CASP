# CASP
CASP(Communication Assistant for Special People) is a device which uses a neural network trained to translate sign hand gestures to their respective word interpretation.

It is made to run on Raspberry pi.

steps to run the script:
1)go to https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter to install tf lite interpreter on raspberry pi.
2)run gesture.py as: python3 gesture.py --model converted6.tflite --labels output_labels6.txt.

For retraining the model run retrain.py and then convert the .pb file to .tflite file.
