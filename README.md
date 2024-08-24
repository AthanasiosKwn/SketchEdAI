.# SketchEdAI
SketchEdAI: A computer vision and AI-powered educational game designed for young children.

Inspired by Google's Quick, Draw! project (https://quickdraw.withgoogle.com/), I leveraged a subset of the data (https://github.com/googlecreativelab/quickdraw-dataset) collected from this project, to create my own project, SketchEdAI.

SketchEdAI is a Computer Vision - based educational game powered by Neural Networks, designed to provide a fun and engaging learning experience for young children. The game helps users learn to identify and recognize different objects by challenging them to draw specific categories, accompanied by interesting facts. SketchEdAI guides users through the drawing process, offering hints when needed to ensure they stay on track.

! COMMENT: insert gif video of me drawing and getting recognized !

The project is implemented in Python, utilizing two key modules: MediaPipe and PyTorch.

MediaPipe (https://ai.google.dev/edge/mediapipe/solutions/guide) is employed for efficient hand detection and landmark extraction. The extracted hand landmarks enable the development of tools that recognize specific gestures based on their geometric properties, allowing users to interact with the program through computer vision. For instance, a pinch gesture lets the user draw by moving their hand in front of the webcam, a thumbs-down gesture clears the drawing, and a thumbs-up gesture submits the drawing.

PyTorch (https://pytorch.org/) is used to build a Convolutional Neural Network (CNN) classifier trained on a subset of the Quick, Draw! dataset. From the 345 categories available, 8 were selected ('Alarm Clock', 'Apple', 'Axe', 'Banana', 'Bed', 'Bench', 'Bicycle', and 'Book') to ensure faster training. Google provides this data in various formats, and the 28x28 numpy bitmaps were chosen to align with CNN requirements for feature extraction. The classifier processes user-submitted drawings, predicting the corresponding label. The drawings are stored as a list of strokes and converted into 28x28 numpy bitmaps, following the exact pre-processing method used by Google Engineers(https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262), ensuring consistency and preventing accuracy loss.

The CNN model features a simple architecture with two convolutional layers, two max pooling layers, and two fully connected layers. The Adam optimizer and Cross Entropy Loss function are used to optimize the network's parameters. Despite its simplicity, the model achieved a validation accuracy of 95.43% and a test accuracy of 95.24% after 10 epochs of training (overfitting began after the 7th epoch) on a 3060 RTX, 6GB GPU, taking approximately 9.57 minutes. The dataset was shuffled and split into training, validation, and test sets with a 70:20:10 ratio.

! COMMENT : INSERT plot !
