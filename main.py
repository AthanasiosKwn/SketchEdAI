import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import cairocffi as cairo
import torch
import torch.nn as nn
from time import time
import random



# Dictionary containing the classes of drawings to be drawn by the user and relative information abou them (Fun-facts, Hints).
classes_of_drawings = {
    'Alarm Clock': [
        [
            "The first mechanical alarm clock was invented in 1787 by Levi Hutchins, and it could only ring at 4 AM!",
            "The snooze button was introduced in the 1950s, and most snooze intervals are 9 minutes long due to the way old mechanical clocks were built."
        ],
        [
            "Try drawing a circular shape with a small rectangle on top for the bell.",
            "Don't forget to add two hands pointing to the time, and maybe some small legs at the bottom."
        ]
    ],
    
    'Apple': [
        [
            "Apples are part of the rose family, just like pears and plums.",
            "The science of growing apples is called pomology."
        ],
        [
            "Start by drawing a simple circle and add a small stem at the top.",
            "You can add a little leaf to the stem and maybe a bite mark on the side for fun!"
        ]
    ],
    
    'Axe': [
        [
            "The axe is one of the oldest tools, dating back to the Stone Age over 1.5 million years ago!",
            "Viking warriors used a type of axe called a Dane axe, which had a very long handle."
        ],
        [
            "Begin with a long, straight handle, then add a sharp, triangular blade at the top.",
            "Don't forget to add a small curve where the blade meets the handle to show the attachment."
        ]
    ],
    
    'Banana': [
        [
            "Bananas are berries, while strawberries are not!",
            "Bananas float in water because they are less dense than water."
        ],
        [
            "Start with a long, curved shape like a smile.",
            "Add some lines at the top for where the banana was attached to the bunch."
        ]
    ],
    
    'Bed': [
        [
            "The oldest known mattress dates back 77,000 years and was made of plant material.",
            "In the 17th century, mattresses were stuffed with straw or wool, and later, with cotton and feathers."
        ],
        [
            "Draw a large rectangle for the mattress, and add smaller rectangles for the pillows.",
            "Don't forget to add the bed frame at the bottom and maybe a blanket on top."
        ]
    ],
    
    'Bench': [
        [
            "Park benches can be found all over the world, and they're often dedicated to loved ones or historical figures.",
            "The design of benches can vary greatly around the world, with some featuring unique artistic elements or historical carvings"
        ],
        [
            "Start by drawing a long, straight seat with some vertical lines for the legs.",
            "Add a backrest by drawing a straight line above the seat, connected by vertical lines."
        ]
    ],
    
    'Bicycle': [
        [
            "The first bicycles, called 'velocipedes,' didn't have pedals, and riders had to push themselves along with their feet.",
            "The fastest speed ever recorded on a bicycle is over 183 mph, set by Denise Mueller-Korenek in 2018."
        ],
        [
            "Draw two large circles for the wheels and connect them with a straight line for the frame.",
            "Add the handlebars on one end and the seat in the middle, with pedals near the lower wheel."
        ]
    ],
    
    'Book': [
        [
            "The world's oldest known book, the 'Epic of Gilgamesh,' is over 4,000 years old and was written on clay tablets.",
            "The largest book ever created weighs over 3,000 pounds and is about the history of Bhutan."
        ],
        [
            "Start with a simple rectangle for the cover and add a few curved lines to show the pages.",
            "You can draw a bookmark sticking out from the top or some lines on the cover to represent the title."
        ]
    ]
}

# Function used to wrap text when it ends up out of the frame boundaries.
def put_multiline_text(img, text, position, font, font_scale, color, thickness, max_width):
    # Split text into words.
    words = text.split()  
    lines = []
    current_line = ""

    # Iterate through the words.
    for word in words:
        # Measure the size of the current line + the next word.
        test_line = current_line + word + " "
        (text_width, text_height), _ = cv.getTextSize(test_line, font, font_scale, thickness)
        
        # If adding the next word exceeds max_width, start a new line.
        if text_width > max_width and current_line != "":
            lines.append(current_line)
            # Start a new line with the current word.
            current_line = word + " "  
        else:
            # Add word to the current line.
            current_line = test_line  

    # Add the last line
    if current_line:
        lines.append(current_line)

    # Draw each line on the image
    x, y = position
    # Line height 
    line_height = text_height + 5  

    for i, line in enumerate(lines):
        y_line = y + i * line_height
        cv.putText(img, line, (x, y_line), font, font_scale, color, thickness)


# Define the CNN model. Used for classifying user drawings. Model is trained in the train.py file.
class CNN(nn.Module):
    """ The CNN model. """
    def __init__(self):
        # Network layers.
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 8)  # 8 classes

    def forward(self, x):
        # Flow of information.
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x, device):
        # Predict method.
        # Set model to evaluation mode.
        self.eval()
        # Send model to device.
        x = x.to(device)  
        # The following mappings are based on the class_labels variable of the create_combined_dataloaders functions (see train.py file).
        class_labels = {0: 'Alarm Clock', 1: 'Apple', 2: 'Axe', 3: 'Banana', 4: 'Bed', 5: 'Bench', 6: 'Bicycle', 7: 'Book'}
        # Disable gradient calculation.
        with torch.no_grad():
            # Network outputs
            outputs = self.forward(x)
            # Output of the highest probability.
            _, predicted = torch.max(outputs, 1)
            # Predicted label.
            predicted_label = class_labels[predicted.item()]  # Convert index to label
        return predicted_label



def thumbs_up_or_down(landmarks):
    '''Expects hand landmarks as input. Detects whether the gesture is 'thumbs up' or 'thumbs down' and returns the corresponding string 
       value. Returns False if neither gesture is detected. '''
    
    # To detect a thumbs up gesture which signifies that the user wishes to submit their drawing the following must be True:

    # 1. The hand orientation is either left or right (depending if the user is right handed or left handed respectively. That mean that the
    #    angle between the line that connects landmark num 0 and num 9 and the horizontal axis is between 135 < θ < 225 (right handed) or 
    #    for the case of left handed θ < 45 or θ > 315).

    # 2. All of the fingers (except of the thumb) are closed. The corresponding finger tip landmark points are closer to landmark 0 than
    #    their respective adjacent landmark points.

    # 3. Thumb is pointing upwards. More details in the thumb_upwards() function.

    # 4. To detect a thumbs down gesture which signifies that the user wishes to delete their drawing, everything mentioned above
    #    must be True with the small difference that the thumb must point downwards. More details can be found in the thumb_downwards() function.

    if hand_orientation(landmarks) and fingers_closed(landmarks):
        if thumb_upwards(landmarks):
            return 'thumbs_up'
        elif thumb_downwards(landmarks):
            return 'thumbs_down'
    return False


def hand_orientation(landmarks):
    'Returns True if the hand is right or left oriented. In every other case it returns False.'

    # Landmarks of interest.
    x0 = landmarks[mp_hands.HandLandmark.WRIST].x
    y0 = landmarks[mp_hands.HandLandmark.WRIST].y
    
    x9 = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    y9 = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

    # Calculate the angle between the line that connects landmark 0 with landmark 9, and the horizontal axis.
    angle_rad = math.atan2(y9-y0,  x9-x0)
    
    # Convert the angle to degrees.
    angle_deg = math.degrees(angle_rad)
    
    # Normalize the angle to be within [0, 360).
    angle_deg = (angle_deg + 360) % 360   # Adding 360 yields the corresponding positive angle of a negative one. %360 makes sure
                                          # the result is in the 0 to 360 range.

    if 135 <= angle_deg <= 225:  # Left orientation.
        return True
    # Check if the angle is in the range [315, 360) or [0, 45].
    elif angle_deg >= 315 or angle_deg <= 45:      # Rght orientation.
        return True
    else:   
        return False
    
def fingers_closed(landmarks):
    'Returns True if the fingers are closed (except thumb). In every other case it returns False.'

    # Landmarks to be considered.
    landmark_0 = landmarks[mp_hands.HandLandmark.WRIST]

    landmark_8 = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    landmark_7 = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]

    landmark_12 = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    landmark_11 = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

    landmark_16 = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    landmark_15 = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]

    landmark_20 = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    landmark_19 = landmarks[mp_hands.HandLandmark.PINKY_DIP]


    # Closed fingers conditions.
    index_finger_closed = np.sqrt((landmark_8.x - landmark_0.x) ** 2 + (landmark_8.y -landmark_0.y) ** 2) < np.sqrt((landmark_7.x - landmark_0.x) ** 2 + (landmark_7.y -landmark_0.y) ** 2)
    middle_finger_closed = np.sqrt((landmark_12.x - landmark_0.x) ** 2 + (landmark_12.y -landmark_0.y) ** 2) < np.sqrt((landmark_11.x - landmark_0.x) ** 2 + (landmark_11.y -landmark_0.y) ** 2)
    ring_finger_closed = np.sqrt((landmark_16.x - landmark_0.x) ** 2 + (landmark_16.y -landmark_0.y) ** 2) < np.sqrt((landmark_15.x - landmark_0.x) ** 2 + (landmark_15.y -landmark_0.y) ** 2)
    pinky_finger_closed = np.sqrt((landmark_20.x - landmark_0.x) ** 2 + (landmark_20.y -landmark_0.y) ** 2) < np.sqrt((landmark_19.x - landmark_0.x) ** 2 + (landmark_19.y -landmark_0.y) ** 2)
    
    if index_finger_closed and middle_finger_closed and ring_finger_closed and pinky_finger_closed:
        return True
    else:
        return False
    
def thumb_upwards(landmarks):
    '''Returns True if the thumb is pointing upwards (landmark 3, landmark 4 x-values are similar, landmark 3 y-value is greater than
       landmark 4 y-value, the knuckle (MCP) z-values are similar and the landmark 5 z-value is smaller than the landmark 17 z-value) else, 
       it returns False'''
    
    # Landmarks of interest.
    landmark_3 = landmarks[mp_hands.HandLandmark.THUMB_IP]  
    landmark_4 = landmarks[mp_hands.HandLandmark.THUMB_TIP]   

    landmark_5_z = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
    landmark_5_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

    landmark_9_z = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
    landmark_13_z = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].z

    landmark_17_z = landmarks[mp_hands.HandLandmark.PINKY_MCP].z
    landmark_17_y = landmarks[mp_hands.HandLandmark.PINKY_MCP].y 

    # Similar knuckle z-values condidion.
    mcps = [landmark_5_z, landmark_9_z, landmark_13_z, landmark_17_z]
    max_value = max(mcps)
    min_value = min(mcps)
    difference = max_value - min_value

    # ATTENTION! y-values get bigger when moving down the vertical axis, and they get smaller when moving up.
    if abs(landmark_3.x - landmark_4.x) < 0.05 and landmark_4.y < landmark_3.y and difference< 0.03 and landmark_5_y < landmark_17_y:
        return True
    else:
        return False
    

def thumb_downwards(landmarks):
    '''Returns True if the thumb is pointing downwards (landmark 3, landmark 4 x-values are similar, landmark 4 y-value is greater than
       landmark 3 y-value, the knuckle (MCP) z values are similar and the landmark 5 z-value is greater than landmark 17 z-value), 
       else it returns False'''

    # Landmarks of interest.
    landmark_3 = landmarks[mp_hands.HandLandmark.THUMB_IP] 
    landmark_4 = landmarks[mp_hands.HandLandmark.THUMB_TIP]   

    landmark_5_z = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
    landmark_5_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

    landmark_9_z = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
    landmark_13_z = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].z

    landmark_17_z = landmarks[mp_hands.HandLandmark.PINKY_MCP].z
    landmark_17_y = landmarks[mp_hands.HandLandmark.PINKY_MCP].y

    # Similar knuckle z-values condidion.
    mcps = [landmark_5_z, landmark_9_z, landmark_13_z, landmark_17_z]
    max_value = max(mcps)
    min_value = min(mcps)
    difference = max_value - min_value

    # ATTENTION! y-values get bigger when moving down the vertical axis, and they get smaller when moving up.
    if abs(landmark_3.x - landmark_4.x) < 0.05 and landmark_4.y > landmark_3.y and difference< 0.03 and landmark_5_y > landmark_17_y:
        return True
    else:
        return False



def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    *** This is the function used from Google engineers for preprocessing vector drawings and transforming them into 28x28 numpy bitmaps ***
    *** There a few minor changes to accommodate our specific case.

    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_image = raster_image.reshape((side, side))  # Reshape to (28, 28)
        raster_images.append(raster_image)
    
    return raster_images


# Making predictions.

# Initiallize classifier.
model = CNN()

# Load the saved model weights.
model.load_state_dict(torch.load('best_model.pth'))

# Move the model to the appropriate device (GPU or CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Initialize MediaPipe's hands solution module.
mp_hands = mp.solutions.hands

# Initialize the hand tracking module. It incorporates both the palm detection and the hand landmarks detection model.
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8)

# Initialize MediaPipe drawing utility.
mp_drawing = mp.solutions.drawing_utils

# Read icon of mascot (picture was taken from : https://www.freepik.com/icon/pencil_6090238).
icon = cv.imread("pencil.png")
# Convert to RGB
icon = cv.cvtColor(icon, cv.COLOR_BGR2RGB)

# Create a white canvas for drawing - same size as the webcam frame.
drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Create a white background of size 480x640 to place the icon.
background = np.ones((480, 640, 3), dtype=np.uint8) 

# Define the offset from the top-right corner.
offset_x = 20  
offset_y = 20  

# Calculate the position for the icon.
x_offset = background.shape[1] - icon.shape[1] - offset_x
y_offset = offset_y

# Place the icon on the background.
background[y_offset:y_offset + icon.shape[0], x_offset:x_offset + icon.shape[1]] = icon

# Capture web camera stream.
cap = cv.VideoCapture(0)

# Define pinch gesture detection threshold. 
pinch_threshold = 0.05

# Check if webcam was succesfully accessed.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Boolean variable that indicates whether the User was drawing on the previous frame.
drawing = False

# Boolean variable that indicates whether a text should be annotated on the frame.
draw_text = False

# Boolean variable indicating whether the user submitted a correct drawing.
correct_submission = True

# Boolean variable indicating whether a prediction message (Congratulations, or try again) should be displayed.
prediction_message_boolean = False

# Boolean variable indicating whether a cooldown period is active. After a gesture is recognized, a cooldown period of a few seconds
# is activated during which gesture recognition is deactivated.
cooldown_active = False

# List containing User's strokes. Drawing are saved as a list of strokes as in the case of the Quick Draw dataset. 
strokes = []

# Number of false submissions.
num_false_submissions = 0

# Time right before the first frame processing.
time_frame1 = time()

# Loop to continuously capture frames.
while True:
    # Capture frame-by-frame.
    ret, frame = cap.read()

    # Check if the frame was successfully captured.
    if not ret:
        print("Error: Failed to capture frame.")
        break

    
    # Flip the frame horizontally to create a mirror effect.
    mirrored_frame = cv.flip(frame, 1)

    # Display introduction text.
    time_ = time()
    if time_ -time_frame1 < 5.0:
        cv.putText(mirrored_frame, "Welcome to SketchEdAI, where we learn about objects", (260,100),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
        cv.putText(mirrored_frame, "while having fun drawing them!", (260,115),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
    elif time_ -time_frame1 > 5.0 and time_ -time_frame1 < 10.0:
        cv.putText(mirrored_frame, "My name is Sketchy and I'll be your AI drawing instructor!", (260,100),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
    elif time_ -time_frame1 > 10.0 and time_ -time_frame1 < 15.0:
        cv.putText(mirrored_frame, "You can draw by moving your hand around just make sure that", (260,100),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
        cv.putText(mirrored_frame, "you pinch together your index finger and thumb.", (260,115),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
    elif time_ -time_frame1 > 15.0 and time_ -time_frame1 < 20.0:
        cv.putText(mirrored_frame, "Make sure you are drawing inside the black border lines!", (260,100),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
    elif time_ -time_frame1 > 20.0 and time_ -time_frame1 < 27.0:
        cv.putText(mirrored_frame, "If you want to erase your drawing, make a", (260,100),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
        cv.putText(mirrored_frame, "'thumbs down' gesture.", (260,115),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
        cv.putText(mirrored_frame, "If you are ready to submit your drawing make a", (260,140),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
        cv.putText(mirrored_frame, "'thumbs up' gesture.", (260,155),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv.LINE_AA )
    elif time_ -time_frame1 > 27.0 and time_ -time_frame1 < 32.0:
        cv.putText(mirrored_frame, "Let's Begin!", (300,100),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv.LINE_AA )

    # After introduction text. Begin game.

    if time_ -time_frame1 > 32.0:
        # If user has submitted a correct drawing.
        if correct_submission is True:

            try:
                # Randomly choose the next category, a fun-fact and a hint about it.
                random_category = random.choice(list(classes_of_drawings.keys())) 
                random_fun_fact = random.choice(classes_of_drawings[random_category][0])
                random_hint = random.choice(classes_of_drawings[random_category][1])
                # Set correct_submission to False
                correct_submission = False

            except IndexError:
                # In case of an IndexError which indicates that the user has drawn correctly all categories, the game is terminated.
                print("You drew every class! You are amazing!")
                break

        # Display the chosen category and fun-fact.
        cv.putText(mirrored_frame, f"Let's Draw: {random_category}", (300,100),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),1,cv.LINE_AA )
        put_multiline_text(mirrored_frame, "Did you know that: "+random_fun_fact, (260,120), cv.FONT_HERSHEY_SIMPLEX, 0.4,(255,0,0),1,370) 
        
        # If the user has failed 3 or more times to correctly draw the category, give him a hint.
        if num_false_submissions >=3 :
            put_multiline_text(mirrored_frame, "Here is a hint: "+random_hint, (360,310), cv.FONT_HERSHEY_SIMPLEX, 0.4,(255,0,0),1,275) 

    
    # Define cropping parameters. User must draw inside the designated area. The drawings must be originally 256x256 as in the case of 
    # the Quick Draw dataset drawings.
    crop_x, crop_y, crop_w, crop_h = 0, 0, 256, 256  
    boundary_color = (0,0,0)
    boundary_thickness = 2

    # Annotate the drawing area.
    cv.line(mirrored_frame, (crop_x, crop_y), (crop_x + crop_w, crop_y), boundary_color, boundary_thickness)  # Top
    cv.line(mirrored_frame, (crop_x, crop_y), (crop_x, crop_y + crop_h), boundary_color, boundary_thickness)  # Left
    cv.line(mirrored_frame, (crop_x + crop_w, crop_y), (crop_x + crop_w, crop_y + crop_h), boundary_color, boundary_thickness)  # Right
    cv.line(mirrored_frame, (crop_x, crop_y + crop_h), (crop_x + crop_w, crop_y + crop_h), boundary_color, boundary_thickness)  # Bottom

    # Convert the frame to RGB as MediaPipe uses RGB images.
    rgb_frame = cv.cvtColor(mirrored_frame, cv.COLOR_BGR2RGB)

    # Check if a cooldown period is active or not. 
    if cooldown_active:

        # Check if the cooldown period time has passed
        if time() - cooldown_start_time >= 2:
            # Reset the cooldown
            cooldown_active = False  

        else:
            # If still in cooldown, skip the gesture detection. Display (or not) texts according to the boolean variable values.
            if draw_text:
                time_passed = time() - start_time
                if time_passed < 2.0: 
                    cv.putText(mirrored_frame, text_to_display, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                else:
                    draw_text = False
            if prediction_message_boolean:
                if time() - time_start__message < 5.0:
                    cv.putText(mirrored_frame, prediction_message, (300,190),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv.LINE_AA )
                else:
                    prediction_message_boolean = False

            # Overlay the drawing canvas and the icon on the frame.
            combined_frame = cv.addWeighted(mirrored_frame, 0.8, drawing_canvas, 0.5, 0)
            combined_frame = cv.addWeighted(combined_frame.copy(), 0.7, background, 0.6, 0)

            # Display the captured frame.
            cv.imshow('Webcam Feed', combined_frame)

            # Wait 1ms, exit the loop if 'q' is pressed. If not move on to next frame.
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            continue

    # Process the RGB frame with the Hands model.
    results = hands.process(rgb_frame)

    # Check if any hands are detected. 'multi_hand_landmarks' is a list containing one entry per hand detected. Each entry
    # contains the landmarks detected for a specific hand. If no hands are detected it will be None and the if statement will not run.
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the original frame.
            mp_drawing.draw_landmarks(mirrored_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks. 
            landmarks = hand_landmarks.landmark

            # Get the thumb and index finger tips.
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized coordinates (the landmark coordinate values are divided by image_width, image_height for x and 
            # y respectively) to pixel values.
            h, w, _ = frame.shape
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)

            # Define pinch gesture as the gesture for drawing. Distance between thumb and index finger tip must be small.
            distance = np.sqrt((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2)

            if drawing == False: # User did not draw in the previous frame.
                if distance < pinch_threshold * min(w, h): # User will draw on the current frame. A new stroke begins.

                    # In order to be in the same format as the Quick Draw dataset drawings, each stroke is a list of two lists
                    # The first contains the x coordinate values, and the second the y coordinate values.
                    stroke = [[],[]]
                    drawing = True

                    # The user must draw inside the designated area.
                    if index_tip_x < 256 and index_tip_y < 256 :
                        # Draw on the canvas.
                        cv.circle(drawing_canvas, (index_tip_x,index_tip_y), 5, (255,255,255), -1)
                        # Append pixel to stroke.
                        stroke[0].append(index_tip_x)
                        stroke[1].append(index_tip_y)

                else: # User will not draw.
                    pass
           
            elif drawing == True: # User drew in the previous frame.
                if distance < pinch_threshold * min(w, h): # User will continue to draw. Point belongs to the same stroke. Stroke continues.
                    # The user must draw inside the designated area
                    if index_tip_x < 256 and index_tip_y < 256 :
                        stroke[0].append(index_tip_x)
                        stroke[1].append(index_tip_y)
                        cv.circle(drawing_canvas, (index_tip_x,index_tip_y), 5, (255,255,255), -1)
                else: # User will not continue to draw. The stroke ends.
                    drawing = False
                    if all(sublist for sublist in stroke):
                        strokes.append(stroke)
       
            # Check for thumbs down gesture.
            if thumbs_up_or_down(landmarks) == 'thumbs_down':
                # Activate cooldown period.
                cooldown_active = True
                # Time screenshot in regards to cooldown period time calculation.
                cooldown_start_time = time()
                # Clear canvas and reset strokes.
                drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                strokes = []
                print('cleared canvas')
                draw_text = True
                # Time screenshot in regards to how long should the 'Drawing Deleted!' text be displayed.
                start_time = time()
                text_to_display = "Drawing Deleted!"
                
            # Check for thumbs up gesture.
            elif thumbs_up_or_down(landmarks) == 'thumbs_up':
                # Activate cooldown period.
                cooldown_active = True
                # Time screenshot in regards to cooldown period time calculation.
                cooldown_start_time = time()
                print('submit drawing')
                draw_text = True
                # Time screenshot in regards to how long should the 'Drawing Submitted!' text be displayed.
                start_time = time()
                text_to_display = "Drawing Submitted!"
                try:
                    # Convert 256x256 drawing to a 28x28 numpy bitmap.
                    canvas = vector_to_raster([strokes])[0]
                    # Convert to tensor.
                    image_tensor = torch.tensor(canvas, dtype=torch.float32)
                        
                    # Add channel dimension and batch dimension.
                    image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension: [1, 28, 28]
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: [1, 1, 28, 28]

                    # Predict drawing label.
                    predicted_label = model.predict(image_tensor, device)
                    # If prediction matches the category label.
                    if predicted_label == random_category:
                        correct_submission = True
                        del classes_of_drawings[random_category]
                        prediction_message_boolean = True
                        prediction_message = "Correct! Congratulations!"
                        time_start__message = time()
                        num_false_submissions = 0
                        print(f"Predicted Label CORRECTLY: {predicted_label}")

                    # If not.
                    else:
                        time_start__message = time()
                        print(f"Predicted Label FALSELY: {predicted_label}")
                        prediction_message_boolean = True
                        prediction_message = "Oops! Let's try again."
                        num_false_submissions += 1                        

                except ValueError:
                    print("Value Error. Stroke contains empty lists.")

            else:
                pass

            # Check conditions for text annotation.
            if draw_text:
                time_passed = time() - start_time
                if time_passed < 2.0: 
                    cv.putText(mirrored_frame, text_to_display, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                else:
                    draw_text = False
            if prediction_message_boolean:
                if time() - time_start__message < 5.0:
                    cv.putText(mirrored_frame, prediction_message, (300,190),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv.LINE_AA )
                else:
                    prediction_message_boolean = False
                   

    # Overlay the drawing canvas and the icon on the frame.
    combined_frame = cv.addWeighted(mirrored_frame, 0.8, drawing_canvas, 0.5, 0)
    combined_frame = cv.addWeighted(combined_frame.copy(), 0.7, background, 0.6, 0)

    # Display the captured frame.
    cv.imshow('Webcam Feed', combined_frame)

    # Wait 1ms, exit the loop if 'q' is pressed. If not move on to next frame.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window.
cap.release()
cv.destroyAllWindows()
