import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

#lips
lips_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
lips_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

#eyeliner
eyeliner_upper_left = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
eyeliner_upper_right = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#eyebrow
left_eyebrow = [70, 63, 105, 66, 107]
right_eyebrow = [336, 296, 334, 293, 300]

#foundation
exclude_regions = [lips_outer, lips_inner, eyeliner_upper_left, eyeliner_upper_right, left_eyebrow, right_eyebrow]

#extract points from landmarks
def get_points(landmarks, indices):
    return [(int(landmarks[i].x * frame_width), int(landmarks[i].y * frame_height)) for i in indices]

#mask for the foundation
def create_foundation_mask(frame, landmarks):
    h, w, _ = frame.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for region in exclude_regions:
        points = get_points(landmarks, region)
        points_np = np.array(points, np.int32)
        cv2.fillPoly(mask, [points_np], 0)
    return mask

#apply lipstick
def apply_lipstick(frame, landmarks, lipstick_color):
    h, w, _ = frame.shape
    global frame_width, frame_height
    frame_width, frame_height = w, h
    
    outer_points = get_points(landmarks, lips_outer)
    inner_points = get_points(landmarks, lips_inner)

    mask_outer = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_inner = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    points_outer_np = np.array(outer_points, np.int32)
    points_inner_np = np.array(inner_points, np.int32)
    
    cv2.fillPoly(mask_outer, [points_outer_np], 255)
    cv2.fillPoly(mask_inner, [points_inner_np], 255)
    
    mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

    colored_mask = np.zeros_like(frame)
    colored_mask[mask == 255] = lipstick_color

    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
    return frame

#apply eyeliner
def apply_eyeliner(frame, landmarks, eyeliner_color):
    h, w, _ = frame.shape
    global frame_width, frame_height
    frame_width, frame_height = w, h

    upper_left_points = get_points(landmarks, eyeliner_upper_left)
    upper_right_points = get_points(landmarks, eyeliner_upper_right)

    mask_upper_left = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_upper_right = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    points_upper_left_np = np.array(upper_left_points, np.int32)
    points_upper_right_np = np.array(upper_right_points, np.int32)
    
    cv2.fillPoly(mask_upper_left, [points_upper_left_np], 255)
    cv2.fillPoly(mask_upper_right, [points_upper_right_np], 255)
    
    # Apply eyeliner 
    colored_mask = np.zeros_like(frame)
    colored_mask[mask_upper_left == 255] = eyeliner_color
    colored_mask[mask_upper_right == 255] = eyeliner_color

    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
    return frame

#apply eyebrow color
def apply_eyebrow_color(frame, landmarks, eyebrow_color):
    h, w, _ = frame.shape
    global frame_width, frame_height
    frame_width, frame_height = w, h

    left_eyebrow_points = get_points(landmarks, left_eyebrow)
    right_eyebrow_points = get_points(landmarks, right_eyebrow)

    mask_left_eyebrow = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_right_eyebrow = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    points_left_eyebrow_np = np.array(left_eyebrow_points, np.int32)
    points_right_eyebrow_np = np.array(right_eyebrow_points, np.int32)
    
    cv2.fillPoly(mask_left_eyebrow, [points_left_eyebrow_np], 255)
    cv2.fillPoly(mask_right_eyebrow, [points_right_eyebrow_np], 255)

    colored_mask = np.zeros_like(frame)
    colored_mask[mask_left_eyebrow == 255] = eyebrow_color
    colored_mask[mask_right_eyebrow == 255] = eyebrow_color

    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
    return frame

#apply foundation
def apply_foundation(frame, landmarks, foundation_color):
    h, w, _ = frame.shape
    global frame_width, frame_height
    frame_width, frame_height = w, h
    
    mask = create_foundation_mask(frame, landmarks)

    colored_mask = np.zeros_like(frame)
    colored_mask[mask == 255] = foundation_color

    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
    return frame

# Streamlit UI
st.title('Virtual Makeup Try-On')

# Menu
st.sidebar.title('Select Makeup Option')
makeup_option = st.sidebar.radio('Choose Makeup', ['Lipstick', 'Foundation', 'Eyeliner', 'Eyebrows'])

if makeup_option == 'Lipstick':
    lipstick_color = st.sidebar.radio('Choose Lipstick Color', ['Deep Red', 'Plum Perfection', 'Vivid Orchid'])
    colors = {
        'Deep Red' : (30, 30, 166),
        'Plum Perfection' : (66, 46, 145),
        'Vivid Orchid' : (121, 30, 166)
    }
    lipstick_color_bgr = colors.get(lipstick_color, (30, 30, 166))

elif makeup_option == 'Eyeliner':
    eyeliner_color = st.sidebar.radio('Choose Eyeliner Color', ['Black', 'Blue'])
    colors = {
        'Black' : (0, 0, 0),
        'Blue' : (255, 0, 0)
    }
    eyeliner_color_bgr = colors.get(eyeliner_color, (0, 0, 0))

elif makeup_option == 'Eyebrows':
    eyebrow_color = st.sidebar.radio('Choose Eyebrow Color', ['Natural Brown', 'Grey Brown'])
    colors = {
        'Natural Brown': (27, 46, 73),  
        'Grey Brown': (19, 21, 52)
    }
    eyebrow_color_bgr = colors.get(eyebrow_color, (27,46,73))

elif makeup_option == 'Foundation':
    foundation_color = st.sidebar.radio('Choose LFoundation Color', ['Natural Beige', 'Toffee', 'Warm Nude'])
    colors = {
        'Natural Beige' : (126, 169, 232),
        'Toffee' : (96, 148, 215),
        'Warm Nude' :  (156, 196, 254)
    }
    foundation_color_bgr = colors.get(foundation_color, (96,148,215) )

st.write("Ensure your webcam is enabled and positioned correctly.")
frame_placeholder = st.empty()


def process_video_stream(lipstick_color_bgr=None, eyeliner_color_bgr=None, eyebrow_color_bgr=None, foundation_color_bgr=None):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                if makeup_option == 'Lipstick':
                    frame = apply_lipstick(frame, landmarks, lipstick_color_bgr)
                elif makeup_option == 'Eyeliner':
                    frame = apply_eyeliner(frame, landmarks, eyeliner_color_bgr)
                elif makeup_option == 'Eyebrows':
                    frame = apply_eyebrow_color(frame, landmarks, eyebrow_color_bgr)
                elif makeup_option == 'Foundation':
                    frame = apply_foundation(frame, landmarks, foundation_color_bgr)

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        frame_placeholder.image(frame_rgb, channels='RGB', use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if makeup_option == 'Lipstick':
    process_video_stream(lipstick_color_bgr=lipstick_color_bgr)
elif makeup_option == 'Eyeliner':
    process_video_stream(eyeliner_color_bgr=eyeliner_color_bgr)
elif makeup_option == 'Eyebrows':
    process_video_stream(eyebrow_color_bgr=eyebrow_color_bgr)
elif makeup_option == 'Foundation':
    process_video_stream(foundation_color_bgr=foundation_color_bgr)