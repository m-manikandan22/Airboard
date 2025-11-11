import cv2
import mediapipe as mp
import numpy as np

# --- Mediapipe Hand Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.8)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
canvas = None

# --- Colors and Brush Settings ---
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
color_index = 0
current_color = colors[color_index]
color_change_cooldown = 0

# --- Filters and State ---
smooth_point = None  # For EMA smoothing
prev_point = None    # For drawing lines
draw_active = False  # Hysteresis state for pinching

ema_alpha = 0.25       # Smoothing factor (0 < alpha < 1). Lower is smoother.
touch_threshold = 28   # Distance (px) to consider thumb/index a "pinch"
hysteresis_margin = 8  # Prevents flickering on/off

# --- Helper Functions ---

def distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_finger_extended(tip, pip):
    """Checks if a finger is extended (tip is vertically higher than pip)."""
    return tip[1] < pip[1]

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Flip for selfie view
    h, w, _ = frame.shape

    # Initialize canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Process frame with Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    gesture = "None"
    cursor_mode = "default"  # Controls cursor type: 'default', 'draw', 'erase'

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            
            # Get landmark coordinates in pixels
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]

            # --- Key landmark points for gestures ---
            thumb_tip_pt = pts[4]
            thumb_mcp_pt = pts[2]
            thumb_pip_pt = pts[3] # Added landmark for thumb
            index_tip_pt = pts[8]
            index_pip_pt = pts[6]
            middle_tip_pt = pts[12]
            middle_pip_pt = pts[10]
            ring_tip_pt = pts[16]
            ring_pip_pt = pts[14]
            pinky_tip_pt = pts[20]
            pinky_pip_pt = pts[18]

            # --- 1. Get Finger Extension States ---
            # Use simple vertical check for all
            thumb_ext = is_finger_extended(thumb_tip_pt, thumb_pip_pt) and \
                        (thumb_tip_pt[1] < index_pip_pt[1]) # Thumb is truly "up"
            index_ext = is_finger_extended(index_tip_pt, index_pip_pt)
            middle_ext = is_finger_extended(middle_tip_pt, middle_pip_pt)
            ring_ext = is_finger_extended(ring_tip_pt, ring_pip_pt)
            pinky_ext = is_finger_extended(pinky_tip_pt, pinky_pip_pt)

            # --- 2. Define High-Level Gestures ---
            
            # Eraser: ONLY Index, Middle, Ring are UP
            is_eraser = index_ext and middle_ext and ring_ext and \
                        (not pinky_ext) and (not thumb_ext)
            
            # Color Change: Thumbs Up (ONLY Thumb is UP)
            is_thumbs_up = thumb_ext and \
                           (not index_ext) and (not middle_ext) and \
                           (not ring_ext) and (not pinky_ext)

            # --- 3. Pinch (Drawing) Logic ---
            thumb_index_dist = distance(thumb_tip_pt, index_tip_pt)
            
            # Midpoint of pinch is the drawing cursor
            mid_x = (thumb_tip_pt[0] + index_tip_pt[0]) // 2
            mid_y = (thumb_tip_pt[1] + index_tip_pt[1]) // 2
            curr_point = np.array([mid_x, mid_y])

            # Apply EMA smoothing to the cursor position
            if smooth_point is None:
                smooth_point = curr_point.copy()
            smooth_point = ema_alpha * curr_point + (1 - ema_alpha) * smooth_point
            cursor_pos = tuple(smooth_point.astype(int))

            # Hysteresis logic for activating drawing
            if thumb_index_dist < touch_threshold - hysteresis_margin:
                draw_active = True
            elif thumb_index_dist > touch_threshold + hysteresis_margin:
                draw_active = False

            # --- 4. Gesture State Machine (Prioritized) ---
            
            if draw_active:
                gesture = "Drawing"
                cursor_mode = "draw"
                if prev_point is not None:
                    # Interpolate points for smooth drawing
                    d = int(distance(prev_point, smooth_point))
                    for i in range(1, d + 1):
                        t = i / d
                        xi = int(prev_point[0] * (1 - t) + smooth_point[0] * t)
                        yi = int(prev_point[1] * (1 - t) + smooth_point[1] * t)
                        cv2.circle(canvas, (xi, yi), 3, current_color, -1)
                prev_point = smooth_point

            elif is_eraser:
                gesture = "Eraser"
                cursor_mode = "erase"
                eraser_size = 80  # Matched to cursor
                half_size = eraser_size // 2
                p1 = (cursor_pos[0] - half_size, cursor_pos[1] - half_size)
                p2 = (cursor_pos[0] + half_size, cursor_pos[1] + half_size)
                # Erase in a square region
                cv2.rectangle(canvas, p1, p2, (0, 0, 0), -1) # Changed from circle
                prev_point = None

            elif is_thumbs_up and color_change_cooldown == 0: # Changed from is_color_change
                gesture = "Color Change"
                cursor_mode = "default"
                color_index = (color_index + 1) % len(colors)
                current_color = colors[color_index]
                color_change_cooldown = 30  # Cooldown for 30 frames
                prev_point = None
            
            else:
                # This state includes "Open Palm" (all 5 fingers up)
                # and any other unrecognized gesture.
                gesture = "None"
                cursor_mode = "default"
                prev_point = None

            # --- 5. Draw Cursor based on mode ---
            if cursor_mode == "draw":
                cv2.circle(frame, cursor_pos, 8, current_color, -1)
            elif cursor_mode == "erase":
                # Draw the "partially filled white box"
                eraser_size = 80
                half_size = eraser_size // 2
                x1 = cursor_pos[0] - half_size
                y1 = cursor_pos[1] - half_size
                x2 = cursor_pos[0] + half_size
                y2 = cursor_pos[1] + half_size

                # Get the actual region on the frame (clamped)
                frame_x1, frame_y1 = max(x1, 0), max(y1, 0)
                frame_x2, frame_y2 = min(x2, w), min(y2, h)

                if frame_x1 < frame_x2 and frame_y1 < frame_y2:
                    # Get the ROI from the frame
                    roi = frame[frame_y1:frame_y2, frame_x1:frame_x2]

                    # Create a white overlay for this ROI
                    white_overlay = np.full(roi.shape, (255, 255, 255), dtype=np.uint8)
                    
                    # Blend (alpha = 0.3 for transparency)
                    alpha = 0.3
                    cv2.addWeighted(white_overlay, alpha, roi, 1 - alpha, 0, roi)
                    # The `roi` is a view on `frame`, so `frame` is modified.

                # Draw the white outline (using original, non-clamped coords for consistency)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            else: # "default" mode
                cv2.circle(frame, cursor_pos, 8, (150, 150, 150), 2) # Grey outline

            # Draw hand landmarks on the original frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        # No hand detected
        smooth_point = None  # Reset smoother
        prev_point = None    # Reset drawing state
        draw_active = False  # Reset pinch state

    # Decrement cooldown timer
    if color_change_cooldown > 0:
        color_change_cooldown -= 1

    # --- Display ---
    # Blend the canvas (drawing) and the frame (webcam)
    # 1.0 for frame, 0.7 for canvas
    blended = cv2.addWeighted(frame, 1.0, canvas, 0.7, 0)

    # Add UI Text
    cv2.putText(blended, f"Mode: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blended, f"Color: {['Red','Green','Blue'][color_index]}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 3)

    cv2.imshow("Air Writer (3-Finger Erase)", blended)

    # Exit on 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()

