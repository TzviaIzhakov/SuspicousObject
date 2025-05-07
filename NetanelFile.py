import cv2
from ultralytics import YOLO
import time

# טוען את מודל YOLOv8 (מודל כללי שמזהה הרבה סוגי אובייקטים)
model = YOLO('yolov8s.pt')

# פתיחת מצלמה
cap = cv2.VideoCapture(0)

# משתנה למעקב אחרי תיקים שזוהו
tracked_bags = {}
next_bag_id = 0

# ספים לזיהוי חשד
STATIC_TIME_THRESHOLD = 30  # זמן עמידה ללא תנועה בשניות
STATIC_MOVEMENT_THRESHOLD = 20  # רדיוס זיהוי תזוזה בפיקסלים
PERSON_DISTANCE_THRESHOLD = 100  # מרחק בין תיק לאדם (פיקסלים)


# פונקציית עזר למציאת מרכז ריבוע
def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# לולאת קריאת מצלמה
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    annotated_frame = frame.copy()
    results = model(frame, conf=0.4)[0]

    detected_bags = []
    detected_people = []

    # מעבר על כל הזיהויים בפריים
    for box in results.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = get_center(x1, y1, x2, y2)

        if name in ['backpack', 'handbag']:
            detected_bags.append((center, (x1, y1, x2, y2), conf))
        elif name == 'person':
            detected_people.append(center)

    # עדכון תיקים קיימים או הוספת תיקים חדשים
    for center, box, conf in detected_bags:
        matched = False
        for bag_id in list(tracked_bags.keys()):
            bag = tracked_bags[bag_id]
            old_x, old_y = bag['last_position']
            distance = ((center[0] - old_x) ** 2 + (center[1] - old_y) ** 2) ** 0.5

            if distance < STATIC_MOVEMENT_THRESHOLD:
                tracked_bags[bag_id]['last_position'] = center
                tracked_bags[bag_id]['box'] = box
                tracked_bags[bag_id]['confidence'] = conf
                if not bag['moving']:
                    tracked_bags[bag_id]['static_timer'] = current_time - bag['start_time']
                else:
                    # תיק התחיל לזוז – אפס את הטיימר
                    tracked_bags[bag_id]['start_time'] = current_time
                    tracked_bags[bag_id]['static_timer'] = 0
                    tracked_bags[bag_id]['moving'] = False
                matched = True
                break

        if not matched:
            # זה תיק חדש – שמור אותו עם מזהה חדש
            tracked_bags[next_bag_id] = {
                'start_time': current_time,
                'last_position': center,
                'box': box,
                'static_timer': 0,
                'suspicious': False,
                'moving': False,
                'confidence': conf
            }
            next_bag_id += 1

    # ציור לכל תיק, כולל טיימר וחשד
    for bag_id, bag in tracked_bags.items():
        x1, y1, x2, y2 = bag['box']
        cx, cy = bag['last_position']
        timer = int(bag['static_timer'])

        color = (0, 255, 0)  # ירוק רגיל
        label = f'Bag {bag_id} - {timer}s'

        # אם עבר הזמן הנדרש ללא תנועה, נבדוק אם יש אנשים קרובים
        suspicious = False
        if timer >= STATIC_TIME_THRESHOLD and not bag['suspicious']:
            suspicious = True
            for px, py in detected_people:
                dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if dist < PERSON_DISTANCE_THRESHOLD:
                    suspicious = False
                    break
            if suspicious:
                bag['suspicious'] = True
                print(f'⚠️ Suspicious bag detected! ID: {bag_id}, time: {timer}s')
                color = (0, 0, 255)

        if bag['suspicious']:
            color = (0, 0, 255)
            label += " ⚠️"

        # ציור ריבוע וטקסט
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # הצגת הפריים
    cv2.imshow('Suspicious Bag Detector', annotated_frame)

    # סגירה עם Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ניקוי משאבים
cap.release()
cv2.destroyAllWindows()
