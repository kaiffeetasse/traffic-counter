import threading
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from time import sleep
import telegram
from prometheus_client import Counter, CollectorRegistry, push_to_gateway
import datetime
import time
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN")
GROUP_ID = os.environ.get("GROUP_ID")

min_width = 100  # min rectangle width
min_height = 100  # min rectangle height

offset = 20  # tolerance between pixels

pos_line = 350  # count line position

delay = 60  # FPS

detected = []
counter = 0


def center_coords(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture(2)  # 640x480
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()


def send_msg_to_telegram(file_name="vehicle.jpg"):
    try:
        bot_token = BOT_TOKEN
        group_id = GROUP_ID
        bot = telegram.Bot(token=bot_token)

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        bot.send_photo(group_id, open(file_name, 'rb'), caption=st)
    except:
        logger.error("could not send picture to telegram!")


def increase_vehicle_counter():
    try:
        registry = CollectorRegistry()
        Counter(
            'oberdorfverkehr',
            'Oberdorfverkehr Fahrzeug Counter',
            registry=registry)

        push_to_gateway(
            gateway='https://agw.pils.cf',
            job='oberdorfverkehr',
            registry=registry)
    except:
        logger.error("could not increase vehicle counter metrics!")


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


while True:
    detected_vehicle = False
    ret, frame1 = cap.read()

    # frame1 = increase_brightness(frame1, 50)

    sleep_time = float(1 / delay)
    sleep(sleep_time)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    detector_frame = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    detector_frame = cv2.morphologyEx(detector_frame, cv2.MORPH_CLOSE, kernel)
    outline, h = cv2.findContours(detector_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (i, c) in enumerate(outline):
        (x, y, w, h) = cv2.boundingRect(c)
        valid_outline = (w >= min_width) and (h >= min_height)
        if not valid_outline:
            # print("outline not big enough! (" + str(w) + "x" + str(h) + ")")
            continue

        # print("outline big enough! (" + str(w) + "x" + str(h) + ")")

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = center_coords(x, y, w, h)
        detected.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detected:
            # print("x = " + str(x) + ", y = " + str(y))
            # if y < (pos_line + offset) and y > (pos_line - offset):
            if 320 + offset > x > 320 - offset:
                counter += 1
                detected_vehicle = True

                detected.remove((x, y))
                print("vehicle detected: " + str(counter))

                cv2.imwrite("vehicle.jpg", frame1)

                msg_to_tg_thread = threading.Thread(target=send_msg_to_telegram, name="msg_to_tg_thread")
                msg_to_tg_thread.start()

                increase_vc_thread = threading.Thread(target=increase_vehicle_counter, name="increase_vc")
                increase_vc_thread.start()

    if detected_vehicle:
        cv2.line(frame1, (320, 25), (320, 480 - 25), (0, 127, 255), 3)
    else:
        cv2.line(frame1, (320, 25), (320, 480 - 25), (255, 127, 0), 3)

    cv2.putText(frame1, "vehicle count: " + str(counter), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("original video", frame1)
    cv2.imshow("detector", detector_frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
