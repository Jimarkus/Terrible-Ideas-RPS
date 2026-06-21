#! /usr/bin/env python3

"""Embeddable version of RPS_Vision_System/RPS_vs_Computer.py.

The original script is not embeddable at all: it runs its own cv2.imshow
window, reads input via cv2.waitKey, and executes its game loop at import
time as bare module-level globals. This wraps the same mediapipe-hand /
rock-paper-scissor logic in an RPSGame class with the same handle_event() /
update() / draw() / done / quit_requested contract as the other embedded
minigames, so a host can render it into a shared pygame surface and feed it
pygame events instead.
"""

import os
import random
import sys
import time

import cv2
import numpy as np
import pygame
from pygame.locals import KEYUP, K_ESCAPE, K_SPACE

RPS_VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rps_vision')
if RPS_VISION_DIR not in sys.path:
    sys.path.insert(0, RPS_VISION_DIR)

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition


CHOICES = ['rock', 'paper', 'scissor']

GESTURE_TO_CHOICE = {
    'fist': 'rock',
    'five': 'paper',
    'three': 'scissor',
    'yeah': 'scissor',
}

COUNTDOWN_SEQUENCE = ['3', '2', '1', 'SHOOT!']
COUNTDOWN_STEP_SEC = 0.8
RESULT_DISPLAY_SEC = 3.0
ICON_SIZE = 100


def judge(player, computer):
    if player == computer:
        return 'Tie'
    beats = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    if beats[player] == computer:
        return 'You win'
    return 'Computer wins'


def load_icon(path):
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise FileNotFoundError('Could not load icon image: %s' % path)
    icon = cv2.resize(icon, (ICON_SIZE, ICON_SIZE))
    if icon.shape[2] == 3:
        alpha = np.full((ICON_SIZE, ICON_SIZE, 1), 255, dtype=np.uint8)
        icon = np.concatenate([icon, alpha], axis=2)
    return icon


def overlay_icon(img, icon, x, y):
    img_height, img_width, _ = img.shape
    h, w = icon.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img_width), min(y + h, img_height)
    if x1 <= x0 or y1 <= y0:
        return img

    icon_crop = icon[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = icon_crop[:, :, 3:4] / 255.0
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - alpha) + icon_crop[:, :, :3] * alpha
    return img


def draw_hand_skeleton(img, disp, param):
    img_height, img_width, _ = img.shape
    for p in param:
        if p['class'] is None:
            continue
        for i in range(21):
            x = int(p['keypt'][i, 0])
            y = int(p['keypt'][i, 1])
            if x > 0 and y > 0 and x < img_width and y < img_height:
                start = p['keypt'][disp.ktree[i], :]
                x_ = int(start[0])
                y_ = int(start[1])
                if x_ > 0 and y_ > 0 and x_ < img_width and y_ < img_height:
                    cv2.line(img, (x_, y_), (x, y), disp.color[i], 2)
                cv2.circle(img, (x, y), 5, disp.color[i], -1)
    return img


def cv2_frame_to_surface(frame):
    # cv2 arrays are (height, width, 3); pygame.surfarray wants
    # (width, height, 3). np.rot90 "fixes" the shape but also rotates the
    # pixels, which combined with surfarray's axis convention nets out to a
    # left-right mirror -- harmless for plain video, but it flips any text
    # baked into the frame (e.g. cv2.putText) backwards. A plain transpose
    # reshapes without rotating/mirroring.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


class RPSGame:
    """One Rock-Paper-Scissor-vs-computer session, embeddable in a host loop.

    Mirrors FlappyBirdGame's contract (handle_event/update/draw, done,
    quit_requested), plus close() to release the mediapipe hand-landmarker
    when the host switches away from this minigame -- it's not part of the
    shared contract since other minigames don't hold native handles.

    Arguments:
    display_surface: Surface to draw into; the camera frame is scaled to
        whatever size it is.
    cap: An already-open cv2.VideoCapture, owned by the host and shared with
        other camera-based minigames.
    """

    def __init__(self, display_surface, cap):
        self.display_surface = display_surface
        self.cap = cap

        self.pipe = MediaPipeHand(static_image_mode=False, max_num_hands=1)
        self.disp = DisplayHand(max_num_hands=1)
        self.gest = GestureRecognition(mode='eval')

        images_dir = os.path.join(RPS_VISION_DIR, 'images')
        self.icons = {
            'rock': load_icon(os.path.join(images_dir, 'Rockimage.png')),
            'paper': load_icon(os.path.join(images_dir, 'Paperimage.png')),
            'scissor': load_icon(os.path.join(images_dir, 'Scissorimage.png')),
        }

        self.state = 'wait'  # wait -> countdown -> result -> wait
        self.computer_choice = None
        self.player_choice = None
        self.result_text = None
        self.countdown_start = None
        self.result_start = None

        self.done = False
        self.quit_requested = False
        self._frame = None

    def handle_event(self, event):
        if event.type == KEYUP and event.key == K_ESCAPE:
            self.quit_requested = True
        elif event.type == KEYUP and event.key == K_SPACE and self.state == 'wait':
            self.computer_choice = random.choice(CHOICES)
            self.state = 'countdown'
            self.countdown_start = time.time()

    def update(self, dt=None):
        ret, img = self.cap.read()
        if not ret:
            return
        img = cv2.flip(img, 1)

        img.flags.writeable = False
        param = self.pipe.forward(img)
        for p in param:
            if p['class'] is not None:
                p['gesture'] = self.gest.eval(p['angle'])
        img.flags.writeable = True

        img = draw_hand_skeleton(img, self.disp, param)

        p = param[0]
        if p['class'] is not None:
            live_choice = GESTURE_TO_CHOICE.get(p['gesture'])
            if live_choice is not None:
                x = int(p['keypt'][0, 0]) - 30
                y = int(p['keypt'][0, 1]) + 40
                img = overlay_icon(img, self.icons[live_choice], x, y)

        img_height, img_width, _ = img.shape

        if self.state == 'countdown':
            elapsed = time.time() - self.countdown_start
            idx = int(elapsed // COUNTDOWN_STEP_SEC)
            if idx < len(COUNTDOWN_SEQUENCE):
                text = COUNTDOWN_SEQUENCE[idx]
                size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
                x = int((img_width - size[0]) / 2)
                cv2.putText(img, text, (x, 150), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 3)
            else:
                p = param[0]
                self.player_choice = GESTURE_TO_CHOICE.get(p['gesture'])
                if self.player_choice is None:
                    self.result_text = 'No hand gesture detected'
                else:
                    self.result_text = judge(self.player_choice, self.computer_choice)
                self.state = 'result'
                self.result_start = time.time()

        elif self.state == 'result':
            line1 = 'You: %s' % (self.player_choice.upper() if self.player_choice else 'NONE')
            line2 = 'Computer: %s' % self.computer_choice.upper()
            line3 = self.result_text.upper()

            for i, line in enumerate([line1, line2, line3]):
                size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                x = int((img_width - size[0]) / 2)
                cv2.putText(img, line, (x, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            if time.time() - self.result_start > RESULT_DISPLAY_SEC:
                self.state = 'wait'

        else:  # state == 'wait'
            text = 'Press SPACE to play, Esc for menu'
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x = int((img_width - size[0]) / 2)
            cv2.putText(img, text, (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        self._frame = img

    def draw(self):
        if self._frame is None:
            return
        size = self.display_surface.get_size()
        frame_surface = cv2_frame_to_surface(self._frame)
        frame_surface = pygame.transform.scale(frame_surface, size)
        self.display_surface.blit(frame_surface, (0, 0))

    def close(self):
        """Release the mediapipe hand-landmarker. Call when leaving this game."""
        self.pipe.pipe.close()
