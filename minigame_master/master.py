#! /usr/bin/env python3

"""Master file hosting three embedded minigames in one pygame app:
Emotion Scanner (mediapipe-free, cv2 + hsemotion), Flappy Bird (velocity
physics), and Rock-Paper-Scissors vs Computer (mediapipe hand tracking).

All three minigames share this process's single pygame.init()/display/event
loop/clock; none of them create their own. The two camera-based minigames
(emotion, RPS) also share a single cv2.VideoCapture(0), opened lazily on
first use and kept open for the rest of the session, so switching between
them doesn't repeatedly open/close the webcam device.

Controls:
- On the menu: 1 = Emotion Scanner, 2 = Flappy Bird, 3 = Rock-Paper-Scissors,
  Esc = quit the whole app.
- In a minigame: Esc returns to the menu.
"""

import sys

import cv2
import pygame
from pygame.locals import QUIT, KEYUP, K_ESCAPE, K_1, K_2, K_3

from flappybird_velocity import WIN_WIDTH as FB_WIDTH, WIN_HEIGHT as FB_HEIGHT, FPS, load_images
from flappybird_embed import FlappyBirdGame

from emotion_embed import EmotionGame
from rps_embed import RPSGame


WIN_WIDTH, WIN_HEIGHT = 960, 720

MENU, PLAYING_EMOTION, PLAYING_FLAPPY, PLAYING_RPS = 'menu', 'emotion', 'flappy', 'rps'


def draw_menu(display_surface, font):
    display_surface.fill((10, 10, 40))
    lines = [
        'MASTER MINIGAME MENU',
        '',
        '1 - Emotion Scanner (webcam)',
        '2 - Flappy Bird (velocity physics)',
        '3 - Rock-Paper-Scissors vs Computer (webcam)',
        'Esc - Quit',
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, (255, 255, 255))
        display_surface.blit(surf, (20, 20 + i * 28))


class CameraResources:
    """Lazily-loaded resources shared by the two camera-based minigames.

    Loading the emotion ONNX model and opening the webcam are both slow, so
    they happen once on first use rather than once per round.
    """

    def __init__(self):
        self.cap = None
        self.face_cascade = None
        self.recognizer = None

    def get_cap(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        return self.cap

    def get_emotion_models(self):
        if self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.recognizer is None:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self.recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
        return self.face_cascade, self.recognizer

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main():
    pygame.init()
    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Master minigame demo')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    flappy_images = load_images()
    flappy_rect = pygame.Rect((WIN_WIDTH - FB_WIDTH) // 2,
                               (WIN_HEIGHT - FB_HEIGHT) // 2,
                               FB_WIDTH, FB_HEIGHT)

    camera = CameraResources()

    state = MENU
    minigame = None
    done = False

    while not done:
        dt = clock.tick(FPS) / 1000.0

        events = pygame.event.get()
        for e in events:
            if e.type == QUIT:
                done = True

        if state == MENU:
            for e in events:
                if e.type == KEYUP and e.key == K_ESCAPE:
                    done = True
                elif e.type == KEYUP and e.key == K_1:
                    state = PLAYING_EMOTION
                    face_cascade, recognizer = camera.get_emotion_models()
                    minigame = EmotionGame(display_surface, camera.get_cap(),
                                            face_cascade, recognizer)
                elif e.type == KEYUP and e.key == K_2:
                    state = PLAYING_FLAPPY
                    display_surface.fill((0, 0, 0))
                    minigame = FlappyBirdGame(display_surface.subsurface(flappy_rect),
                                               flappy_images)
                elif e.type == KEYUP and e.key == K_3:
                    state = PLAYING_RPS
                    minigame = RPSGame(display_surface, camera.get_cap())
            draw_menu(display_surface, font)

        else:
            for e in events:
                minigame.handle_event(e)

            if state == PLAYING_FLAPPY:
                minigame.update()
            else:
                minigame.update(dt)
            minigame.draw()

            if minigame.quit_requested or minigame.done:
                if hasattr(minigame, 'close'):
                    minigame.close()
                state = MENU
                minigame = None

        pygame.display.flip()

    if minigame is not None and hasattr(minigame, 'close'):
        minigame.close()
    camera.release()
    pygame.quit()
    sys.exit(0)

if __name__ == '__main__':
    main()