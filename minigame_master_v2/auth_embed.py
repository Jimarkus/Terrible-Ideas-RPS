#! /usr/bin/env python3

"""Embeddable wrapper around auth_web/app.py's Flask-based 2FA + LinkedIn
login flow, for the master pygame menu.

The Flask app and the OAuth redirect dance both have to be real HTTP --
there's no way to do the LinkedIn consent screen without a browser hitting
a real redirect URI. So instead of porting that flow into pygame widgets,
this starts the Flask app once in a background thread and opens the Pi's
default browser to it with webbrowser.open(), which returns immediately.
The pygame side never blocks on it: AuthGame just polls auth_web.app's
auth_complete threading.Event() once per frame (see update()) the same way
the host's main loop polls minigame.done, so the rest of the menu/other
minigames keep running normally while the user finishes the flow in their
browser.

Same handle_event()/update()/draw()/done/quit_requested contract as the
other embedded minigames (see rps_embed.RPSGame's docstring).
"""

import os
import sys
import threading
import time
import webbrowser

import pygame
from pygame.locals import KEYUP, K_ESCAPE

AUTH_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auth_web')
if AUTH_WEB_DIR not in sys.path:
    sys.path.insert(0, AUTH_WEB_DIR)

import app as auth_app  # noqa: E402  (path must be set up first)

HOST, PORT = '127.0.0.1', 5000
BASE_URL = f'http://{HOST}:{PORT}/'

_server_thread = None


def _ensure_server_running():
    """Start the Flask app in a daemon thread, once per process."""
    global _server_thread
    if _server_thread is None:
        _server_thread = threading.Thread(
            target=auth_app.run, kwargs={'host': HOST, 'port': PORT}, daemon=True)
        _server_thread.start()
        time.sleep(0.5)  # give the dev server a moment to bind before we open a browser tab


class AuthGame:
    """One soap-dispenser login session, embeddable in the host pygame loop.

    The actual login UI (EULA, TOTP QR, LinkedIn consent) all happens in the
    system browser, not in this pygame surface -- this just shows a status
    screen and waits for auth_web.app.auth_complete to be set.
    """

    def __init__(self, display_surface, font):
        self.display_surface = display_surface
        self.font = font

        self.done = False
        self.quit_requested = False
        self.result_name = None

        _ensure_server_running()
        auth_app.auth_complete.clear()
        auth_app.auth_result.clear()
        webbrowser.open(BASE_URL)

    def handle_event(self, event):
        if event.type == KEYUP and event.key == K_ESCAPE:
            self.quit_requested = True

    def update(self, dt=None):
        if auth_app.auth_complete.is_set():
            self.result_name = auth_app.auth_result.get('name', 'Unknown')
            self.done = True

    def draw(self):
        self.display_surface.fill((10, 40, 10))
        if self.result_name is not None:
            lines = [
                'AUTHENTICATION COMPLETE',
                '',
                'Welcome, %s -- soap dispensed.' % self.result_name,
                '',
                'Returning to menu...',
            ]
        else:
            lines = [
                'WAITING FOR LOGIN',
                '',
                'Finish the EULA / 2FA / LinkedIn login in your browser.',
                'Other minigames keep running -- this just waits in the background.',
                '',
                'Esc - cancel and return to menu',
            ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (255, 255, 255))
            self.display_surface.blit(surf, (20, 20 + i * 28))
