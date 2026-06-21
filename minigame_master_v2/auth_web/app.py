#! /usr/bin/env python3

"""Flask app for the soap-dispenser 2FA + LinkedIn-OAuth flow.

Adapted from indexauth.py so it can run in a background thread inside the
pygame master process (see ../auth_embed.py) instead of as its own
standalone process. The only behavioral addition is `auth_complete`, a
threading.Event the embedding minigame polls once per frame to find out
when the browser-side flow has finished -- everything else (routes,
TOTP/QR generation, LinkedIn OAuth exchange) is unchanged from the original.
"""

import io
import base64
import os
import threading

from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for
import pyotp
import qrcode
import requests

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or os.urandom(24)

# Set once the LinkedIn callback completes successfully; the pygame minigame
# polls this instead of blocking on the Flask thread.
auth_complete = threading.Event()
auth_result = {}


@app.route('/')
def eula():
    return render_template('eula.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return '<h2> login page here </h2>'


@app.route('/display')
def display():
    url = os.getenv('LINKEDIN_REDIRECT_URI', '/')
    img = qrcode.make(url)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return render_template('display.html', qr_code=img_b64)


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        code = request.form.get('code')
        secret = session.get('totp_secret')

        if not secret:
            return redirect(url_for('totp'))

        totp_obj = pyotp.TOTP(secret)
        if totp_obj.verify(code):
            return redirect(url_for('linkedin'))
        else:
            return render_template('verify.html', error='Invalid code, try again.')

    return redirect(url_for('totp'))


@app.route('/totp')
def totp():
    secret = pyotp.random_base32()
    session['totp_secret'] = secret

    totp_obj = pyotp.TOTP(secret)
    uri = totp_obj.provisioning_uri(name='SoapUser', issuer_name='Dispenser Auth')
    img = qrcode.make(uri)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return render_template('verify.html', qr_code=img_b64)


LINKEDIN_CLIENT_ID = os.getenv('LINKEDIN_CLIENT_ID')
LINKEDIN_CLIENT_SECRET = os.getenv('LINKEDIN_CLIENT_SECRET')
LINKEDIN_REDIRECT_URI = os.getenv('LINKEDIN_REDIRECT_URI')


@app.route('/linkedin')
def linkedin():
    auth_url = (
        'https://www.linkedin.com/oauth/v2/authorization'
        '?response_type=code'
        f'&client_id={LINKEDIN_CLIENT_ID}'
        f'&redirect_uri={LINKEDIN_REDIRECT_URI}'
        '&scope=openid%20profile%20email'
    )
    return redirect(auth_url)


@app.route('/linkedin/callback')
def linkedin_callback():
    code = request.args.get('code')

    token_response = requests.post(
        'https://www.linkedin.com/oauth/v2/accessToken',
        data={
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': LINKEDIN_REDIRECT_URI,
            'client_id': LINKEDIN_CLIENT_ID,
            'client_secret': LINKEDIN_CLIENT_SECRET,
        }
    )
    access_token = token_response.json().get('access_token')

    profile_response = requests.get(
        'https://api.linkedin.com/v2/userinfo',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    profile = profile_response.json()

    name = profile.get('name', 'Unknown')
    picture = profile.get('picture', '')

    auth_result['name'] = name
    auth_result['picture'] = picture
    auth_complete.set()

    return render_template('success.html', name=name, picture=picture)


def run(host='127.0.0.1', port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)
