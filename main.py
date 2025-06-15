

!pip install -r '/content/req.txt'



!pip install pyngrok

from pyngrok import ngrok
!ngrok update

import os
from threading import Thread

from google.colab import userdata
key=userdata.get('authotoken_key')

ngrok.set_auth_token(key)

def run_streamlit():
    # Change the port if 8501 is already in use or if you prefer another port
    os.system('streamlit run /content/medical_app.py --server.port 8501')

# Start a thread to run the Streamlit app
thread = Thread(target=run_streamlit)
thread.start()
!pkill ngrok
# Open a tunnel to the streamlit port 8501
public_url = ngrok.connect(8501)
print('Your Streamlit app is live at:', public_url)

