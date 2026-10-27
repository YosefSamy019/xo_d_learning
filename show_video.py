#   pip install pyvirtualdisplay
#   apt-get install -y xvfb python-opengl ffmpeg


from IPython.display import HTML
from IPython import display as ipythondisplay
import io
import base64

def show_video(file):
      mp4 = file
      video = io.open(mp4, 'r+b').read()
      encoded = base64.b64encode(video)
      ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                  loop controls style="height: 400px;">
                  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii'))))
