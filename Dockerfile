FROM python:3.8.6-buster

COPY project_drowsy /project_drowsy
COPY airhorn.wav /airhorn.wav
COPY streamlit_app.py /streamlit_app.py
COPY setup.py /setup.py
COPY alert_example.png /alert_example.png
COPY drowsy_example.png /drowsy_example.png
COPY packages.txt /packages.txt
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN apt-get update
RUN apt-get install -y freeglut3-dev
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y libasound2-dev

RUN pip install -r requirements.txt
CMD streamlit run streamlit_app.py
