FROM python:3.8

ADD . ./FOCUS
WORKDIR ./FOCUS
ENV PIP_ROOT_USER_ACTION=ignore
RUN echo "$PWD"
RUN pip install -r ./requirements.txt
RUN pip install dlib
CMD ls
CMD python ./show_instructions.py
