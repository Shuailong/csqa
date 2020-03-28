FROM allennlp/allennlp:v0.8.3

WORKDIR /root/csqa/
ADD ./requirements.txt /root/csqa
RUN pip install -r requirements.txt
ADD ./csqa/ /root/csqa/csqa