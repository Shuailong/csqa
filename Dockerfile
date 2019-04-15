FROM allennlp/allennlp:v0.8.3

# install source code
ADD ./ /root/csqa/
WORKDIR /root/csqa/
RUN pip install -r requirements.txt
