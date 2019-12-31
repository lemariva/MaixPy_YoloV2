# MaixPy_YoloV2
This repository helps you to extend the models is to detect objects using [YOLO-V2](https://pjreddie.com/media/files/papers/YOLO9000.pdf) on a MaixPY

# DIY
Install Docker on your machine and deploy the `tensorflow/tensorflow:latest-py3-jupyter` using:
```
$ curl -sSL https://get.docker.com | sh

$ sudo docker run -d -p 8888:8888 tensorflow/tensorflow:latest-py3-jupyter
```
Open the following URL in your host web browser: `http://localhost:8888`

You need a token to log in. The token is inside the container. List the container to get the ID with following command:

```
$ docker container ls
CONTAINER ID        IMAGE    [....]
5082a85283bb        tensorflow/  [....]
```

Then, read the logs typing:

```
$ docker logs 5082a85283bb

[...]
The Jupyter Notebook is running at:
[...] http://ac64d540a1cb:8888/?token=df050fa3b53de5f9203ca862e5f3656962b665dc224243a8
[...]
```
The hash after `token=` is the token to log in.

Download this repository and upload it under `/notebooks/`

I added a training example with the brio 33594. You can train the model running the code inside `training.ipynb`. 

Additional information about the porting to MaixPy is coming this week.

Note: Some additional libraries for the container are required and installed on the first cell block of the notebook. You don't need to run it everytime that you compile the model. If you start a new container (not restart the stopped one), you need to install them again.

# Acknowledgement
* Ported from [penny4860/Yolo-digit-detector](https://github.com/penny4860/Yolo-digit-detector) to Jupyter and upgraded to Tensorflow 2.0. ```