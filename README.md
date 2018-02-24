# Machine_Learning_Interface
Provides common interface to popular python machine learning and statistics packages

python tests.py #Runs testing suite, svc currently fails

#To use docker in cmd, run these commands. Otherwise use docker quickstart terminal
#https://runnable.com/docker/python/dockerize-your-python-application
docker-machine env --shell cmd default
@FOR /f "tokens=*" %i IN ('docker-machine env --shell cmd default') DO @%i

#Building and running in docker
#Run from directory containing Dockerfile to build docker image.
docker build --no-cache -t mli:latest .
docker run -i -t mli #Run docker container

#Docker in jupyter notebook
docker run -it -p 8888:8888 mli
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root #Run from within docker image, token created

#Getting docker IP for jupyter
docker-machine env #Look in Docker Host
IP:8888/token=TOKEN FROM DOCKER

#Documentation
#Go to doc directory
make html