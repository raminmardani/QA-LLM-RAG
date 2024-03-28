#test this
#FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM python:3.11-bullseye
#RUN apt-get update && \
#  apt-get install -y python3-pip python3-dev && \
#  rm -rf /var/lib/apt/lists/*


# Add Microsoft packages GPG key
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Add the Microsoft package repository
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Continue with the rest of your Dockerfile
RUN apt-get -y update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
#RUN apt-get -y update && apt-get install -y curl gnupg 
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install poppler-utils -y
#RUN apt-get update \
#  && apt-get -y install tesseract-ocr && apt-get -y install ffmpeg libsm6 libxext6 # required for opencv
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8

ENV JAVA_HOME /usr/local/openjdk-8

RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1

RUN apt-get --no-install-recommends install libreoffice -y
RUN apt-get install -y libreoffice-java-common



# Method2 - additionally installing unoconv
RUN apt-get install unoconv
# download appropriate package for the OS version
# Debian 11
RUN curl https://packages.microsoft.com/config/debian/11/prod.list  \
  > /etc/apt/sources.list.d/mssql-release.list

RUN exit
RUN apt-get -y update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
COPY ./app/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install bitsandbytes
WORKDIR /app

COPY /app .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8100"]