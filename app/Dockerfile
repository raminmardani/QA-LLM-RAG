FROM python:3.11-bullseye

# Add Microsoft packages GPG key
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Add the Microsoft package repository
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Continue with the rest of your Dockerfile
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17


COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME /usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1
RUN apt-get --no-install-recommends install libreoffice -y
RUN apt-get install -y libreoffice-java-common

COPY ./app/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY /app .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8100"]