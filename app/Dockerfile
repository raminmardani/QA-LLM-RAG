# Use Python 3.11 image from the bullseye distribution as the base image
FROM python:3.11-bullseye

# Add Microsoft packages GPG key and repository for SQL Server ODBC Driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
  curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install system dependencies including ffmpeg, libsm6, libxext6, and msodbcsql17
RUN apt-get update && \
  apt-get install ffmpeg libsm6 libxext6 -y && \
  ACCEPT_EULA=Y apt-get install -y msodbcsql17 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Copy Java Runtime Environment (JRE) from openjdk:8-jre-slim image
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/local/openjdk-8

# Install Java for LibreOffice and update alternatives for Java
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1 && \
  apt-get update && \
  apt-get --no-install-recommends install libreoffice -y && \
  apt-get install -y libreoffice-java-common && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY ./app/requirements.txt .
RUN pip install -r requirements.txt

# Set the working directory to /app
WORKDIR /app

# Copy the application code to the container
COPY /app .

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8100"]