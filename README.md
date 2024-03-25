# QA-LLM-RAG

This is a microservice with Vector Database and Embedding MOdel

You will be able to load documents to vector DB.

You will be be able to ask chat questions and get answers based on docs stored in vector DB.

Postman Collection is inculded in the test folder.

## How to run?

In main folder:

```bash
 docker-compose up --build
```

This creates a docker container which handles the swagger page on:

http://127.0.0.1:8100/llm-rag/docs

## Workflow

![Alt text](image.png)

![Alt text](image-1.png)

## LLM

Google Flan T5 XL
