FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_server.py .
COPY agent_lambda.py .


# FastAPI + Mangum adapter
RUN pip install --no-cache-dir fastapi mangum uvicorn

# Lambda entry: handler = app via Mangum
ENV AWS_REGION=us-east-2

CMD [ "agent_lambda.handler" ]
