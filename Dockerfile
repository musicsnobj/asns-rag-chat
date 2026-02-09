FROM public.ecr.aws/lambda/python:3.13

# Install your deps
RUN pip install --no-cache-dir pydantic anthropic instructor

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the handler (module.function)
CMD ["app.lambda_handler"]
