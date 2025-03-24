# Use the AWS base image for python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# COPY the requirements.txt file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# INSTALL the libs in requirements.txt file
RUN pip install -r requirements.txt

#COPY pdf file
COPY lol.pdf ${LAMBDA_TASK_ROOT}/lol.pdf

#COPY the function code
COPY embed_book.py ${LAMBDA_TASK_ROOT}

#SET permission to make the file executable
RUN chmod +x embed_book.py

#SET CMD to your handler lambda
CMD ["embed_book.lambda_handler"]