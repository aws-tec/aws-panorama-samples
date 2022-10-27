FROM public.ecr.aws/panorama/panorama-application
COPY src /panorama
RUN pip3 install opencv-python boto3
RUN python3 -m pip install awsiotsdk
RUN python3 -m pip install sympy
RUN python3 -m pip install pyzmq