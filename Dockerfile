FROM python:3.9.5
COPY . .
RUN apt-get update

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir os vtk shutil numpy nibabel pathlib && \
    pip3 install --no-cache-dir sklearn plotly.express matplotlib.pyplot && \ 
    pip3 install --no-cache-dir nipype.interfaces plotly.graph_objects && \ 
    pip3 install --no-cache-dir sklearn.cluster nipype.testing && \
    pip3 install --no-cache-dir nipype.testing mpl_toolkits.mplot3d
    
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -r requirements.txt

WORKDIR /BRAIN/Scripts
RUN pwd
