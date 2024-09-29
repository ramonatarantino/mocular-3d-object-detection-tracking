# Omni3D with Custom Object Tracker

**Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild**  
This repository extends the Omni3D model by integrating a custom object tracking mechanism, enhancing 3D detection with continuous tracking across frames.

## Table of Contents:
1. [Overview](#overview)
2. [Installation](#installation)
3. [Running the Demo](#demo)
4. [Training](#training)
5. [Inference](#inference)
6. [Tracker Implementation](#tracker)
7. [Citing Omni3D](#citing)

## Overview <a name="overview"></a>

Omni3D, originally developed by [Garrick Brazil et al.](https://garrickbrazil.com/omni3d), is a state-of-the-art model for 3D object detection. This project incorporates a custom tracking mechanism to extend the detection capabilities, enabling real-time object tracking in various environments.

For more details on the Omni3D project, refer to the [original repository](https://garrickbrazil.com/omni3d).

## Installation <a name="installation"></a>

Follow the steps below to set up the environment:

```bash
# Create and activate a new conda environment
conda create -n cubercnn python=3.8
source activate cubercnn

# Install main dependencies
conda install -c fvcore -c iopath -c conda-forge -c pytorch3d -c pytorch fvcore iopath pytorch3d pytorch=1.8 torchvision=0.9.1 cudatoolkit=10.1

# Install additional dependencies
pip install cython opencv-python
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
conda install -c conda-forge scipy seaborn
```

## Running the Demo <a name="demo"></a>

You can run the Omni3D demo with tracking enabled using the following commands:

```bash
# Download sample images
sh demo/download_demo_COCO_images.sh

# Run the demo with the custom tracker
python demo/demo_detection.py \
--config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
--input-folder "datasets/coco_examples" \
--threshold 0.25 --display \
MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
OUTPUT_DIR output/demo_with_tracking
```


## Running the Demo with Tracker <a name="demo"></a>

You can run the Omni3D demo with tracking enabled using the following commands:

```bash
# Download sample images
sh demo/download_demo_COCO_images.sh

# Run the demo with the custom tracker
python demo/demo_tracker.py \
--config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
--input-video "demo/video_indoor2.mp4" \
--threshold 0.40 --display 

```


## Training <a name="training"></a>

To train the Omni3D model with tracking, use the same Omni3D setup as described in the [original Omni3D repository](https://garrickbrazil.com/omni3d), with the following command:

```bash
python tools/train_net.py \
  --config-file configs/Base_Omni3D.yaml \
  OUTPUT_DIR output/omni3d_with_tracking
```

## Inference <a name="inference"></a>

To run inference, use:

```bash
python tools/train_net.py \
  --eval-only --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
  MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
  OUTPUT_DIR output/evaluation
```

## Tracker Implementation <a name="tracker"></a>

This project adds an object tracker to the original Omni3D model, allowing for multi-frame tracking of detected objects. The tracker matches detected objects across frames based on a custom algorithm that utilizes 3D bounding box information, object centers, and category types.

Key features of the tracker:
- Custom matching logic for continuous tracking of objects over time
- Handling of high-cost matches across consecutive frames
- Integration with Omni3D’s 3D detection pipeline


# Chat with Phi-3 Vision

This folder is part of the **Monocular 3D Object Detection and Tracking** project and contains a Streamlit-based application that allows users to interact with a vision model, specifically Phi-3 Vision, to analyze images and return detailed descriptions of their contents.

## Overview

The application is designed to:
- Receive images from a server via a socket connection.
- Allow users to enter specific requests regarding the description of the images.
- Leverage the Phi-3 Vision model to generate accurate and detailed image descriptions.

Key features:
- **Real-time communication** with a server for image acquisition.
- **Text-based interaction** for requesting specific details about an image.
- **Automatic handling of inputs**, where image frames are fetched automatically upon submission of a user query.
- **Customizable chat interface**, which automatically resets after each response.
- **Responsive interface** that adapts to the user's input with minimal interaction, ensuring a smooth user experience.

## Installation

To set up and run the application, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ramonatarantino/mocular-3d-object-detection-tracking.git
   cd mocular-3d-object-detection-tracking/chat-with-phi-3-vision
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Launch the Streamlit app by running the following command:
   ```bash
   streamlit run app.py
   ```

4. **Start the server:**
   Make sure the server providing the image stream is running and accessible.

## Usage

1. After launching the app, you'll see a chat interface where you can submit queries about the images being streamed.
2. The Phi-3 Vision model will analyze the current frame and provide a detailed response based on the query.
3. You can refine your requests in natural language, and the app will process the input and update with a new description.

## Customization

### Input Behavior
- The input bar is fixed to the bottom of the interface for easy access. It automatically fetches a new frame when you hit "Enter" after typing your query.

### Image Processing
- Images are processed in real-time as they are received from the server. There is no need to manually click any buttons—everything happens seamlessly after a user submits their request.

### Hidden Features
- Advanced settings are embedded in the code but not visible on the user interface, keeping the layout clean and user-friendly.



## Citing Omni3D <a name="citing"></a>

Please cite the original Omni3D paper if you use this project in your research:

```BibTeX
@inproceedings{brazil2023omni3d,
  author =       {Garrick Brazil and Abhinav Kumar and Julian Straub and Nikhila Ravi and Justin Johnson and Georgia Gkioxari},
  title =        {{Omni3D}: A Large Benchmark and Model for {3D} Object Detection in the Wild},
  booktitle =    {CVPR},
  address =      {Vancouver, Canada},
  month =        {June},
  year =         {2023},
  organization = {IEEE},
}
```

## License (Cube R-CNN) <a name="license"></a>
Cube R-CNN is released under [CC-BY-NC 4.0](LICENSE.md).


Additionally, if you use the Omni3D benchmark, please cite the datasets listed in the original Omni3D repository.

## Citing <a name="citing"></a>

Please use the following BibTeX entry if you use Omni3D and/or Cube R-CNN in your research or refer to our results.

```BibTeX
@inproceedings{brazil2023omni3d,
  author =       {Garrick Brazil and Abhinav Kumar and Julian Straub and Nikhila Ravi and Justin Johnson and Georgia Gkioxari},
  title =        {{Omni3D}: A Large Benchmark and Model for {3D} Object Detection in the Wild},
  booktitle =    {CVPR},
  address =      {Vancouver, Canada},
  month =        {June},
  year =         {2023},
  organization = {IEEE},
}
```

If you use the Omni3D benchmark, we kindly ask you to additionally cite all datasets. BibTex entries are provided below.

<details><summary>Dataset BibTex</summary>

```BibTex
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR},
  year = {2012}
}
``` 

```BibTex
@inproceedings{caesar2020nuscenes,
  title={nuscenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={CVPR},
  year={2020}
}
```

```BibTex
@inproceedings{song2015sun,
  title={Sun rgb-d: A rgb-d scene understanding benchmark suite},
  author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
  booktitle={CVPR},
  year={2015}
}
```

```BibTex
@inproceedings{dehghan2021arkitscenes,
  title={{ARK}itScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile {RGB}-D Data},
  author={Gilad Baruch and Zhuoyuan Chen and Afshin Dehghan and Tal Dimry and Yuri Feigin and Peter Fu and Thomas Gebauer and Brandon Joffe and Daniel Kurz and Arik Schwartz and Elad Shulman},
  booktitle={NeurIPS Datasets and Benchmarks Track (Round 1)},
  year={2021},
}
```

```BibTex
@inproceedings{hypersim,
  author    = {Mike Roberts AND Jason Ramapuram AND Anurag Ranjan AND Atulit Kumar AND
                 Miguel Angel Bautista AND Nathan Paczan AND Russ Webb AND Joshua M. Susskind},
  title     = {{Hypersim}: {A} Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding},
  booktitle = {ICCV},
  year      = {2021},
}
```

```BibTex
@article{objectron2021,
  title={Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations},
  author={Ahmadyan, Adel and Zhang, Liangkai and Ablavatski, Artsiom and Wei, Jianing and Grundmann, Matthias},
  journal={CVPR},
  year={2021},
}
```

</details>


## License (Chat with Phi 3 Vision)

This project is licensed under the MIT License.  
The server code is based on the work from the following repository:  
[https://github.com/bhimrazy/chat-with-phi-3-vision](https://github.com/bhimrazy/chat-with-phi-3-vision)

## Contributing

I welcome contributions! If you'd like to add features, improve the documentation, or report a bug, feel free to open an issue or submit a pull request.
