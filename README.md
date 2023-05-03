<!-- # Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer. ICCV2021. -->

<!-- ## Introduction
We proposed a novel model training paradigm for few-shot semantic segmentation. Instead of meta-learning the whole, complex segmentation model, we focus on the simplest
classifier part to make new-class adaptation more tractable. Also, a novel meta-learning algorithm that leverages a Classifier Weight Transformer (CWT) for adapting dynamically the classifier weights to every query sample is introduced to eliminate the impact of intra-class discripency.  -->

<!-- ## Architecture
<a href="url"><img src="https://github.com/zhiheLu/CWT-for-FSS/blob/main/doc/framework.jpg" align="center" height="350" width="900" ></a> -->

## Environment
Other configurations can also work, just make sure you have this package.
- einops==0.4.1

## Running the testing
Run the below in the command line and you should get the result
```python
python -m simvptest --root {dir of the hidden dataset}
```
- This is an example with hidden dataset is stored at '../../../dataset/dl/hidden'
```python
python -m simvptest --root ../../../dataset/dl/hidden
```



<!-- ## Contact
Please write down issues or contact me via zhihe.lu [at] surrey.ac.uk if you have any questions. -->


<!-- ## Acknowledgments
Thanks to the code contributors. Some parts of code are borrowed from https://github.com/Jia-Research-Lab/PFENet and https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation. -->