# MA_SridharVeena

## Deep Learning Based Identification of Recurring Load Paths from Topology Optimized Structures 

Topology Optimization (TO) can produce varying structural solutions for the same railway car
body component depending on the applied loads, boundary conditions and optimization objectives.
Identifying recurring load paths across these solutions can support their interpretation and could
provide an improved initialization for further optimizations. This thesis investigates the automated
extraction of such recurring load paths by formulating the problem as the prediction of a single
consensus structure from a group of TO results.
A dataset was curated from 2D images of TO results and organized into groups with corresponding
manually defined consensus targets. Convolutional Neural Network (CNN) was investigated for
consensus extraction directly in the image domain. Additionally, Variational Autoencoder (VAE)
with a latent aggregation module was investigated to aggregate input representations in the latent
space.
The proposed approaches achieved similar average centerline Dice (clDice) performance with
a relative improvement of 4.8% on the test set over the overall best-performing pixel-wise averaging
baseline. The results demonstrate the potential of Deep Learning (DL) methods for
extracting recurring structural patterns from multiple TO solutions and provide a basis for further
development of automated consensus identification methods.

To run the program:
```python
python main.py
```
