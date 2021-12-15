# Sketch-to-Image-Model-by-GAN
this project was finished by me and my team member Yingning Ma. 

Reference：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

training data: https://github.com/Malikanhar/Face-Sketch-to-Image-Generation-using-GAN

## this project is based on CycleGAN, and we use self-attention to improve our results
The generated image is basically similar to a real image. Compared with the predicted image, some details are closer to the sketch. With the increase of iteration, the loss are decreases. It shows our model is more and more exact.
![image](https://user-images.githubusercontent.com/79315517/146282571-52c4dae6-81ed-4782-8a60-6ab066d06d40.png)

Based on cycleGAN, this model is not limited to the conversion of sketches to images. From images to sketches are also achievable. And any two sets of corresponding images can be used to transform. For example, apple to orange. with a large enough capacity, the network can map the same set of input images to any random arrangement of images in the target domain. Therefore, opposing Loss alone does not guarantee that a single input can be mapped.
![image](https://user-images.githubusercontent.com/79315517/146282711-b55473eb-dcdf-416f-bac0-f638a59a6dbe.png)
