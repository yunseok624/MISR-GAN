# MISR-GAN

## Model
The proposed model is an adaptation of ESRGAN with CA (Coordinate Attention) module adapted for an input as stack of multiple LR images

## Datasets

### PROBA-V

### S2-NAIP

## Training
To train a model on given dataset, run the following command, with the desired configuration file:

`python -m ssr.train -opt ssr/options/*.yml`

There are several sample configuration files in `ssr/options/`. Make sure the configuration file specifies 
correct paths to your downloaded data, the desired number of low-resolution input images, model parameters, 
and pretrained weights (if applicable).

Training process step:
1. Pre-training the model to minimize the pixel loss
2. GAN training the model with total loss (pixel + perceptual + adversarial)

## Testing
To evaluate the model on a test set run the following command, with the desired configuration file:

`python -m ssr.test -opt ssr/options/*.yml`

## Results

## TODO (Future works)
- [ ] Finish the README.md
- [ ] Upload the final version of the code
- [ ] Upload the weights
- [ ] Train on bigger number of iterations
- [ ] Set learning rate scheduler
- [ ] Run on multiple GPUs for faster computation

## Acknowledgements
Thanks to these codebases for foundational Super-Resolution code and inspiration:

[BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master})

[ESRGAN](https://github.com/xinntao/ESRGAN/tree/master)

[Coordinate Attention](https://github.com/houqb/CoordAttention)

## Contact
If you have any questions, please email `yunseok.park@skoltech.ru` or [open an issue](https://github.com/yunseok624/MISR-GAN/issues/new).
